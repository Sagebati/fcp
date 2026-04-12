mod tagger;

use anyhow::Context;
use bytes::Bytes;
use clap::Parser;
use fpd::{
    compute_new_path, copy, init_gpu, load_image_for_clip, scan_library_paths,
    ClipTaggerManager, ClipTaggerPool,
    Configuration, FileIgnoredReason, Photo, Res,
};
use futures::channel::mpsc::{unbounded, UnboundedReceiver};
use indicatif::ProgressStyle;
use pumps::{Concurrency, Pipeline};
use std::future::{ready, Future};
use std::path::PathBuf;
use std::sync::Arc;
use std::thread::available_parallelism;
use enclose::enclose;
use tagger::{Cli, Commands};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use tokio::task::spawn_blocking;
use tracing::level_filters::LevelFilter;
use tracing::{debug, info_span, instrument, warn, Span};
use tracing_chrome::ChromeLayerBuilder;
use tracing_indicatif::span_ext::IndicatifSpanExt;
use tracing_indicatif::IndicatifLayer;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::EnvFilter;

const DEFAULT_TAGS: &[&str] = &[
    "a photo of a cat",
    "a photo of a dog",
    "a landscape",
    "a portrait",
];

// --- pipeline stages ---

fn stage_scan(config: Arc<Configuration>, progress: Span) -> UnboundedReceiver<PathBuf> {
    let (tx, rx) = unbounded();
    let scan_span = info_span!("scan", root = ?config.from);
    spawn_blocking(move || {
        let _enter = scan_span.enter();
        let mut count = 0u64;
        for (i, path) in scan_library_paths(&config).enumerate() {
            if tx.unbounded_send(path).is_err() {
                break;
            }
            count = i as u64;
            if i % 10 == 0 {
                progress.pb_set_length(count);
            }
        }
        progress.pb_set_length(count + 1);
    });
    rx
}

#[instrument(skip_all, fields(file = ?path.file_name()))]
async fn stage_load_meta(path: PathBuf) -> Res<Photo> {
    Photo::new(path).load_meta().await
}

#[instrument(skip_all)]
fn stage_route(
    result: Res<Photo>,
    config: Arc<Configuration>,
) -> impl Future<Output = Option<(Photo, PathBuf)>> {
    async move {
        match result {
            Ok(photo) => {
                let new_path = compute_new_path(&config.dest, &config.path_format, &photo);
                if !new_path.exists() || config.force {
                    Some((photo, new_path))
                } else {
                    debug!(
                        file_path = ?photo.original_path(),
                        ignored = true,
                        reason = %FileIgnoredReason::FileAlreadyExists,
                    );
                    None
                }
            }
            Err(e) => {
                warn!("{e:?}");
                None
            }
        }
    }
}

#[instrument(skip_all, fields(batch_size = batch.len()))]
async fn stage_tag_batch(
    batch: Vec<(Photo, PathBuf)>,
    pool: Option<Arc<ClipTaggerPool>>,
) -> Vec<(Photo, PathBuf)> {
    let Some(pool) = pool else {
        return batch;
    };

    // Load bytes for all photos concurrently.
    let bytes_per_photo: Vec<Option<Bytes>> =
        futures::future::join_all(batch.iter().map(|(photo, _)| photo.bytes()))
            .await
            .into_iter()
            .map(|r| r.map(|b| b.clone()).map_err(|e| { warn!("Failed to load bytes: {e:?}"); e }).ok())
            .collect();

    let mut tagger = match pool.get().await {
        Ok(t) => t,
        Err(e) => {
            warn!("Failed to acquire tagger from pool: {e:?}");
            return batch;
        }
    };

    let span = tracing::Span::current();
    let result = spawn_blocking(move || {
        let _enter = span.enter();
        // Decode images in parallel using rayon.
        let decoded: Vec<(usize, image::DynamicImage)> = bytes_per_photo
            .into_par_iter()
            .enumerate()
            .filter_map(|(i, maybe)| {
                let b = maybe?;
                load_image_for_clip(&b[..]).ok().map(|img| (i, img))
            })
            .collect();

        if decoded.is_empty() {
            return Ok(vec![]);
        }

        let indices: Vec<usize> = decoded.iter().map(|(i, _)| *i).collect();
        let images: Vec<image::DynamicImage> = decoded.into_iter().map(|(_, img)| img).collect();
        let tags: Vec<String> = DEFAULT_TAGS.iter().map(|s| s.to_string()).collect();

        let tag_results = tagger.predict_batch(&images, &tags, 0.2)?;

        Ok::<Vec<(usize, Vec<String>)>, anyhow::Error>(
            indices.into_iter().zip(tag_results).collect(),
        )
    })
    .await;

    match result {
        Ok(Ok(tagged)) => {
            for (idx, tags) in &tagged {
                debug!(file = ?batch[*idx].0.original_path().file_name(), ?tags);
            }
        }
        Ok(Err(e)) => warn!("Batch tagging failed: {e:?}"),
        Err(e) => warn!("Batch tagger task panicked: {e:?}"),
    }

    batch
}

#[instrument(skip_all, fields(file = ?photo.original_path().file_name()))]
async fn stage_copy(photo: Photo, new_path: PathBuf, config: Arc<Configuration>) -> Res<()> {
    copy(photo.original_path(), &new_path, &config)
        .await
        .context(format!(
            "Error occurred when copying {:?}",
            photo.original_path()
        ))
}

// --- main ---

fn main() -> Res {
    tokio::runtime::Builder::new_multi_thread()
        .enable_io_uring()
        .enable_all()
        .build()?
        .block_on(async_main())
}

async fn async_main() -> Res {
    let cli = Cli::parse();

    let env_filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();

    let indicatif_layer = IndicatifLayer::new();

    let (chrome_layer, _chrome_guard) = cli
        .trace
        .as_deref()
        .map(|path| ChromeLayerBuilder::new().file(path).build())
        .unzip();

    let reg = tracing_subscriber::registry()
        .with(env_filter)
        .with(tracing_subscriber::fmt::layer().with_writer(indicatif_layer.get_stderr_writer()))
        .with(indicatif_layer)
        .with(chrome_layer);

    #[cfg(feature = "debug")]
    let reg = reg.with(console_subscriber::ConsoleLayer::builder().spawn());

    reg.init();

    if let Some(command) = cli.command {
        match command {
            Commands::Tag(tag_args) => {
                return tagger::run_tagger(tag_args).await;
            }
        }
    }

    let tag_batch_size = cli.tag_batch_size;
    let configuration = Arc::new(cli.config);

    if let Err(e) = init_gpu() {
        warn!("ROCm unavailable, falling back to CPU: {e:?}");
    }

    let cpu_parallelism = available_parallelism()
        .unwrap_or_else(|_| 4.try_into().unwrap())
        .get();
    let io_parallelism = configuration.concurrency_limit;

    let tagger_pool = match deadpool::managed::Pool::builder(ClipTaggerManager)
        .max_size(cpu_parallelism)
        .build()
    {
        Ok(pool) => {
            tracing::info!("tagger pool ready (max_size={cpu_parallelism})");
            Some(Arc::new(pool))
        }
        Err(e) => {
            warn!("Tagger pool unavailable: {e:?}");
            None
        }
    };

    let global_span = info_span!("fpd");
    global_span.pb_set_style(&ProgressStyle::default_bar());
    global_span.pb_set_length(0);
    let _global_enter = global_span.enter();

    let (mut rx, _handle) =
        Pipeline::from_stream(stage_scan(Arc::clone(&configuration), global_span.clone()))
            .map(
                |path| stage_load_meta(path),
                Concurrency::concurrent_ordered(io_parallelism),
            )
            .filter_map(
                enclose!((configuration)
                    move |result| stage_route(result, Arc::clone(&configuration))
                ),
                Concurrency::concurrent_ordered(io_parallelism),
            )
            .batch(tag_batch_size)
            .map(
                enclose!(
                    (tagger_pool)
                    move |batch| stage_tag_batch(batch, tagger_pool.clone())
                ),
                Concurrency::concurrent_ordered(cpu_parallelism),
            )
            .map(|batch| ready(batch), Concurrency::serial())
            .flatten_iter()
            .map(
                move |(photo, new_path)| stage_copy(photo, new_path, Arc::clone(&configuration)),
                Concurrency::concurrent_ordered(io_parallelism),
            )
            .build();

    while let Some(result) = rx.recv().await {
        let global_span = global_span.clone();
        match result {
            Ok(()) => global_span.pb_inc(1),
            Err(e) => warn!("{e:?}"),
        }
    }

    Ok(())
}
