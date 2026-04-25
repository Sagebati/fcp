mod tagger;

use anyhow::Context;
use bytes::Bytes;
use clap::Parser;
use enclose::enclose;
use fpd::clip::clip_pool;
use fpd::{
    compute_new_path, copy, init_gpu, load_image_for_clip, scan_library_paths, ClipTaggerManager,
    ClipTaggerPool, Configuration, FileIgnoredReason, Photo, Res,
};
use futures::channel::mpsc::{unbounded, UnboundedReceiver};
use indicatif::ProgressStyle;
use pumps::{Concurrency, Pipeline};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use std::future::{ready, Future};
use std::path::PathBuf;
use std::sync::Arc;
use std::thread::available_parallelism;
use tagger::{Cli, Commands};
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

    let cpu_parallelism = available_parallelism()
        .unwrap_or_else(|_| 4.try_into().unwrap())
        .get();

    let io_parallelism = configuration.concurrency_limit;

    let tagger_pool = clip_pool(cpu_parallelism).expect();

    let global_span = info_span!("fpd");
    global_span.pb_set_style(&ProgressStyle::default_bar());
    global_span.pb_set_length(0);
    let _global_enter = global_span.enter();

    let (mut rx, _handle) = Pipeline::from_stream(fpd::stage_scan(
        Arc::clone(&configuration),
        global_span.clone(),
    ))
    .map(
        |path| fpd::stage_load_meta(path),
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
        move |(photo, new_path)| fpd::stage_copy(photo, new_path, Arc::clone(&configuration)),
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
