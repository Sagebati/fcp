
use clap::Parser;
use ecow::EcoString;
use enclose::enclose;
use fcp::{stage_route, CopyParams, Res};
use indicatif::ProgressStyle;
use pumps::{Concurrency, Pipeline};
use std::path::PathBuf;
use std::sync::Arc;
use derive_more::Display;
use tracing::level_filters::LevelFilter;
use tracing::{info_span, warn};
use tracing_indicatif::span_ext::IndicatifSpanExt;
use tracing_indicatif::IndicatifLayer;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::EnvFilter;


#[derive(Display, Clone, Debug, FromStr)]
#[display("{_0:?}")]
pub struct ImageExtension(Vec<EcoString>);

fn ext_default() -> ImageExtension {
    ImageExtension(vec![
        EcoString::from("raf"),
        EcoString::from("RAF"),
        //EcoString::from("jpg"),
        //EcoString::from("JPG"),
        EcoString::from("NEF"),
        EcoString::from("nef"),
    ])
}


fn available_parallelism() -> usize {
    std::thread::available_parallelism()
        .unwrap_or_else(|_| 4.try_into().unwrap())
        .get()
}
#[derive(Parser, Clone)]
#[command(version, about, long_about = None)]
pub struct Args {
    pub from: PathBuf,
    pub dest: PathBuf,
    #[arg(short, default_value_t = ext_default())]
    pub image_extensions: ImageExtension,
    #[arg(short)]
    pub ignore_extensions: Vec<EcoString>,
    #[arg(short, default_value_t = todo!())]
    pub path_format: EcoString,
    /// Uses Hard Links instead of copying
    #[arg(long)]
    pub ln: bool,
    #[arg(long)]
    pub dry: bool,
    #[arg(short, long, default_value_t = false)]
    pub force: bool,
    #[arg(short = 'j', long, default_value_t = available_parallelism())]
    pub concurrency_limit: usize,
}

impl From<Args> for CopyParams {
    fn from(value: Args) -> Self {
        Self {
            from: value.from,
            dest: value.dest,
            image_extensions: value.image_extensions.0,
            ignore_extensions: value.ignore_extensions,
            path_format: value.path_format,
            dry: value.dry,
            hard_links: value.ln,
            hash: false,
            force: false,
            concurrency_limit: value.concurrency_limit,
        }
    }
}

fn main() -> Res {
    tokio::runtime::Builder::new_multi_thread()
        .enable_io_uring()
        .enable_all()
        .build()?
        .block_on(async_main())
}

async fn async_main() -> Res {
    let cli_args = Args::parse();

    let env_filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();

    let indicative_layer = IndicatifLayer::new();

    let reg = tracing_subscriber::registry()
        .with(env_filter)
        .with(tracing_subscriber::fmt::layer().with_writer(indicative_layer.get_stderr_writer()))
        .with(indicative_layer);

    #[cfg(feature = "debug")]
    let reg = reg.with(console_subscriber::ConsoleLayer::builder().spawn());

    let configuration = Arc::new(cli_args.into());

    reg.init();

    let io_parallelism = available_parallelism() * 2;

    let global_span = info_span!("fpd");
    global_span.pb_set_style(&ProgressStyle::default_bar());
    global_span.pb_set_length(0);
    let _global_enter = global_span.enter();

    let (mut rx, _handle) = Pipeline::from_stream(fcp::stage_scan(
        Arc::clone(&configuration),
        global_span.clone(),
    ))
    .map(
        |path| fcp::stage_load_meta(path),
        Concurrency::concurrent_ordered(io_parallelism),
    )
    .filter_map(
        enclose!((configuration)
            move |result| stage_route(result, Arc::clone(&configuration))
        ),
        Concurrency::concurrent_ordered(io_parallelism),
    )
    .map(
        move |(photo, new_path)| fcp::stage_copy(photo, new_path, Arc::clone(&configuration)),
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
