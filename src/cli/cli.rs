
use clap::Parser;
use ecow::EcoString;
use enclose::enclose;
use fcp::{stage_route, CopyParams, Res};
use indicatif::ProgressStyle;
use pumps::{Concurrency, Pipeline};
use std::convert::Infallible;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;
use derive_more::Display;
use tracing::level_filters::LevelFilter;
use tracing::{info_span, warn};
use tracing_chrome::ChromeLayerBuilder;
use tracing_indicatif::span_ext::IndicatifSpanExt;
use tracing_indicatif::IndicatifLayer;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::EnvFilter;

static DEFAULT_DEST: &str = "{{year}}/{{year}}_{{month}}_{{day}}/{{original}}";

#[derive(Display, Clone, Debug)]
#[display("{}", _0.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(","))]
pub struct ImageExtension(Vec<EcoString>);

impl FromStr for ImageExtension {
    type Err = Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(ImageExtension(
            s.split(',').map(EcoString::from).collect(),
        ))
    }
}

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
    #[arg(long, default_value_t = ext_default())]
    pub image_extensions: ImageExtension,
    #[arg(short)]
    pub ignore_extensions: Vec<EcoString>,
    #[arg(short, default_value_t = DEFAULT_DEST.into())]
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
    /// Write a Chrome tracing JSON file to PATH (load in chrome://tracing or ui.perfetto.dev)
    #[arg(long, value_name = "PATH")]
    pub trace: Option<PathBuf>,
    /// Override the path for the per-file CSV report. Defaults to a file in the system temp dir.
    #[arg(long, value_name = "PATH")]
    pub report: Option<PathBuf>,
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
            force: value.force,
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

    let (chrome_layer, _chrome_guard) = cli_args
        .trace
        .as_deref()
        .map(|path| ChromeLayerBuilder::new().file(path).include_args(true).build())
        .unzip();

    let reg = tracing_subscriber::registry()
        .with(env_filter)
        .with(tracing_subscriber::fmt::layer().with_writer(indicative_layer.get_stderr_writer()))
        .with(indicative_layer)
        .with(chrome_layer);

    #[cfg(feature = "debug")]
    let reg = reg.with(console_subscriber::ConsoleLayer::builder().spawn());

    let report_path = cli_args.report.clone().unwrap_or_else(|| {
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        std::env::temp_dir().join(format!("fcp-report-{ts}.csv"))
    });

    let configuration: Arc<fcp::CopyParams> = Arc::new(cli_args.into());

    reg.init();

    let (reporter, reporter_task) = fcp::start_reporter(report_path)?;

    let io_parallelism = configuration.concurrency_limit;

    let global_span = info_span!("fpd");
    global_span.pb_set_style(&ProgressStyle::default_bar());
    global_span.pb_set_length(0);
    let _global_enter = global_span.enter();

    let (mut rx, _handle) = Pipeline::from_stream(fcp::stage_scan(
        Arc::clone(&configuration),
        global_span.clone(),
    ))
    .filter_map(
        enclose!((reporter) move |path| fcp::stage_load_meta(path, Arc::clone(&reporter))),
        Concurrency::concurrent_ordered(io_parallelism),
    )
    .filter_map(
        enclose!((configuration, reporter)
            move |photo| stage_route(photo, Arc::clone(&configuration), Arc::clone(&reporter))
        ),
        Concurrency::concurrent_ordered(io_parallelism),
    )
    .map(
        enclose!((configuration, reporter)
            move |(photo, new_path)| fcp::stage_copy(photo, new_path, Arc::clone(&configuration), Arc::clone(&reporter))
        ),
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

    drop(reporter);
    let summary = reporter_task.join.await??;
    eprintln!(
        "{} copied, {} hard-linked, {} dry-run, {} skipped, {} errors — report: {}",
        summary.copied,
        summary.hard_linked,
        summary.dry_run,
        summary.skipped,
        summary.meta_error + summary.copy_error,
        reporter_task.path.display(),
    );

    Ok(())
}
