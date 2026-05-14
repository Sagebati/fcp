use clap::{ArgAction, Parser};
use derive_more::Display;
use ecow::EcoString;
use enclose::enclose;
use fcp::{stage_dedup, stage_route, CopyParams, DedupIndex, Res};
use indicatif::ProgressStyle;
use pumps::{Concurrency, Pipeline};
use std::convert::Infallible;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;
use tracing::level_filters::LevelFilter;
use tracing::{info_span, warn};
use tracing_chrome::ChromeLayerBuilder;
use tracing_indicatif::span_ext::IndicatifSpanExt;
use tracing_indicatif::IndicatifLayer;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::EnvFilter;

static DEFAULT_DEST: &str = "{{year}}/{{year}}-{{month}}-{{day}}/{{month}}/{{original}}";

#[derive(Display, Clone, Debug)]
#[display("{}", _0.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(","))]
pub struct ImageExtension(Vec<EcoString>);

impl FromStr for ImageExtension {
    type Err = Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(ImageExtension(s.split(',').map(EcoString::from).collect()))
    }
}

fn ext_default() -> ImageExtension {
    ImageExtension(
        fcp::extensions::DEFAULT_PHOTO_EXTENSIONS
            .iter()
            .copied()
            .map(EcoString::from)
            .collect(),
    )
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
    /// Increase log verbosity. -v = debug, -vv = trace. Overridden by RUST_LOG if set.
    #[arg(short, long, action = ArgAction::Count)]
    pub verbose: u8,
    /// Disable the dedup index entirely (every source file is treated as fresh)
    #[arg(long)]
    pub no_index: bool,
    /// Override the dedup index path (default: ~/.cache/fcp/index.rkyv)
    #[arg(long, value_name = "PATH")]
    pub index_path: Option<PathBuf>,
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

    let default_level = match cli_args.verbose {
        0 => LevelFilter::INFO,
        1 => LevelFilter::DEBUG,
        _ => LevelFilter::TRACE,
    };
    let env_filter = EnvFilter::builder()
        .with_default_directive(default_level.into())
        .from_env_lossy();

    let indicative_layer = IndicatifLayer::new();

    let (chrome_layer, _chrome_guard) = cli_args
        .trace
        .as_deref()
        .map(|path| {
            ChromeLayerBuilder::new()
                .file(path)
                .include_args(true)
                .build()
        })
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

    let no_index = cli_args.no_index;
    let dry = cli_args.dry;
    let index_path = cli_args.index_path.clone().unwrap_or_else(|| {
        dirs::cache_dir()
            .unwrap_or_else(std::env::temp_dir)
            .join("fcp")
            .join("index.rkyv")
    });

    let configuration: Arc<fcp::CopyParams> = Arc::new(cli_args.into());

    reg.init();

    let index = if no_index || dry {
        DedupIndex::disabled()
    } else {
        DedupIndex::open(index_path.clone())?
    };

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
        enclose!((index, reporter) move |path| stage_dedup(path, Arc::clone(&index), Arc::clone(&reporter))),
        Concurrency::concurrent_ordered(io_parallelism),
    )
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
        enclose!((configuration, reporter, index)
            move |(photo, new_path)| fcp::stage_copy(photo, new_path, Arc::clone(&configuration), Arc::clone(&reporter), Arc::clone(&index))
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

    let index_total = match index.save() {
        Ok(n) => Some(n),
        Err(e) => {
            warn!("couldn't save dedup index: {e:?}");
            None
        }
    };

    eprintln!(
        "{} copied, {} hard-linked, {} dry-run, {} skipped (dest), {} skipped (dedup), {} errors — report: {}",
        summary.copied,
        summary.hard_linked,
        summary.dry_run,
        summary.skipped,
        summary.skipped_dedup,
        summary.meta_error + summary.copy_error,
        reporter_task.path.display(),
    );
    if let Some(n) = index_total {
        if !no_index && !dry {
            eprintln!("dedup index: {} entries at {}", n, index_path.display());
        }
    }

    Ok(())
}
