use clap::{Args, Subcommand, Parser};
use fpd::{ClipTagger, Configuration, Res};
use std::path::PathBuf;

#[derive(Parser)]
pub struct Cli {
    #[command(flatten)]
    pub config: Configuration,

    #[command(subcommand)]
    pub command: Option<Commands>,

    #[arg(long, help = "Write a Chrome trace JSON to this path (open in chrome://tracing)")]
    pub trace: Option<std::path::PathBuf>,

    #[arg(
        long,
        default_value_t = 32,
        help = "Number of images per CLIP inference batch — larger values better saturate the GPU"
    )]
    pub tag_batch_size: usize,
}

#[derive(Subcommand)]
pub enum Commands {
    Tag(Tag),
}

#[derive(Args)]
pub struct Tag {
    pub files: Vec<PathBuf>,
    #[arg(short, long)]
    pub recursive: bool,
    #[arg(short, long, default_value = "0.2")]
    pub threshold: f32,
    #[arg(long)]
    pub use_embedded: bool,
}

pub async fn run_tagger(tag: Tag) -> Res {
    let mut tagger = if tag.use_embedded {
        #[cfg(feature = "embed-models")]
        {
            ClipTagger::from_embedded()?
        }
        #[cfg(not(feature = "embed-models"))]
        {
            return Err(anyhow::anyhow!("Embedded models feature is not enabled. Recompile with --features embed-models"));
        }
    } else {
        ClipTagger::from_local_default()?
    };
    let tags = vec![
        "a photo of a cat".to_string(),
        "a photo of a dog".to_string(),
        "a landscape".to_string(),
        "a portrait".to_string(),
    ];

    for file in tag.files {
        if file.is_file() {
            let img = image::open(&file)?;
            let matched_tags = tagger.predict(&img, &tags, tag.threshold)?;
            println!("{:?}: {:?}", file, matched_tags);
        }
    }

    Ok(())
}