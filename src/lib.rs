pub mod clip;
mod exif;
mod error;

pub use crate::clip::{
    load_image_for_clip, ClipError, ClipTagger, ClipTaggerManager, ClipTaggerPool,
};
use bon::bon;

use crate::exif::PhotoMeta;
use anyhow::{anyhow, Context};
use bytes::Bytes;
use clap::Parser;
use derive_more::with_trait::{From, Unwrap};
use derive_more::Display;
use ecow::EcoString;
use futures::channel::mpsc::{unbounded, UnboundedReceiver};
use jiff::civil::DateTime;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use std::borrow::Cow;
use std::collections::{BTreeSet, HashMap};
use std::fmt::Debug;
use std::fs::File;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::OnceCell;
use tokio::task::spawn_blocking;
use tracing::{debug, info, info_span, instrument, warn, Span};
use tracing_indicatif::span_ext::IndicatifSpanExt;
use walkdir::DirEntry;

pub type PhotoHash = [u8; 32];

pub type Res<T = ()> = anyhow::Result<T>;

pub type MetaNotLoaded = ();
pub type MetaLoaded = PhotoMeta;

#[derive(bon::Builder)]
pub struct LoadLibrary {
    path: PathBuf,

    autotagging: bool,

    #[builder(default = std::thread::available_parallelism()
        .unwrap_or_else(|_| 1.try_into().unwrap()).into())]
    cpu_parallelism: usize,
    io_parallelism: usize,
}

pub struct Library {}

#[instrument(skip_all)]
pub fn load_library(load_library: &LoadLibrary) -> Library {
    Library {}
}

#[derive(Debug)]
pub struct Photo<Meta = MetaLoaded> {
    original_path: PathBuf,
    meta: Meta,
    bytes: tokio::sync::OnceCell<Bytes>,
}

impl Photo<MetaNotLoaded> {
    pub fn new(path: PathBuf) -> Self {
        Self {
            original_path: path,
            meta: (),
            bytes: OnceCell::new(),
        }
    }
    #[tracing::instrument(skip_all, fields(path = ?self.original_path.file_name()))]
    pub async fn load_meta(self) -> Res<Photo> {
        let meta = if let Some(b) = self.bytes.get() {
            rexiv2::Metadata::new_from_buffer(b)
        } else {
            rexiv2::Metadata::new_from_path(&self.original_path)
        }
        .context("Unable to load metadata with rexiv2")?;

        let date = meta
            .get_tag_string("Exif.Photo.DateTimeOriginal")
            .or_else(|_| meta.get_tag_string("Exif.Image.DateTime"))
            .map_err(|e| anyhow!("date not found: {}", e))?;

        static FORMAT: &str = "%Y:%m:%d %H:%M:%S";

        let date = DateTime::strptime(FORMAT, date)?;
        let meta = PhotoMeta {
            year: date.year() as i16,
            month: date.month() as i8,
            day: date.day() as i8,
            minutes: date.minute() as i8,
            seconds: date.second() as i8,
        };

        Ok(Photo {
            original_path: self.original_path,
            meta,
            bytes: self.bytes,
        })
    }
}

impl Photo {
    pub fn meta(&self) -> &PhotoMeta {
        &self.meta
    }
}

impl<T> Photo<T> {
    pub fn original_path(&self) -> &Path {
        &self.original_path
    }

    pub async fn bytes(&self) -> Res<&Bytes> {
        let bytes = self
            .bytes
            .get_or_try_init(|| load_file_bytes(&self.original_path))
            .await?;

        Ok(bytes)
    }
}

#[instrument]
pub fn hash_photo(bytes: &[u8]) -> Res<PhotoHash> {
    let mut hasher = blake3::Hasher::new();
    hasher.update_rayon(&bytes);
    Ok(hasher.finalize().into())
}

#[instrument]
pub fn hash_photo_file(path: &Path) -> Res<PhotoHash> {
    let mut hasher = blake3::Hasher::new();
    hasher.update_reader(File::open(path)?)?;
    Ok(hasher.finalize().into())
}

pub fn compute_new_path(folder: &Path, conf: &str, photo: &Photo) -> PathBuf {
    let path_string = conf
        .replace("{{year}}", &photo.meta.year.to_string())
        .replace("{{month}}", &photo.meta.month.to_string())
        .replace("{{day}}", &photo.meta.day.to_string())
        .replace(
            "{{original}}",
            photo.original_path.file_name().unwrap().to_str().unwrap(),
        );

    folder.join(&path_string)
}

pub type Index = HashMap<PhotoHash, String>;

pub static DEFAULT_DEST: &str = "{{year}}/{{year}}_{{month}}_{{day}}/{{original}}";

pub fn ext_default() -> Vec<EcoString> {
    vec![
        EcoString::from("raf"),
        EcoString::from("RAF"),
        //EcoString::from("jpg"),
        //EcoString::from("JPG"),
        EcoString::from("NEF"),
        EcoString::from("nef"),
    ]
}

#[derive(Parser, Debug, Clone)]
pub struct Configuration {
    pub from: PathBuf,

    pub dest: PathBuf,

    #[arg(short, long, default_values_t = ext_default())]
    pub image_extensions: Vec<EcoString>,

    #[arg(short = 'x', long)]
    pub ignore_extensions: Vec<EcoString>,

    #[arg(long, default_value_t = DEFAULT_DEST.into())]
    pub path_format: Cow<'static, str>,

    #[arg(short = 'n', long, default_value_t = false)]
    pub dry: bool,

    #[arg(
        short = 'l',
        long,
        default_value_t = false,
        help = "Will use hard links instead of copying it's faster but the files must in the same filesystem"
    )]
    pub hard_links: bool,

    #[arg(
        long,
        default_value_t = false,
        help = "compare photos using names not hashes"
    )]
    pub hash: bool,

    #[arg(
        short,
        long,
        default_value_t = false,
        help = "if the photo already exists overwrite it"
    )]
    pub force: bool,

    #[arg(
        short,
        long,
        default_value_t = 20,
        help = "The number of copies at the same time will limit"
    )]
    pub concurrency_limit: usize,
}

#[instrument(skip(conf))]
pub fn scan_library_paths(conf: &Configuration) -> impl Iterator<Item = PathBuf> + '_ {
    #[instrument(skip(image_extensions))]
    fn filter_file(image_extensions: &BTreeSet<EcoString>, file: &DirEntry) -> Option<PathBuf> {
        let file_path = file.path();

        let ext = file_path.extension()?;
        let ext = ext
            .to_str()
            .context("Couldn't cast the extension {x} to UTF-8")
            .inspect_err(|e| warn!("{}", e))
            .ok()?;

        if image_extensions.contains(ext) {
            debug!(path.examinated=?file.path());
            Some(file_path.to_path_buf())
        } else {
            debug!("file ignored because extension unmatched {:?}", file_path);
            None
        }
    }

    let filter: Box<dyn Fn(&DirEntry) -> Option<PathBuf>> = {
        let image_extensions = conf
            .image_extensions
            .iter()
            .cloned()
            .collect::<BTreeSet<_>>();

        Box::new(move |file| filter_file(&image_extensions, file))
    };

    let walker = {
        let root_path = conf.from.as_path();
        walkdir::WalkDir::new(root_path)
            .follow_links(true)
            .contents_first(true)
    };

    walker
        .into_iter()
        .filter(|e| match e {
            Ok(e) => e.file_type().is_file(),
            _ => false,
        })
        .filter_map(move |file| {
            if file.is_err() {
                return None;
            }
            let file = file.unwrap();
            filter(&file)
        })
}

#[instrument]
pub async fn load_file_bytes(path: &Path) -> Res<Bytes> {
    let b = tokio::fs::read(path)
        .await
        .context("couldn't load the bytes from the file")?;
    Ok(b.into())
}

#[derive(From, Unwrap)]
pub enum CopyFrom<'a> {
    Path(&'a Path),
    Bytes(&'a Photo),
}

impl CopyFrom<'_> {
    pub fn path(&self) -> &Path {
        match self {
            CopyFrom::Path(p) => p,
            CopyFrom::Bytes(b) => &b.original_path,
        }
    }
}

#[tracing::instrument(skip(configuration, from), fields(src))]
pub async fn copy(
    from: impl Into<CopyFrom<'_>>,
    to: &Path,
    configuration: &Configuration,
) -> anyhow::Result<()> {
    let folder_structure = to.parent().unwrap();
    let from = from.into();
    tracing::Span::current().record("src", tracing::field::debug(from.path().file_name()));

    if !folder_structure.exists() {
        tokio::fs::create_dir_all(folder_structure)
            .await
            .context("Couldn't create the folder structure to store the photo")?
    }

    if configuration.dry {
        info!(to=?to, " dry");
    }
    if configuration.hard_links {
        tokio::fs::hard_link(&from.path(), &to)
            .await
            .context("Couldn't create hard link")?;
        info!(file_path = ?from.path(), to=?to, " hard link");
    } else {
        match from {
            CopyFrom::Path(p) => {
                tokio::fs::copy(p, &to)
                    .await
                    .context("Couldn't copy the photo")?;
            }
            CopyFrom::Bytes(_) => {
                todo!()
            }
        }
        info!(file_path = ?from.path(), to=?to, " copied");
    }
    Ok(())
}

#[derive(Display)]
pub enum FileIgnoredReason {
    FileAlreadyExists,
}

pub fn stage_scan(config: Arc<Configuration>, progress: Span) -> UnboundedReceiver<PathBuf> {
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
pub async fn stage_load_meta(path: PathBuf) -> Res<Photo> {
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
    batch: &[Photo],
    tags: &[String],
    pool: &ClipTaggerPool,
) -> Vec<Vec<String>> {
    // Load bytes for all photos concurrently.
    let bytes_per_photo: Vec<Option<Bytes>> =
        futures::future::join_all(batch.iter().map(|photo| photo.bytes()))
            .await
            .into_iter()
            .map(|r| {
                r.map(|b| b.clone())
                    .map_err(|e| {
                        warn!("Failed to load bytes: {e:?}");
                        e
                    })
                    .ok()
            })
            .collect();

    let mut tagger = match pool.get().await {
        Ok(t) => t,
        Err(e) => {
            warn!("Failed to acquire tagger from pool: {e:?}");
            todo!()
        }
    };

    let span = tracing::Span::current();
    let result = spawn_blocking( move || {
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

        let tag_results = tagger.predict_batch(&images, &tags, 0.2)?;

        Ok::<Vec<(usize, Vec<String>)>, anyhow::Error>(
            indices.into_iter().zip(tag_results).collect(),
        )
    })
    .await;

    match result {
        Ok(Ok(tagged)) => {
            for (idx, tags) in &tagged {
                todo!()
            }
        }
        Ok(Err(e)) => warn!("Batch tagging failed: {e:?}"),
        Err(e) => warn!("Batch tagger task panicked: {e:?}"),
    }

    todo!()
}

#[instrument(skip_all, fields(file = ?photo.original_path().file_name()))]
pub async fn stage_copy(photo: Photo, new_path: PathBuf, config: Arc<Configuration>) -> Res<()> {
    copy(photo.original_path(), &new_path, &config)
        .await
        .context(format!(
            "Error occurred when copying {:?}",
            photo.original_path()
        ))
}
