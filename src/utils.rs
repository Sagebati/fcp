use bon::builder;
use clap::Parser;
use tracing_indicatif::IndicatifLayer;
mod cli;
mod exif;

use crate::cli::Configuration;
use crate::exif::{parse_exif, File};
use crate::lib::{compute_new_path, copy, scan_library_paths, FileIgnoredReason, Photo, Res};
use anyhow::Context;
use bytes::Bytes;
use futures::stream::FuturesUnordered;
use futures::StreamExt;
use indicatif::ProgressStyle;
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock};
use tokio::io::AsyncReadExt;
use tokio::sync::Semaphore;
use tokio::task::spawn_blocking;
use tracing::{debug, debug_span, info_span, warn, Instrument, Span};
use tracing_indicatif::span_ext::IndicatifSpanExt;
use tracing_subscriber::filter::LevelFilter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::EnvFilter;

#[tracing::instrument]
pub async fn load_file_bytes(path: &Path) -> Bytes {
    let mut buff = vec![0; 29 * 1024 * 1024];
    let mut x = tokio::fs::File::open(path).await.unwrap();
    x.read_to_end(&mut buff).await.unwrap();
    buff.into()
}

#[tracing::instrument(skip(include_bytes))]
#[builder]
pub async fn load_file(file_path: PathBuf, #[builder(default)] include_bytes: bool) -> Res<Photo> {
    let bytes = if include_bytes {
        Some(load_file_bytes(&file_path).await)
    } else {
        None
    };

    let f = File {
        path: file_path.clone(),
        bytes,
    };

    let exif = spawn_blocking(|| parse_exif(f)).await.unwrap();

    Ok(Photo {
        meta: exif.context(format!("Failed to load exif of {file_path:?}"))?,
        bytes: None,
        original_path: file_path,
    })
}

