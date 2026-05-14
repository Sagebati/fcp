use std::path::Path;

use anyhow::{anyhow, Context};
use bytes::Bytes;
use chrono::{Datelike, Timelike};
use nom_exif::{AsyncMediaSource, ExifIter, ExifTag, MediaParser, TagOrCode};
use serde::Serialize;
use tokio::fs::File;

use crate::Res;

#[derive(Serialize, Debug)]
pub struct PhotoMeta {
    pub year: i16,
    pub month: i8,
    pub day: i8,
    pub minutes: i8,
    pub seconds: i8,
}

pub async fn parse_exif_from_path(path: &Path) -> Res<PhotoMeta> {
    let file = File::open(path)
        .await
        .with_context(|| format!("open {path:?}"))?;
    let mut parser = MediaParser::new();
    let source = AsyncMediaSource::seekable(file)
        .await
        .context("nom-exif: AsyncMediaSource::seekable failed")?;
    let iter = parser
        .parse_exif_async(source)
        .await
        .context("nom-exif: parse_exif_async failed")?;
    extract_datetime(iter)
}

pub async fn parse_exif_from_bytes(bytes: Bytes) -> Res<PhotoMeta> {
    let mut parser = MediaParser::new();
    let source = AsyncMediaSource::from_memory(bytes)
        .context("nom-exif: AsyncMediaSource::from_memory failed")?;
    let iter = parser
        .parse_exif_async(source)
        .await
        .context("nom-exif: parse_exif_async failed")?;
    extract_datetime(iter)
}

fn extract_datetime(iter: ExifIter) -> Res<PhotoMeta> {
    let mut original = None;
    let mut fallback = None;
    for entry in iter {
        match entry.tag() {
            TagOrCode::Tag(ExifTag::DateTimeOriginal) => {
                original = entry.value().and_then(|v| v.as_datetime());
                if original.is_some() {
                    break;
                }
            }
            TagOrCode::Tag(ExifTag::ModifyDate) => {
                fallback = entry.value().and_then(|v| v.as_datetime());
            }
            _ => {}
        }
    }
    let dt = original
        .or(fallback)
        .ok_or_else(|| anyhow!("no DateTimeOriginal or ModifyDate tag in EXIF"))?
        .into_naive();
    Ok(PhotoMeta {
        year: dt.year() as i16,
        month: dt.month() as i8,
        day: dt.day() as i8,
        minutes: dt.minute() as i8,
        seconds: dt.second() as i8,
    })
}
