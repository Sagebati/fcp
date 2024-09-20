use std::io::{BufReader, Cursor};
use std::time::SystemTime;
use bytes::{Buf, Bytes};
use chrono::{DateTime, Datelike, FixedOffset, Timelike};
use nom_exif::{Exif, ExifIter, ExifTag, MediaParser, MediaSource};
use serde::Serialize;
use time::convert::Minute;

#[derive(Serialize)]
pub struct PhotoMeta {
    pub year: i32,
    pub month: u32,
    pub day: u32,
    pub minutes: u32,
    pub seconds: u32,
}

pub fn parse_exif(bytes: Bytes, parser: &mut MediaParser) -> PhotoMeta {
    let cursor = Cursor::new(bytes);
    let exif = MediaSource::seekable(cursor).unwrap();
    if exif.has_exif() {
        let exif: Exif = parser.parse::<_, _, ExifIter>(exif).unwrap().into();
        let date = exif.get(ExifTag::DateTimeOriginal).unwrap().as_time().unwrap_or_default();

        return PhotoMeta {
            year: date.year(),
            month: date.month(),
            day: date.day(),
            minutes: date.minute(),
            seconds: date.second(),
        };
    }
    panic!()
}