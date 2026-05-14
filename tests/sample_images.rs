//! Coverage test: download real-world camera samples and report which file
//! extensions `fcp::Photo::load_meta()` can parse.
//!
//! Marked `#[ignore]` — runs only when explicitly requested:
//!
//!     cargo test --test sample_images -- --ignored --nocapture
//!     just test-samples
//!
//! Downloads ~50 MB on first run and caches under `target/tmp/<pkg>/`.

use std::collections::BTreeMap;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use fcp::Photo;
use futures::stream::{self, StreamExt};
use walkdir::WalkDir;

// Per-vendor JPEG sample archives from https://exiftool.org/sample_images.html
// (relative links on that page resolve to https://exiftool.org/<Vendor>.tar.gz).
//
// Verified 2026-05-14: all return HTTP 200.
const ARCHIVE_URLS: &[&str] = &[
    "https://exiftool.org/Apple.tar.gz",
    "https://exiftool.org/Canon.tar.gz",
    "https://exiftool.org/Nikon.tar.gz",
    "https://exiftool.org/Sony.tar.gz",
    "https://exiftool.org/FujiFilm.tar.gz",
    "https://exiftool.org/Panasonic.tar.gz",
    "https://exiftool.org/Samsung.tar.gz",
    "https://exiftool.org/DJI.tar.gz",
    "https://exiftool.org/Google.tar.gz",
    "https://exiftool.org/GoPro.tar.gz",
    "https://exiftool.org/Olympus.tar.gz",
    "https://exiftool.org/Pentax.tar.gz",
];

// Individual RAW/HEIC/video samples — the exiftool sample_images archives are
// JPEG-only, so we pull these from upstream test fixtures to also exercise
// nom-exif's RAW, HEIC, and ISO-base-media (mov/mp4) paths.
//
// URLs pinned to commit SHAs so they cannot drift.
//   exiftool/exiftool @ 38bdbace3037a89d2c332664fc60bddad118958a
//   mindeng/nom-exif  @ 12de89950b6d3798419981a9a61f284178712da5
// Verified 2026-05-14: all return HTTP 200.
const RAW_URLS: &[&str] = &[
    "https://raw.githubusercontent.com/exiftool/exiftool/38bdbace3037a89d2c332664fc60bddad118958a/t/images/Nikon.nef",
    "https://raw.githubusercontent.com/exiftool/exiftool/38bdbace3037a89d2c332664fc60bddad118958a/t/images/CanonRaw.cr2",
    "https://raw.githubusercontent.com/exiftool/exiftool/38bdbace3037a89d2c332664fc60bddad118958a/t/images/CanonRaw.cr3",
    "https://raw.githubusercontent.com/exiftool/exiftool/38bdbace3037a89d2c332664fc60bddad118958a/t/images/FujiFilm.raf",
    "https://raw.githubusercontent.com/exiftool/exiftool/38bdbace3037a89d2c332664fc60bddad118958a/t/images/DNG.dng",
    "https://raw.githubusercontent.com/exiftool/exiftool/38bdbace3037a89d2c332664fc60bddad118958a/t/images/Panasonic.rw2",
    "https://raw.githubusercontent.com/mindeng/nom-exif/12de89950b6d3798419981a9a61f284178712da5/testdata/exif.heic",
    "https://raw.githubusercontent.com/mindeng/nom-exif/12de89950b6d3798419981a9a61f284178712da5/testdata/canon-r6.cr3",
    "https://raw.githubusercontent.com/mindeng/nom-exif/12de89950b6d3798419981a9a61f284178712da5/testdata/tif.tif",
    "https://raw.githubusercontent.com/mindeng/nom-exif/12de89950b6d3798419981a9a61f284178712da5/testdata/meta.mov",
    "https://raw.githubusercontent.com/mindeng/nom-exif/12de89950b6d3798419981a9a61f284178712da5/testdata/meta.mp4",
];

#[derive(Default)]
struct ExtStats {
    total: usize,
    parsed: usize,
    no_date: usize,
    failed: usize,
    first_error: Option<String>,
}

#[test]
#[ignore = "downloads ~50 MB from the internet; run with --ignored"]
fn sample_images_coverage() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .expect("build tokio runtime");
    rt.block_on(async { run().await.expect("test body") })
}

async fn run() -> Result<()> {
    let cache = cache_root()?;
    fs::create_dir_all(&cache).with_context(|| format!("mkdir {cache:?}"))?;

    let cache_for_blocking = cache.clone();
    let paths = tokio::task::spawn_blocking(move || -> Result<Vec<PathBuf>> {
        let extract_root = cache_for_blocking.join("extracted");
        let archive_dir = cache_for_blocking.join("archives");
        let raw_dir = cache_for_blocking.join("raw");
        fs::create_dir_all(&extract_root)?;
        fs::create_dir_all(&archive_dir)?;
        fs::create_dir_all(&raw_dir)?;

        for url in ARCHIVE_URLS {
            ensure_archive(url, &archive_dir, &extract_root)?;
        }
        for url in RAW_URLS {
            let name = url.rsplit('/').next().unwrap();
            ensure_file(url, &raw_dir.join(name))?;
        }
        // Walk only the extracted/ and raw/ trees — the archives/ dir holds
        // the .tar.gz downloads themselves, which we don't want to parse.
        let mut files = collect_files(&extract_root);
        files.extend(collect_files(&raw_dir));
        Ok(files)
    })
    .await??;

    eprintln!("found {} files in cache, parsing…", paths.len());
    let start = Instant::now();

    let results: Vec<(PathBuf, anyhow::Result<()>)> = stream::iter(paths)
        .map(|p| async move {
            let res = Photo::new(p.clone()).load_meta().await.map(|_| ());
            (p, res)
        })
        .buffer_unordered(8)
        .collect()
        .await;

    let elapsed = start.elapsed();
    let mut stats: BTreeMap<String, ExtStats> = BTreeMap::new();
    for (path, res) in &results {
        let ext = match path.extension().and_then(|e| e.to_str()) {
            Some(e) => e.to_ascii_lowercase(),
            None => continue,
        };
        let s = stats.entry(ext).or_default();
        s.total += 1;
        match res {
            Ok(()) => s.parsed += 1,
            Err(e) => {
                let msg = format!("{e:#}");
                // `parse_exif_from_path` returns this exact message when nom-exif
                // could read the file but no DateTimeOriginal/ModifyDate was present.
                if msg.contains("no DateTimeOriginal or ModifyDate") {
                    s.no_date += 1;
                } else {
                    s.failed += 1;
                }
                if s.first_error.is_none() {
                    s.first_error = Some(msg);
                }
            }
        }
    }

    print_report(&stats, elapsed);

    let jpg = stats.get("jpg").cloned_or_default();
    assert!(
        jpg.total >= 100,
        "expected ≥100 jpg samples, got {} (cache may be empty)",
        jpg.total,
    );
    let jpg_ok = jpg.parsed + jpg.no_date;
    let jpg_rate = jpg_ok as f64 / jpg.total as f64;
    assert!(
        jpg_rate >= 0.95,
        "jpg readability {jpg_rate:.3} below 0.95 ({jpg_ok}/{} read by nom-exif)",
        jpg.total,
    );
    for required in ["nef", "raf"] {
        let s = stats.get(required).cloned_or_default();
        assert!(
            s.total >= 1,
            "no {required} samples found — URL list may have changed",
        );
        let ok = s.parsed + s.no_date;
        assert!(
            ok >= 1,
            "no {required} files readable by nom-exif ({}/{} failed)",
            s.failed,
            s.total,
        );
    }
    Ok(())
}

fn cache_root() -> Result<PathBuf> {
    // CARGO_TARGET_TMPDIR is set by cargo for integration tests since 1.54
    // and points at target/tmp/<pkg>/. Falls back to manifest-relative for
    // tooling that doesn't set it.
    if let Some(p) = option_env!("CARGO_TARGET_TMPDIR") {
        return Ok(PathBuf::from(p).join("sample-images"));
    }
    Ok(PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("sample-images"))
}

fn ensure_archive(url: &str, archive_dir: &Path, extract_root: &Path) -> Result<PathBuf> {
    let name = url
        .rsplit('/')
        .next()
        .with_context(|| format!("bad url: {url}"))?;
    let stem = name.trim_end_matches(".tar.gz");
    let extract_dir = extract_root.join(stem);
    let marker = extract_dir.join(".complete");
    if marker.exists() {
        return Ok(extract_dir);
    }
    // Wipe any partial state from an earlier interrupted run.
    if extract_dir.exists() {
        fs::remove_dir_all(&extract_dir).ok();
    }
    fs::create_dir_all(&extract_dir)?;

    let archive_path = archive_dir.join(name);
    download_to(url, &archive_path)?;

    let file = fs::File::open(&archive_path)?;
    let gz = flate2::read::GzDecoder::new(file);
    let mut tar = tar::Archive::new(gz);
    tar.unpack(&extract_dir)
        .with_context(|| format!("untar {name}"))?;

    fs::File::create(&marker)?;
    Ok(extract_dir)
}

fn ensure_file(url: &str, dest: &Path) -> Result<PathBuf> {
    if dest.exists() && fs::metadata(dest).map(|m| m.len() > 0).unwrap_or(false) {
        return Ok(dest.to_path_buf());
    }
    download_to(url, dest)?;
    Ok(dest.to_path_buf())
}

fn download_to(url: &str, dest: &Path) -> Result<()> {
    eprintln!("downloading {url}");
    let resp = ureq::get(url)
        .timeout(std::time::Duration::from_secs(120))
        .call();
    let resp = match resp {
        Ok(r) => r,
        Err(e) => bail!("ureq GET {url}: {e}"),
    };
    if resp.status() != 200 {
        bail!("HTTP {} for {url}", resp.status());
    }
    let tmp = dest.with_extension("partial");
    let mut out = fs::File::create(&tmp)?;
    let mut reader = resp.into_reader();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n])?;
    }
    out.flush()?;
    drop(out);
    fs::rename(&tmp, dest)?;
    Ok(())
}

fn collect_files(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for entry in WalkDir::new(root).follow_links(false) {
        let Ok(entry) = entry else { continue };
        if !entry.file_type().is_file() {
            continue;
        }
        let name = entry.file_name().to_string_lossy();
        // Skip dotfiles and macOS AppleDouble forks that some exiftool
        // archives ship alongside the real samples.
        if name.starts_with('.') || name.starts_with("._") {
            continue;
        }
        out.push(entry.into_path());
    }
    out
}

fn print_report(stats: &BTreeMap<String, ExtStats>, elapsed: std::time::Duration) {
    let mut rows: Vec<(&String, &ExtStats)> = stats.iter().collect();
    rows.sort_by_key(|r| std::cmp::Reverse(r.1.total));

    println!();
    println!(
        "## Sample-image parse coverage  ({} extensions, {:?})",
        rows.len(),
        elapsed
    );
    println!();
    println!("| extension | total | parsed | no_date | failed | first error |");
    println!("|-----------|------:|-------:|--------:|-------:|-------------|");
    for (ext, s) in rows {
        let err = s
            .first_error
            .as_deref()
            .map(|e| {
                let one = e.replace('\n', " ");
                if one.len() > 80 {
                    format!("{}…", &one[..80])
                } else {
                    one
                }
            })
            .unwrap_or_default();
        println!(
            "| {:<9} | {:>5} | {:>6} | {:>7} | {:>6} | {} |",
            ext, s.total, s.parsed, s.no_date, s.failed, err,
        );
    }
    println!();
}

// Tiny helper: BTreeMap::get + default-clone for assertion ergonomics.
trait ClonedOrDefault<T> {
    fn cloned_or_default(self) -> T;
}
impl<T: Default + Clone> ClonedOrDefault<T> for Option<&T> {
    fn cloned_or_default(self) -> T {
        self.cloned().unwrap_or_default()
    }
}

impl Clone for ExtStats {
    fn clone(&self) -> Self {
        Self {
            total: self.total,
            parsed: self.parsed,
            no_date: self.no_date,
            failed: self.failed,
            first_error: self.first_error.clone(),
        }
    }
}
