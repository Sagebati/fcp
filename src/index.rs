use anyhow::{anyhow, Context};
use memmap2::Mmap;
use rkyv::rancor;
use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};
use tracing::{debug, instrument, warn};

use crate::Res;

const FORMAT_VERSION: u32 = 1;

#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize, Clone, Debug)]
pub struct DbEntry {
    pub size: u64,
    pub mtime_ns: i64,
    pub name: String,
    pub src_path: String,
    pub dst_path: String,
    pub imported_at_ns: i64,
}

#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize, Debug)]
struct DbV1 {
    version: u32,
    entries: Vec<DbEntry>,
}

#[derive(Hash, Eq, PartialEq, Clone, Debug)]
pub struct Fingerprint {
    pub size: u64,
    pub mtime_ns: i64,
    pub name: String,
}

impl Fingerprint {
    pub async fn from_source(path: &Path) -> Res<Self> {
        let meta = tokio::fs::metadata(path)
            .await
            .with_context(|| format!("Couldn't stat source file {path:?}"))?;
        let mtime = meta
            .modified()
            .context("filesystem does not report modification time")?;
        let mtime_ns = match mtime.duration_since(std::time::UNIX_EPOCH) {
            Ok(d) => d.as_nanos() as i64,
            Err(e) => -(e.duration().as_nanos() as i64),
        };
        let name = path
            .file_name()
            .and_then(|s| s.to_str())
            .ok_or_else(|| anyhow!("source path {path:?} has no valid UTF-8 file name"))?
            .to_string();
        Ok(Fingerprint {
            size: meta.len(),
            mtime_ns,
            name,
        })
    }
}

impl DbEntry {
    fn fingerprint(&self) -> Fingerprint {
        Fingerprint {
            size: self.size,
            mtime_ns: self.mtime_ns,
            name: self.name.clone(),
        }
    }
}

pub struct DedupIndex {
    disabled: bool,
    disk_path: PathBuf,
    _mmap: Option<Mmap>,
    baseline: Vec<DbEntry>,
    seen: RwLock<HashSet<Fingerprint>>,
    pending: Mutex<Vec<DbEntry>>,
}

impl DedupIndex {
    pub fn disabled() -> Arc<Self> {
        Arc::new(Self {
            disabled: true,
            disk_path: PathBuf::new(),
            _mmap: None,
            baseline: Vec::new(),
            seen: RwLock::new(HashSet::new()),
            pending: Mutex::new(Vec::new()),
        })
    }

    #[instrument(skip_all, fields(path = ?path))]
    pub fn open(path: PathBuf) -> Res<Arc<Self>> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).with_context(|| {
                format!("Couldn't create parent directory for index at {parent:?}")
            })?;
        }

        let (mmap, baseline) = match File::open(&path) {
            Ok(f) => {
                let mmap = unsafe { Mmap::map(&f) }
                    .with_context(|| format!("Couldn't mmap index {path:?}"))?;
                if mmap.is_empty() {
                    (None, Vec::new())
                } else {
                    let archived = rkyv::access::<ArchivedDbV1, rancor::Error>(&mmap[..])
                        .with_context(|| format!("Index {path:?} is corrupt or wrong version"))?;
                    if archived.version != FORMAT_VERSION {
                        warn!(
                            found = u32::from(archived.version),
                            expected = FORMAT_VERSION,
                            "ignoring index: unexpected format version"
                        );
                        (None, Vec::new())
                    } else {
                        let entries: Vec<DbEntry> =
                            rkyv::deserialize::<Vec<DbEntry>, rancor::Error>(&archived.entries)
                                .context("Couldn't deserialize index entries")?;
                        (Some(mmap), entries)
                    }
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => (None, Vec::new()),
            Err(e) => return Err(e).context(format!("Couldn't open index {path:?}")),
        };

        let seen: HashSet<Fingerprint> = baseline.iter().map(DbEntry::fingerprint).collect();
        debug!(loaded = baseline.len(), "dedup index opened");

        Ok(Arc::new(Self {
            disabled: false,
            disk_path: path,
            _mmap: mmap,
            baseline,
            seen: RwLock::new(seen),
            pending: Mutex::new(Vec::new()),
        }))
    }

    pub fn contains(&self, fp: &Fingerprint) -> bool {
        if self.disabled {
            return false;
        }
        self.seen.read().unwrap().contains(fp)
    }

    pub fn record(&self, entry: DbEntry) {
        if self.disabled {
            return;
        }
        let fp = entry.fingerprint();
        self.seen.write().unwrap().insert(fp);
        self.pending.lock().unwrap().push(entry);
    }

    pub fn pending_count(&self) -> usize {
        self.pending.lock().unwrap().len()
    }

    pub fn baseline_count(&self) -> usize {
        self.baseline.len()
    }

    #[instrument(skip_all, fields(path = ?self.disk_path))]
    pub fn save(&self) -> Res<usize> {
        if self.disabled {
            return Ok(0);
        }

        let pending = std::mem::take(&mut *self.pending.lock().unwrap());
        if pending.is_empty() {
            return Ok(self.baseline.len());
        }

        let mut all = Vec::with_capacity(self.baseline.len() + pending.len());
        all.extend(self.baseline.iter().cloned());
        all.extend(pending);
        let total = all.len();
        let db = DbV1 {
            version: FORMAT_VERSION,
            entries: all,
        };

        let bytes = rkyv::to_bytes::<rancor::Error>(&db).context("Couldn't serialize index")?;

        let tmp = self.disk_path.with_extension("rkyv.tmp");
        {
            let mut f = OpenOptions::new()
                .create(true)
                .truncate(true)
                .write(true)
                .open(&tmp)
                .with_context(|| format!("Couldn't open {tmp:?}"))?;
            f.write_all(&bytes)
                .with_context(|| format!("Couldn't write index to {tmp:?}"))?;
            f.sync_all()
                .with_context(|| format!("Couldn't fsync {tmp:?}"))?;
        }
        std::fs::rename(&tmp, &self.disk_path)
            .with_context(|| format!("Couldn't rename {tmp:?} → {:?}", self.disk_path))?;

        debug!(total, "dedup index saved");
        Ok(total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(name: &str, size: u64, mtime: i64) -> DbEntry {
        DbEntry {
            size,
            mtime_ns: mtime,
            name: name.to_string(),
            src_path: format!("/src/{name}"),
            dst_path: format!("/dst/{name}"),
            imported_at_ns: 0,
        }
    }

    #[test]
    fn round_trip() {
        let tmp = tempdir();
        let path = tmp.join("index.rkyv");

        let idx = DedupIndex::open(path.clone()).unwrap();
        assert_eq!(idx.baseline_count(), 0);
        idx.record(entry("a.jpg", 100, 1));
        idx.record(entry("b.jpg", 200, 2));
        idx.record(entry("c.jpg", 300, 3));
        let saved = idx.save().unwrap();
        assert_eq!(saved, 3);

        let idx2 = DedupIndex::open(path).unwrap();
        assert_eq!(idx2.baseline_count(), 3);
        assert!(idx2.contains(&Fingerprint {
            size: 100,
            mtime_ns: 1,
            name: "a.jpg".to_string()
        }));
        assert!(idx2.contains(&Fingerprint {
            size: 300,
            mtime_ns: 3,
            name: "c.jpg".to_string()
        }));
        assert!(!idx2.contains(&Fingerprint {
            size: 999,
            mtime_ns: 9,
            name: "nope.jpg".to_string()
        }));
    }

    #[test]
    fn disabled_is_noop() {
        let idx = DedupIndex::disabled();
        idx.record(entry("a.jpg", 100, 1));
        assert!(!idx.contains(&Fingerprint {
            size: 100,
            mtime_ns: 1,
            name: "a.jpg".to_string()
        }));
        assert_eq!(idx.save().unwrap(), 0);
    }

    #[test]
    fn save_no_pending_no_write() {
        let tmp = tempdir();
        let path = tmp.join("index.rkyv");
        let idx = DedupIndex::open(path.clone()).unwrap();
        assert_eq!(idx.save().unwrap(), 0);
        assert!(!path.exists());
    }

    fn tempdir() -> PathBuf {
        let p = std::env::temp_dir().join(format!(
            "fcp-index-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&p).unwrap();
        p
    }
}
