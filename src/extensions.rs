//! Default photo file extensions recognized by fcp.
//!
//! Both casings are listed because the filesystem walker compares the
//! extension byte-for-byte against this set. To recognize a new format,
//! add it here once.

pub const DEFAULT_PHOTO_EXTENSIONS: &[&str] = &[
    // RAW
    "raf", "RAF", // Fujifilm
    "nef", "NEF", // Nikon
    "cr2", "CR2", // Canon (CR2)
    "cr3", "CR3", // Canon (CR3)
    "arw", "ARW", // Sony
    "dng", "DNG", // Adobe / generic
    "rw2", "RW2", // Panasonic
    "orf", "ORF", // Olympus
    "pef", "PEF", // Pentax
    "raw", "RAW", // generic
    // Standard / processed
    "jpg", "JPG",
    "jpeg", "JPEG",
    "png", "PNG",
    "heic", "HEIC",
    "heif", "HEIF",
    "tiff", "TIFF",
    "tif", "TIF",
    "webp", "WEBP",
];
