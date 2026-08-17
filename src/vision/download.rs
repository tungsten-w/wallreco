//! Fetching the CLIP weights, once, on first use.

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use indicatif::{ProgressBar, ProgressStyle};

const REPO: &str = "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main";

/// The files that have to be on disk before the vision stage can run, with the
/// size upstream currently serves — used for the progress bar, and as a floor
/// to notice a file that is obviously not the model.
pub const FILES: &[(&str, &str, u64)] = &[
    ("vision_model.onnx", "onnx/vision_model.onnx", 351_685_709),
    ("text_model.onnx", "onnx/text_model.onnx", 254_058_553),
    ("tokenizer.json", "tokenizer.json", 2_224_119),
];

/// Anything at least this fraction of the expected size is accepted, so a
/// re-export upstream does not make an installed model look missing. Truncated
/// downloads are handled by the `.part` rename, not by this check.
fn plausible(actual: u64, expected: u64) -> bool {
    actual >= expected / 10 * 9
}

/// Where the weights live. Honours `WALLRECO_MODEL_DIR`, then the XDG data dir.
pub fn model_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("WALLRECO_MODEL_DIR") {
        return PathBuf::from(dir);
    }
    let base = std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/share")))
        .unwrap_or_else(|| PathBuf::from("."));
    base.join("wallreco/models/clip-vit-base-patch32")
}

/// Is every weight file present?
pub fn is_installed(dir: &Path) -> bool {
    FILES
        .iter()
        .all(|(name, _, size)| fs::metadata(dir.join(name)).is_ok_and(|m| plausible(m.len(), *size)))
}

pub fn total_bytes() -> u64 {
    FILES.iter().map(|(_, _, size)| size).sum()
}

/// Download whatever is missing into `dir`.
pub fn install(dir: &Path) -> Result<()> {
    fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;

    for (name, remote, size) in FILES {
        let dest = dir.join(name);
        if fs::metadata(&dest).is_ok_and(|m| plausible(m.len(), *size)) {
            continue;
        }
        fetch(&format!("{REPO}/{remote}"), &dest, *size)
            .with_context(|| format!("download {name}"))?;
    }
    Ok(())
}

fn fetch(url: &str, dest: &Path, expected: u64) -> Result<()> {
    let mut response = ureq::get(url).call().context("http request failed")?;
    if response.status() != 200 {
        bail!("server answered {}", response.status());
    }
    let total = response
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(expected);

    let bar = ProgressBar::new(total);
    bar.set_style(
        ProgressStyle::with_template(
            "  {msg:<20} [{bar:30.cyan/blue}] {bytes:>10}/{total_bytes} {bytes_per_sec:>12}",
        )
        .unwrap()
        .progress_chars("=> "),
    );
    bar.set_message(dest.file_name().unwrap_or_default().to_string_lossy().to_string());

    // Straight to a temporary next to the target, so an interrupted download
    // never leaves something that looks installed.
    let tmp = dest.with_extension("part");
    let mut file = fs::File::create(&tmp).with_context(|| format!("create {}", tmp.display()))?;
    let mut reader = response.body_mut().as_reader();
    let mut buf = vec![0u8; 256 * 1024];
    let mut written = 0u64;
    loop {
        let n = reader.read(&mut buf).context("read from server")?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n]).context("write to disk")?;
        written += n as u64;
        bar.set_position(written);
    }
    file.flush()?;
    drop(file);
    bar.finish();

    if written < total {
        let _ = fs::remove_file(&tmp);
        bail!("connection dropped after {written} of {total} bytes");
    }
    fs::rename(&tmp, dest).with_context(|| format!("install {}", dest.display()))?;
    Ok(())
}
