//! Putting the previous run's filenames back.
//!
//! Tagging rewrites filenames in place, which is the one genuinely destructive
//! thing this tool does. The log makes that reversible with a single command.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct Rename {
    pub from: PathBuf,
    pub to: PathBuf,
}

#[derive(Serialize, Deserialize)]
pub struct Log {
    pub renames: Vec<Rename>,
}

fn log_path() -> PathBuf {
    let base = std::env::var_os("XDG_STATE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/state")))
        .unwrap_or_else(|| PathBuf::from("."));
    base.join("wallreco/last-run.json")
}

pub fn save(renames: Vec<Rename>) -> Result<PathBuf> {
    let path = log_path();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    fs::write(&path, serde_json::to_vec_pretty(&Log { renames })?)
        .with_context(|| format!("write {}", path.display()))?;
    Ok(path)
}

pub struct Outcome {
    pub restored: usize,
    /// Files that moved on, or were renamed again, since the run.
    pub missing: usize,
    /// The original name is taken by something else now.
    pub blocked: Vec<PathBuf>,
}

/// Reverse the last run. Only touches files that are still exactly where the
/// log left them and whose original name is free.
pub fn revert(dry_run: bool) -> Result<Outcome> {
    let path = log_path();
    let data = fs::read(&path)
        .with_context(|| format!("no record of a previous run at {}", path.display()))?;
    let log: Log = serde_json::from_slice(&data)
        .with_context(|| format!("parse {}", path.display()))?;

    let mut outcome = Outcome { restored: 0, missing: 0, blocked: Vec::new() };
    for r in &log.renames {
        if !r.to.exists() {
            outcome.missing += 1;
            continue;
        }
        if r.from.exists() {
            outcome.blocked.push(r.from.clone());
            continue;
        }
        if dry_run {
            outcome.restored += 1;
            continue;
        }
        match fs::rename(&r.to, &r.from) {
            Ok(()) => outcome.restored += 1,
            Err(_) => outcome.blocked.push(r.from.clone()),
        }
    }

    // A spent log would otherwise invite a second --undo that does nothing.
    if !dry_run && outcome.restored > 0 {
        let _ = fs::remove_file(&path);
    }
    Ok(outcome)
}

pub fn record_exists() -> bool {
    Path::new(&log_path()).exists()
}
