//! wallreco — tag wallpapers with #hashtags for their colours, mood and content.

mod analyse;
mod cli;
mod naming;
mod sample;
mod ui;
mod undo;

#[cfg(feature = "vision")]
mod vision;

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::{Context, Result, bail};
use clap::Parser;
use rayon::prelude::*;
use walkdir::WalkDir;

use crate::analyse::Metrics;
use crate::cli::Args;

const EXTENSIONS: &[&str] = &["jpg", "jpeg", "png", "gif", "webp", "bmp", "avif"];

/// Files handled per pass. Bounds peak memory: only this many decoded images
/// and CLIP tensors are alive at once, whatever the size of the folder.
const CHUNK: usize = 128;

/// Heuristic scene tags CLIP is simply better at, paired with the vocabulary
/// category that overrules them.
///
/// The colour rules were tuned to be shy, but they still disagree with CLIP
/// sometimes — a lamplit room reads as `night`, a green field as `forest`.
/// When the matching CLIP category actually answered, its answer wins; when it
/// stayed silent, the heuristic is all we have and is kept.
const OVERRULED: &[(&str, &[&str])] =
    &[("time", &["night", "sunset"]), ("setting", &["forest", "nature", "space", "sky"])];

fn main() -> ExitCode {
    let args = Args::parse();
    match run(&args) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("{} {e:#}", ui::red("error:"));
            ExitCode::FAILURE
        }
    }
}

fn run(args: &Args) -> Result<ExitCode> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(args.jobs())
        .build_global()
        .ok();

    if args.undo {
        return undo_last(args);
    }
    if args.vision_setup {
        return setup_vision();
    }
    if args.paths.is_empty() {
        bail!("nothing to do: give me a directory to scan (see --help)");
    }

    let files = collect(&args.paths)?;
    if files.is_empty() {
        bail!("no images found in {}", list_paths(&args.paths));
    }

    let (todo, already_tagged) = if args.retag || args.explain {
        (files, 0)
    } else {
        let total = files.len();
        let todo: Vec<PathBuf> = files
            .into_iter()
            .filter(|f| !naming::is_tagged(&file_name(f)))
            .collect();
        let skipped = total - todo.len();
        (todo, skipped)
    };

    if todo.is_empty() {
        println!(
            "nothing to do: {already_tagged} file(s) already tagged {}",
            ui::dim("(use --retag to redo them)")
        );
        return Ok(ExitCode::SUCCESS);
    }

    let mut engine = Engine::new(args)?;
    let results = engine.process(args, &todo)?;

    if args.explain {
        report_explain(&results);
        return Ok(ExitCode::SUCCESS);
    }

    apply(args, results, already_tagged)
}

// ---------------------------------------------------------------------------
// gathering work
// ---------------------------------------------------------------------------

fn has_image_extension(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| EXTENSIONS.iter().any(|k| k.eq_ignore_ascii_case(e)))
}

fn collect(paths: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let mut out = BTreeSet::new();
    for path in paths {
        if !path.exists() {
            bail!("no such file or directory: {}", path.display());
        }
        if path.is_file() {
            if !has_image_extension(path) {
                bail!("{} is not an image wallreco can read", path.display());
            }
            out.insert(path.clone());
            continue;
        }
        // Hidden directories are caches, not collections: `.thumbnails` next
        // to a wallpaper folder would otherwise get tagged and renamed too.
        // A hidden path named explicitly on the command line is still scanned.
        let walker = WalkDir::new(path)
            .follow_links(false)
            .into_iter()
            .filter_entry(|e| e.depth() == 0 || !is_hidden(e.file_name()));
        for entry in walker.filter_map(Result::ok) {
            if entry.file_type().is_file() && has_image_extension(entry.path()) {
                out.insert(entry.into_path());
            }
        }
    }
    Ok(out.into_iter().collect())
}

fn is_hidden(name: &std::ffi::OsStr) -> bool {
    name.to_string_lossy().starts_with('.')
}

fn file_name(p: &Path) -> String {
    p.file_name().unwrap_or_default().to_string_lossy().into_owned()
}

fn list_paths(paths: &[PathBuf]) -> String {
    paths.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join(", ")
}

// ---------------------------------------------------------------------------
// analysis
// ---------------------------------------------------------------------------

struct Outcome {
    path: PathBuf,
    tags: Vec<String>,
    metrics: Option<Metrics>,
    #[cfg(feature = "vision")]
    scored: Vec<vision::Scored>,
    error: Option<String>,
}

struct Engine {
    #[cfg(feature = "vision")]
    clip: Option<vision::Clip>,
}

impl Engine {
    fn new(args: &Args) -> Result<Engine> {
        #[cfg(feature = "vision")]
        {
            if !args.vision {
                return Ok(Engine { clip: None });
            }
            let dir = vision::download::model_dir();
            let vocab = load_vocab()?;
            if !args.quiet {
                eprintln!(
                    "{} {} tags over {} categories",
                    ui::dim("vocabulary:"),
                    vocab.tag_count(),
                    vocab.categories.len()
                );
            }
            let clip = vision::Clip::load(&dir, vocab, args.jobs())?;
            return Ok(Engine { clip: Some(clip) });
        }
        #[cfg(not(feature = "vision"))]
        {
            if args.vision || args.vision_setup {
                bail!("this build has no vision support (compiled with --no-default-features)");
            }
            Ok(Engine {})
        }
    }

    fn process(&mut self, args: &Args, files: &[PathBuf]) -> Result<Vec<Outcome>> {
        let bar = if args.quiet {
            indicatif::ProgressBar::hidden()
        } else {
            ui::progress(files.len() as u64, "analysing")
        };

        let mut results = Vec::with_capacity(files.len());
        for chunk in files.chunks(CHUNK) {
            let mut batch = self.analyse_chunk(args, chunk, &bar);
            #[cfg(feature = "vision")]
            self.add_vision_tags(&mut batch)?;
            results.extend(batch.into_iter().map(|p| p.outcome));
        }
        bar.finish_and_clear();
        Ok(results)
    }

    /// Decode and measure one chunk, every core at once.
    fn analyse_chunk(
        &self,
        args: &Args,
        chunk: &[PathBuf],
        bar: &indicatif::ProgressBar,
    ) -> Vec<Prepared> {
        #[cfg(feature = "vision")]
        let want_pixels = self.clip.is_some();

        chunk
            .par_iter()
            .map(|path| {
                let outcome = |tags, metrics, error| Outcome {
                    path: path.clone(),
                    tags,
                    metrics,
                    #[cfg(feature = "vision")]
                    scored: Vec::new(),
                    error,
                };
                let prepared = match sample::decode(path) {
                    Err(e) => Prepared {
                        outcome: outcome(Vec::new(), None, Some(format!("{e:#}"))),
                        #[cfg(feature = "vision")]
                        pixels: None,
                    },
                    Ok(img) => {
                        let grid = sample::to_grid(&img, args.grid);
                        let a = analyse::analyse(&grid);
                        Prepared {
                            outcome: outcome(a.tags, Some(a.metrics), None),
                            // Prepared here, inside the parallel pass, so the
                            // session never waits on pixel work.
                            #[cfg(feature = "vision")]
                            pixels: want_pixels.then(|| vision::Clip::prepare(&img)),
                        }
                    }
                };
                bar.inc(1);
                prepared
            })
            .collect()
    }

    #[cfg(feature = "vision")]
    fn add_vision_tags(&mut self, batch: &mut [Prepared]) -> Result<()> {
        let Some(clip) = self.clip.as_mut() else {
            return Ok(());
        };

        // Only the files that decoded have pixels; keep track of which is which
        // so embeddings land back on the right outcome.
        let indices: Vec<usize> =
            batch.iter().enumerate().filter(|(_, p)| p.pixels.is_some()).map(|(i, _)| i).collect();

        for window in indices.chunks(vision::BATCH) {
            let pixels: Vec<_> =
                window.iter().map(|&i| batch[i].pixels.take().expect("filtered above")).collect();
            let embeddings = clip.embed(&pixels)?;
            for (&i, embedding) in window.iter().zip(embeddings) {
                let scored = clip.score(&embedding);
                merge_vision(&mut batch[i].outcome.tags, &scored);
                batch[i].outcome.scored = scored;
            }
        }
        Ok(())
    }
}

struct Prepared {
    outcome: Outcome,
    #[cfg(feature = "vision")]
    pixels: Option<vision::Pixels>,
}

/// Fold CLIP's answers into the colour tags, letting it settle the arguments
/// it is qualified to settle.
#[cfg(feature = "vision")]
fn merge_vision(tags: &mut Vec<String>, scored: &[vision::Scored]) {
    for (category, overruled) in OVERRULED {
        if scored.iter().any(|s| s.category == *category) {
            tags.retain(|t| !overruled.contains(&t.as_str()));
        }
    }
    for s in scored {
        if !tags.iter().any(|t| t == &s.tag) {
            tags.push(s.tag.clone());
        }
    }
}

#[cfg(feature = "vision")]
fn load_vocab() -> Result<vision::Vocab> {
    const BUILTIN: &str = include_str!("../labels.txt");

    let user = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))
        .map(|d| d.join("wallreco/labels.txt"));

    if let Some(path) = user.filter(|p| p.exists()) {
        let text =
            std::fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
        return vision::Vocab::parse(&text).with_context(|| format!("in {}", path.display()));
    }
    vision::Vocab::parse(BUILTIN).context("in the built-in vocabulary")
}

// ---------------------------------------------------------------------------
// applying
// ---------------------------------------------------------------------------

/// Where a file should end up, or `None` if the name would not change.
fn target(path: &Path, tags: &[String], retag: bool, max_tags: usize) -> Option<PathBuf> {
    let name = file_name(path);
    let (stem, ext) = name.rsplit_once('.')?;
    let base = if retag { naming::strip_tags(stem) } else { stem.to_string() };
    let composed = naming::compose(&base, ext, tags, max_tags)?;
    if composed == name {
        return None;
    }

    let dir = path.parent().unwrap_or(Path::new("."));
    let mut candidate = dir.join(&composed);
    if !candidate.exists() {
        return Some(candidate);
    }
    // Another wallpaper already earned this exact name; number this one.
    let (cstem, cext) = composed.rsplit_once('.')?;
    for n in 2..100 {
        candidate = dir.join(format!("{cstem}-{n}.{cext}"));
        if !candidate.exists() {
            return Some(candidate);
        }
    }
    None
}

#[derive(Default)]
struct Tally {
    renamed: usize,
    would_rename: usize,
    unchanged: usize,
    untagged: usize,
    failed: usize,
    unreadable: usize,
}

fn apply(args: &Args, results: Vec<Outcome>, already_tagged: usize) -> Result<ExitCode> {
    let dry = args.is_dry_run();
    let mut tally = Tally::default();
    let mut log = Vec::new();

    for r in &results {
        if let Some(err) = &r.error {
            tally.unreadable += 1;
            eprintln!("{} {}: {err}", ui::red("unreadable:"), ui::dim(&file_name(&r.path)));
            continue;
        }
        if r.tags.is_empty() {
            tally.untagged += 1;
            if args.json {
                print_json(&r.path, &r.tags, None);
            }
            continue;
        }

        let Some(new_path) = target(&r.path, &r.tags, args.retag, args.max_tags) else {
            tally.unchanged += 1;
            if args.json {
                print_json(&r.path, &r.tags, None);
            }
            continue;
        };

        if dry {
            tally.would_rename += 1;
        } else if let Err(e) = std::fs::rename(&r.path, &new_path) {
            tally.failed += 1;
            eprintln!("{} {}: {e}", ui::red("failed:"), ui::dim(&file_name(&r.path)));
            continue;
        } else {
            tally.renamed += 1;
            log.push(undo::Rename { from: r.path.clone(), to: new_path.clone() });
        }

        if args.json {
            print_json(&r.path, &r.tags, Some(&new_path));
        } else if !args.quiet {
            println!("  {}", ui::dim(&file_name(&r.path)));
            println!("    {} {}", ui::green("->"), ui::tags(&r.tags));
        }
    }

    if !log.is_empty() {
        match undo::save(log) {
            Ok(_) => {}
            Err(e) => eprintln!("{} could not record this run for --undo: {e:#}", ui::yellow("warning:")),
        }
    }

    if !args.json {
        print_summary(&tally, already_tagged, dry);
    }
    Ok(if tally.failed > 0 { ExitCode::FAILURE } else { ExitCode::SUCCESS })
}

fn print_json(path: &Path, tags: &[String], renamed_to: Option<&Path>) {
    let value = serde_json::json!({
        "path": path,
        "tags": tags,
        "renamed_to": renamed_to,
    });
    println!("{value}");
}

fn print_summary(tally: &Tally, already_tagged: usize, dry: bool) {
    let line = |label: &str, n: usize| {
        if n > 0 {
            println!("  {:<12} {n}", format!("{label}:"));
        }
    };
    println!();
    if dry {
        println!("{}", ui::bold("dry run — nothing was renamed"));
        line("would tag", tally.would_rename);
    } else {
        line("tagged", tally.renamed);
    }
    line("skipped", already_tagged);
    line("unchanged", tally.unchanged);
    line("no tags", tally.untagged);
    line("unreadable", tally.unreadable);
    line("failed", tally.failed);

    if already_tagged > 0 && tally.renamed == 0 && tally.would_rename == 0 {
        println!("{}", ui::dim("  (--retag re-computes tags for files that already have them)"));
    }
    if tally.renamed > 0 {
        println!("{}", ui::dim("  wallreco --undo puts these names back"));
    }
}

// ---------------------------------------------------------------------------
// --explain
// ---------------------------------------------------------------------------

fn report_explain(results: &[Outcome]) {
    for r in results {
        println!("{}", ui::bold(&file_name(&r.path)));
        if let Some(err) = &r.error {
            println!("  {} {err}", ui::red("unreadable:"));
            continue;
        }
        println!("  tags      {}", ui::tags(&r.tags));

        if let Some(m) = &r.metrics {
            println!("  {}", ui::dim("— colour statistics —"));
            println!(
                "  light     mean {:.3}  sd {:.3}  bright {:.1}%  dark<0.26 light>0.70",
                m.mean_l,
                m.sd_l,
                m.bright_frac * 100.0
            );
            println!(
                "  colour    chroma {:.3}  coloured {:.1}%  grey {:.1}%  vivid {:.1}%",
                m.mean_c,
                m.chroma_frac * 100.0,
                m.achrom_frac * 100.0,
                m.vivid_frac * 100.0
            );
            println!(
                "  texture   detail {:.3}  edges {:.1}%  flat {:.1}%  distinct {}",
                m.detail,
                m.edge_frac * 100.0,
                m.flat_frac * 100.0,
                m.distinct
            );
            println!(
                "  layout    thirds L {:.2}/{:.2}/{:.2}  ramp {:.3}  fit {:.3}",
                m.zone_l[0], m.zone_l[1], m.zone_l[2], m.ramp, m.grad_res
            );
            println!(
                "  balance   warm {:.0}%  cool {:.0}%  green {:.3}  clusters {}",
                m.warm_ratio * 100.0,
                m.cool_ratio * 100.0,
                m.green_frac,
                m.n_clusters
            );
            for (hue, frac, name) in &m.clusters {
                println!(
                    "  hue       {:>5.0}deg  {:>5.1}% of pixels  -> {}",
                    hue,
                    frac * 100.0,
                    name.map_or_else(|| ui::dim("below threshold"), |n| ui::blue(n))
                );
            }
        }

        #[cfg(feature = "vision")]
        if !r.scored.is_empty() {
            println!("  {}", ui::dim("— what CLIP saw —"));
            for s in &r.scored {
                println!(
                    "  {:<9} {} {}",
                    s.category,
                    ui::blue(&format!("#{}", s.tag)),
                    ui::dim(&format!("{:.0}% within category", s.probability * 100.0))
                );
            }
        }
        println!();
    }
}

// ---------------------------------------------------------------------------
// subcommands
// ---------------------------------------------------------------------------

fn undo_last(args: &Args) -> Result<ExitCode> {
    if !undo::record_exists() {
        bail!("no previous run to undo");
    }
    let outcome = undo::revert(args.is_dry_run())?;
    if args.is_dry_run() {
        println!("would restore {} file(s)", outcome.restored);
    } else {
        println!("restored {} file(s)", outcome.restored);
    }
    if outcome.missing > 0 {
        println!("{}", ui::dim(&format!("  {} moved or renamed since, left alone", outcome.missing)));
    }
    for path in &outcome.blocked {
        eprintln!("{} {} is taken", ui::yellow("skipped:"), path.display());
    }
    Ok(ExitCode::SUCCESS)
}

#[cfg(feature = "vision")]
fn setup_vision() -> Result<ExitCode> {
    let dir = vision::download::model_dir();
    if vision::download::is_installed(&dir) {
        println!("already installed in {}", dir.display());
    } else {
        println!(
            "downloading CLIP ViT-B/32 ({} MB) into {}",
            vision::download::total_bytes() / 1_000_000,
            dir.display()
        );
        vision::download::install(&dir)?;
        println!("{}", ui::green("done"));
    }

    // Embedding the vocabulary now means the first real run starts instantly,
    // and a broken labels file is reported here rather than mid-scan.
    let vocab = load_vocab()?;
    println!("embedding {} prompts...", vocab.prompt_count());
    vision::Clip::load(&dir, vocab, num_threads())?;
    println!("{} run `wallreco --vision DIR` to use it", ui::green("ready."));
    Ok(ExitCode::SUCCESS)
}

#[cfg(not(feature = "vision"))]
fn setup_vision() -> Result<ExitCode> {
    bail!("this build has no vision support (compiled with --no-default-features)")
}

fn num_threads() -> usize {
    std::thread::available_parallelism().map_or(4, |n| n.get())
}
