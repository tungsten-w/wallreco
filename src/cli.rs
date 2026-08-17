use std::path::PathBuf;

use clap::Parser;

const AFTER_HELP: &str = "\
Tag vocabulary (colours, always on):
  colours   black white gray red orange yellow green teal cyan blue purple
            pink brown
  mood      dark light vibrant pastel muted monochrome warm cool colorful
  scene     night space sunset sky nature forest gradient minimal pattern

  --vision adds what is actually in the picture — anime, cat, castle, city,
  robot, rain... from a vocabulary you can edit. Run --vision-setup once first.

Examples:
  wallreco ~/Pictures/Wallpapers                 tag new wallpapers
  wallreco -n --retag ~/Pictures/Wallpapers      preview a full re-tag
  wallreco -V ~/Pictures/Wallpapers              colours + objects
  wallreco -c -n ~/Pictures/Wallpapers           preview removing every tag
  wallreco --explain sunset.png                  show why those tags
  wallreco --undo                                put the last run's names back

Starting over:
  --clean removes the tags and stops, leaving the original filenames.
  --retag removes them and computes new ones in the same pass. Use --clean
  when you want the folder back the way it was, --retag when you just want
  the tags refreshed.
";

#[derive(Parser, Debug)]
#[command(
    name = "wallreco",
    version,
    // `-V` belongs to --vision here, so --version keeps only its long form
    // rather than quietly sharing a letter with it.
    disable_version_flag = true,
    about = "Tag wallpapers with #hashtags for their colours, mood and content",
    long_about = "Tag wallpapers with English #hashtags describing their dominant colours, mood \
                  and content, so a rofi picker can filter them by typing a keyword.\n\n  \
                  sunset.png  ->  sunset-#orange-#blue-#warm-#sunset-#sky.png\n\n\
                  Tags go in the filename because launchers feed bare filenames to rofi: \
                  whatever is in the name is what rofi can search.",
    after_help = AFTER_HELP
)]
pub struct Args {
    /// Directories to scan, or single images with --explain
    #[arg(value_name = "PATH")]
    pub paths: Vec<PathBuf>,

    /// Parallel workers [default: one per core]
    #[arg(short, long, value_name = "N")]
    pub jobs: Option<usize>,

    /// Strip existing #tags and recompute, instead of skipping tagged files
    #[arg(short, long)]
    pub retag: bool,

    /// Remove every #tag from the filenames and stop; tag nothing
    #[arg(short = 'c', long, conflicts_with_all = ["retag", "vision", "explain"])]
    pub clean: bool,

    /// Show what would be renamed, change nothing
    #[arg(short = 'n', long)]
    pub dry_run: bool,

    /// Keep at most N tags per file [default: all]
    #[arg(short, long, value_name = "N", default_value_t = 0)]
    pub max_tags: usize,

    /// Sampling grid size; higher is finer but slower
    #[arg(short, long, value_name = "N", default_value_t = 64, value_parser = clap::value_parser!(u32).range(24..=512))]
    pub grid: u32,

    /// Also name the objects in the image (CLIP, offline)
    #[arg(short = 'V', long)]
    pub vision: bool,

    /// Download the object recogniser (~600 MB, once), then exit
    #[arg(long)]
    pub vision_setup: bool,

    /// Print every metric and confidence behind the tags; implies --dry-run
    #[arg(long)]
    pub explain: bool,

    /// Put back the names from the last run
    #[arg(long)]
    pub undo: bool,

    /// One JSON object per file on stdout, for scripting
    #[arg(long, conflicts_with = "explain")]
    pub json: bool,

    /// Only errors and the summary
    #[arg(short, long, conflicts_with = "explain")]
    pub quiet: bool,

    /// Print version
    #[arg(long, action = clap::ArgAction::Version)]
    pub version: Option<bool>,
}

/// Hard ceiling on `-j`, independent of what the user asks for.
///
/// Every worker is a real OS thread with its own stack, and each one that
/// touches the vision stage holds a CLIP session's worth of ONNX Runtime
/// state. A typo like `-j 999999` should give an error, not spend minutes and
/// gigabytes standing up threads nothing can schedule.
const MAX_JOBS: usize = 256;

impl Args {
    pub fn jobs(&self) -> usize {
        self.jobs
            .filter(|j| *j > 0)
            .unwrap_or_else(|| std::thread::available_parallelism().map_or(4, |n| n.get()))
            .min(MAX_JOBS)
    }

    /// `--explain` is a read-only view: it should never move a file out from
    /// under the person inspecting it. `--json` still renames, so a script can
    /// act on what actually happened.
    pub fn is_dry_run(&self) -> bool {
        self.dry_run || self.explain
    }
}
