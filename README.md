# wallreco

Tag wallpapers with English `#hashtags` describing their dominant colours,
mood and content, so a rofi picker can filter them by typing a keyword.

```
sunset.png  ->  sunset-#orange-#blue-#warm-#sunset-#sky.png
```

Tags are written into the *filename* on purpose: launchers like lumen feed bare
filenames to rofi, so whatever is in the name is what rofi can search.

## Install

```sh
git clone https://github.com/tungsten-w/wallreco.git
cd wallreco
cargo build --release
install -Dm755 target/release/wallreco ~/.local/bin/wallreco
```

No ImageMagick, no Python, no runtime dependencies — a single binary that
decodes JPEG, PNG, GIF, WebP, BMP and AVIF itself.

Optional object recognition needs the CLIP weights, downloaded once:

```sh
wallreco --vision-setup      # ~600 MB into ~/.local/share/wallreco
```

Building without it (`cargo build --release --no-default-features`) drops the
ONNX runtime entirely and leaves the colour stage intact.

## Usage

```sh
wallreco ~/Pictures/Wallpapers              # tag new wallpapers
wallreco -n --retag ~/Pictures/Wallpapers   # preview a full re-tag
wallreco --vision ~/Pictures/Wallpapers     # colours + objects
wallreco --explain sunset.png               # why those tags?
wallreco --undo                             # put the last run's names back
```

Files that already contain a `#` are skipped unless you pass `--retag`. Hidden
directories are never scanned, so thumbnail caches are left alone.

Renaming is the one destructive thing this tool does, so there are two safety
nets: `-n` previews without touching anything, and `--undo` reverses the last
run.

### Options

| Option | Effect |
| --- | --- |
| `-j, --jobs N` | parallel workers (default: one per core) |
| `-r, --retag` | strip existing `#tags` and recompute |
| `-n, --dry-run` | show what would be renamed, change nothing |
| `-m, --max-tags N` | keep at most N tags per file (default: all) |
| `-g, --grid N` | sampling grid size (default 64, 24–512) |
| `-V, --vision` | also name the objects in the image (CLIP, offline) |
| `--vision-setup` | download the object recogniser, then exit |
| `--explain` | print every metric and confidence behind the tags |
| `--undo` | restore the filenames from the last run |
| `--json` | one JSON object per file, for scripting |
| `-q, --quiet` | only errors and the summary |

## How it works

Two independent stages.

**Colours** run on a 64×64 sample of each image. One pass builds a 72-bin hue
histogram, luminance statistics, per-third layout, edge density and a flat-area
fraction; a set of thresholds turns those into colour, mood and scene tags. No
model, no network.

**Objects** (`--vision`) embed each image with CLIP ViT-B/32 through ONNX
Runtime and compare it against the prompt vocabulary in `labels.txt`. The text
side is embedded once and cached to disk, so only the vision tower runs per
image; rayon decodes and prepares pixels in parallel while the session works
through batches.

### What makes the tags precise

Zero-shot CLIP is only as good as the question you ask it. Three things do the
work here:

- **Per-category scoring.** Prompts are grouped (`style`, `people`, `subject`,
  `setting`, `time`, `weather`) and softmaxed within their group. "Forest, city,
  beach or desert?" gives a sharp answer; "which of these ninety things is it?"
  spreads the probability until nothing clears a useful threshold.
- **Decoys.** Every category carries `~` prompts that are scored but never
  emitted. They absorb images that match nothing, so an abstract gradient is not
  called a forest merely because forest was the closest of four bad options.
- **Category dependencies.** `time` and `weather` declare `requires=setting`:
  they are only asked once the image has been agreed to be a place. Without
  that, a samurai on a flat red background comes out `#night #storm`, because
  "a scene at night" genuinely is the closest option offered and softmax has to
  put its mass somewhere.

Where the two stages disagree about something CLIP is better at — time of day,
or what the setting is — CLIP wins. Where CLIP stays silent, the colour
heuristic is kept.

### Tuning it yourself

Copy `labels.txt` to `~/.config/wallreco/labels.txt` and edit. wallreco notices
the change and re-embeds the prompts on the next run. `--explain` shows the
confidence behind every tag, which is what you want when deciding whether a
threshold or a prompt is at fault:

```
samurai-red.png
  tags      #red #vibrant #warm #minimal #anime #samurai
  — what CLIP saw —
  style     #anime 45% within category
  subject   #samurai 91% within category
```

When a tag fires where it should not, a better decoy usually beats a higher
threshold: a threshold turns the category off for every image, a decoy only
takes the ones that genuinely belong to it.

## Tag vocabulary

| Group | Tags |
| --- | --- |
| colours | `black` `white` `gray` `red` `orange` `yellow` `green` `teal` `cyan` `blue` `purple` `pink` `brown` |
| mood | `dark` `light` `vibrant` `pastel` `muted` `monochrome` `warm` `cool` `colorful` |
| scene | `night` `space` `sunset` `sky` `nature` `forest` `gradient` `minimal` `pattern` |
| vision | 76 more — `anime` `pixelart` `cat` `dragon` `robot` `samurai` `castle` `city` `forest` `rain`… |

The scene list from colour statistics alone is deliberately short. Only rules
that survived checking against hand-labelled wallpapers were kept: city, water,
mountain, desert and snow all collapsed into their backgrounds, so they were
dropped rather than guessed at. A wrong tag is worse than a missing one,
because it puts the wrong wallpaper in front of you in rofi. `--vision` is how
those get named properly.

## Performance

385 wallpapers, 16 cores:

| | wall clock | CPU |
| --- | --- | --- |
| colours only | 5.3 s | 54 s |
| colours + CLIP | 20 s | 209 s |
| the previous shell version (colours only) | 12.5 s | 158 s |

## History

wallreco started as `wallpaper_recognition.sh`, a bash and ImageMagick script.
The Rust rewrite keeps that analyser's tuned thresholds — they were fitted
against hand-labelled wallpapers and are treated as data, not something to
re-derive — and adds the object recognition the shell version only had a broken
stub for. The original is in the git history.
