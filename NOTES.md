# wallreco — how it works, how to run it, and the full tag list

This is a working note on top of `README.md`: what the code actually does
internally, how to build and activate every feature, the complete tag
vocabulary, and what was checked/fixed during a stability pass.

## What it does, in one sentence

Point it at a folder of wallpapers and it renames each file to embed
`#hashtags` describing its colours, mood and (optionally) its content, so a
rofi-style launcher — which only ever sees bare filenames — can filter your
wallpapers by typing a keyword.

```
sunset.png  →  sunset-#orange-#blue-#warm-#sunset-#sky.png
```

## Build & activate

```sh
cd ~/.config/wallreco
cargo build --release
install -Dm755 target/release/wallreco ~/.local/bin/wallreco   # put it on PATH
```

That's the whole "activation" — it's a single static-ish binary, no daemon,
no service, no config file required. Two build modes:

- **Default** (`cargo build --release`): colours + CLIP object recognition.
  Pulls in `ort` (ONNX Runtime), `ndarray`, `tokenizers`, `ureq`.
- **Slim** (`cargo build --release --no-default-features`): colours only, no
  ONNX runtime linked in, no model ever downloaded. Use this if you don't
  want the ~600 MB CLIP weights or don't care about object tags.

To turn on object recognition at runtime (only needed once, ever):

```sh
wallreco --vision-setup
```

This downloads `vision_model.onnx` (~352 MB), `text_model.onnx` (~254 MB) and
`tokenizer.json` (~2 MB) from the `Xenova/clip-vit-base-patch32` HuggingFace
repo into `~/.local/share/wallreco/models/clip-vit-base-patch32` (override
with `$WALLRECO_MODEL_DIR`), then embeds the prompt vocabulary once so the
first real `--vision` run starts instantly. After that everything runs fully
offline. From then on, add `-V`/`--vision` to any run to get object tags in
addition to the colour ones:

```sh
wallreco ~/Pictures/Wallpapers               # colours only
wallreco -V ~/Pictures/Wallpapers             # colours + objects
```

I ran `--vision-setup` on this machine as part of testing, so it's already
installed and ready to use.

## How the code is structured

```
src/main.rs        orchestration: CLI → collect files → analyse → rename
src/cli.rs          clap argument definitions
src/sample.rs       decode an image, flatten alpha, box-average to a small grid
src/analyse.rs       colour/mood/scene tags from pure pixel statistics
src/naming.rs        turn a tag list into `stem-#tag-#tag.ext`, and back
src/undo.rs          JSON log of the last run's renames, for `--undo`
src/ui.rs            terminal colour + progress bar
src/vision/mod.rs    CLIP embedding + scoring (feature-gated)
src/vision/vocab.rs   parses labels.txt into categories/prompts/thresholds
src/vision/download.rs  fetches and verifies the CLIP ONNX weights
```

### Stage 1 — colours (`sample.rs` + `analyse.rs`), always on

1. `sample::decode` opens the file with the `image` crate, applies EXIF
   orientation, and flattens any alpha channel onto black (so transparent
   corners don't skew the colour read).
2. `sample::to_grid` box-averages the image down to a small square (default
   64×64, tunable with `-g`/`--grid`, 24–512). A nearest-neighbour prepass
   caps how much data the real filter touches on huge (8K) wallpapers before
   a proper Triangle-filter area average runs on what's left.
3. `analyse::analyse` does **one pass** over that grid building: a 72-bin hue
   histogram (weighted by chroma × value, so muddy/dark pixels barely count),
   luminance mean/variance, per-third-of-image luminance and hue (top/middle/
   bottom), edge density, a "flat area" fraction (identical neighbouring
   pixels — the poster-art tell), and a linear-regression fit of how well row
   luminance follows a vertical ramp (the gradient/sunset tell).
4. Those statistics run through hand-tuned thresholds — ported *literally*
   from the project's original bash/awk analyser and treated as calibrated
   data, not something to re-derive — to emit colour, mood and scene tags.

This stage is pure math, no model, no network, and is what you get from a
plain `wallreco DIR` with no `-V`.

### Stage 2 — objects (`vision/`), opt-in with `-V`

1. Each image is resized/cropped to CLIP's 224×224 input and normalised
   (`vision::Clip::prepare`) — cheap, so it runs under rayon in parallel with
   everything else.
2. Prompts from `labels.txt` are embedded **once** (`embed_prompts`) through
   CLIP's text tower, cached to disk at
   `~/.local/share/wallreco/models/.../prompts-<hash>.bin` keyed by a hash of
   the vocabulary text — edit `labels.txt` and the cache invalidates itself
   automatically on the next run. The (large) text tower is then dropped; it
   has no reason to stay resident while images are being scored.
3. Images are batched (16 at a time, `vision::BATCH`) through the vision
   tower (`Clip::embed`), each producing an L2-normalised embedding.
4. `Clip::score` compares that embedding against the cached prompt
   embeddings **per category**, softmaxing within the category rather than
   across the whole vocabulary — "forest, city, beach or desert?" gives a
   confident answer where "which of these 76 things is it?" spreads
   probability too thin to ever clear a threshold. Categories can `require`
   an earlier category to have answered first (`time`/`weather` require
   `setting`), so a samurai portrait doesn't get called `#night` just because
   "a scene at night" was softmax's best guess among four unrelated options.
5. `~` decoy prompts are scored like everything else but never emitted —
   they're what makes an abstract gradient not get called `#forest` just
   because forest was the closest of four bad choices.
6. Back in `main.rs::merge_vision`, wherever CLIP's `time` or `setting`
   category actually answered, its answer overrules the colour heuristic's
   guess for the tags it's specifically better at (`night`, `sunset`,
   `forest`, `nature`, `space`, `sky`); where CLIP stayed silent, the colour
   heuristic's tag is kept.

### Applying the result (`main.rs::apply` / `naming.rs`)

- `naming::compose` builds `stem-#tag-#tag....ext`, dropping trailing tags
  that would push the filename past 250 bytes, and refuses to produce a
  useless name if not even one tag fits (file is left untouched, reported as
  "unchanged").
- If the target name is already taken by something else, it numbers the file
  (`-2`, `-3`, …) instead of overwriting it.
- Every actual rename is appended to a JSON log at
  `~/.local/state/wallreco/last-run.json` (or `$XDG_STATE_HOME`), which is
  what `--undo` replays in reverse. `--undo` only restores files that are
  still exactly where the log left them and whose original name is free —
  anything moved or reused since is reported and left alone rather than
  guessed at.

## Safety nets

Renaming is the one destructive thing this tool does:

- `-n`/`--dry-run` (and `--explain`, which implies it) previews without
  touching a single file.
- `--undo` reverses the last run, once.
- Files that already contain a `#` are skipped by default — you have to pass
  `--retag` to recompute them, so re-running the tool on an already-tagged
  folder is a safe no-op.
- Hidden directories (`.thumbnails`, etc.) are never scanned, so thumbnail
  caches never get renamed out from under whatever generated them.

## Options

| Option | Effect |
| --- | --- |
| `-j, --jobs N` | parallel workers (default: one per core, capped at 256 — see "stability" below) |
| `-r, --retag` | strip existing `#tags` and recompute |
| `-n, --dry-run` | show what would be renamed, change nothing |
| `-m, --max-tags N` | keep at most N tags per file (default: all) |
| `-g, --grid N` | sampling grid size, 24–512 (default 64; higher = finer, slower) |
| `-V, --vision` | also name the objects in the image (CLIP, offline after setup) |
| `--vision-setup` | download the object recogniser once, then exit |
| `--explain` | print every metric/confidence behind the tags (implies `-n`) |
| `--undo` | restore the filenames from the last run |
| `--json` | one JSON object per file on stdout, for scripting |
| `-q, --quiet` | only errors and the summary |

Note: `-V` is "vision"; `--version` only has the long form on purpose, so it
doesn't silently share a letter with `-V`.

## The complete tag vocabulary

### Colours (always on) — 13
`black` `white` `gray` `red` `orange` `yellow` `green` `teal` `cyan` `blue`
`purple` `pink` `brown`

### Mood (always on) — 9
`dark` `light` `vibrant` `pastel` `muted` `monochrome` `warm` `cool`
`colorful`

### Scene, from colour statistics alone (always on) — 9
`night` `space` `sunset` `sky` `nature` `forest` `gradient` `minimal`
`pattern`

This scene list is deliberately short: rules for city/water/mountain/desert/
snow were tried against hand-labelled wallpapers and dropped because they
kept collapsing into their backgrounds — a wrong tag is worse than a missing
one, since it puts the wrong wallpaper in front of you. That's what `-V`
is for.

### Vision, only with `-V`/`--vision` — 76, in 6 categories

Defined in `labels.txt`, editable (copy it to
`~/.config/wallreco/labels.txt`).

- **style** (pick at most one): `anime` `pixelart` `painting` `photo` `3d`
  `lineart` `vector`
- **people** (pick at most one): `character` `girl` `boy` `crowd`
- **subject** (up to 2 at once): `cat` `dog` `fox` `wolf` `bird` `horse`
  `fish` `butterfly` `dragon` `robot` `samurai` `knight` `astronaut` `car`
  `motorcycle` `plane` `ship` `spaceship` `castle` `temple` `lighthouse`
  `bridge` `skull` `guitar` `flower` `tree` `moon` `planet` `stars` `clouds`
  `lightning` `fire` `waterfall` `train` `food` `book` `window` `sword`
  `eye`
- **setting** (pick at most one): `forest` `jungle` `beach` `ocean` `lake`
  `river` `mountains` `desert` `snow` `field` `city` `village` `ruins`
  `interior` `cave` `underwater` `space` `garden` `sky`
- **time** (pick at most one; only asked once `setting` has answered):
  `night` `sunset` `day`
- **weather** (pick at most one; only asked once `setting` has answered):
  `rain` `fog` `storm` `snowfall`

### Tuning the vocabulary yourself

```sh
cp labels.txt ~/.config/wallreco/labels.txt
$EDITOR ~/.config/wallreco/labels.txt
```

wallreco notices the edit (it hashes the file) and re-embeds automatically on
the next run. `--explain -V some-file.png` prints the confidence behind every
tag CLIP considered, which is the tool for deciding whether a *threshold* or
a *prompt* is at fault when a tag fires where it shouldn't — a better decoy
usually beats a higher threshold, since a threshold turns the category off
for every image while a decoy only intercepts the ones that genuinely belong
to it.

## Stability pass — what I checked and fixed

Full read of every source file, `cargo build --release`, `cargo test
--release` (all 14 unit tests pass), and `cargo clippy` at both default and
`pedantic` level. The codebase was already careful: every fallible path
returns `Result` with context, no unexplained panics, divisions guarded
against zero, atomic-rename-based downloads, self-healing embedding cache
(a corrupt/partial cache file just fails its magic-byte check and rebuilds).

I then ran it for real: colours-only and colours+vision dry runs, `--explain`
on single files, a full rename → `--undo` → rename-again round trip, `--json`
output, and deliberate edge cases (missing path, empty directory, a
non-image file passed directly, a corrupted/truncated image, grid values
outside `24..=512`, `-j 0`). All of those were already handled gracefully —
clear error messages, correct exit codes, no crashes.

I also ran it at real scale against `~/Pictures/Wallpapers` (dry-run only, so
nothing there was touched): 447 non-thumbnail wallpapers, colours-only in
**7.4 s**; the same set forced through `--retag -V` (CLIP object recognition
on every file) in **2 m 9 s** on this machine's 4 cores — in line with the
16-core numbers in `README.md` once scaled down, and every tag it produced
looked sane on manual spot-checking.

One real bug turned up under adversarial input, and I fixed it:

**`-j`/`--jobs` had no upper bound.** `wallreco -j 999999 …` builds a global
rayon thread pool with that many OS threads. On this 4-core machine it spent
minutes standing up threads, climbed past 380% CPU and hundreds of MB of RSS
and counting, and had to be killed — a typo (an extra digit) turns into a
local denial-of-service. Fixed in `src/cli.rs` by capping `Args::jobs()` at
256, a ceiling well above any real machine's core count but far short of
"exhaust the system." `-j 999999` now returns in well under half a second.

Alongside that I hardened the one genuinely destructive code path:

**The `--undo` log was written non-atomically.** `undo::save` used to
`fs::write` straight over `~/.local/state/wallreco/last-run.json`. A crash or
power loss mid-write would leave a truncated file — not fatal (the next
`--undo` would just fail to parse it and error out), but it's the one file
standing between you and a folder full of renamed files, so it's now written
to a `.part` sibling and atomically renamed into place, the same pattern the
model downloader already used for weight files.

I also cleared a small number of pure-cosmetic `clippy` warnings in
`main.rs` (a needless `return`, two closures that were just wrapping a
method call) with zero behavioural change. I deliberately left the handful
of remaining `clippy` style suggestions inside `analyse.rs` and the CLIP
preprocessing constants in `vision/mod.rs` alone — `analyse.rs`'s thresholds
are explicitly documented as tuned-and-ported data rather than something to
freely rewrite, and the CLIP mean/std constants are transcribed from the
model's own `preprocessor_config.json`, so touching either for a style nit
carried more risk than it was worth.

I then independently re-verified all of the above from scratch — full
release rebuild, `cargo test --release` (14/14 pass), `cargo clippy`, and a
live functional pass through the built binary (tag → dry-run → undo →
re-undo-should-fail → bad-path → `-j 999999` timing → `--version`), all on
throwaway files in the scratch directory so nothing real was touched. Every
claim above held up.

That pass also turned up one real gap: **the slim build
(`--no-default-features`) had 4 compiler warnings** — an unused `Context`
import, an unnecessary `mut`, and a dead `OVERRULED` const and `num_threads`
function — all because those items are only referenced from code paths that
only exist when the `vision` feature is on. `cargo build --release` (the
default, vision-on) was always clean; nobody had built the slim
configuration and looked. Fixed in `src/main.rs` by `#[cfg(feature =
"vision")]`-gating the import, constant and function, and
`#[cfg_attr(not(feature = "vision"), allow(unused_mut))]` on the one `let
mut` that only needs mutability when vision tags get merged in. Both build
configurations now compile with zero warnings.

Everything above is committed to the working tree but not yet as a git
commit — say the word if you'd like these changes committed.
