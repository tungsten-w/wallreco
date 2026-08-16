# wallreco

Tag wallpapers with English `#hashtags` describing their dominant colours, mood
and content, so a rofi picker can filter them by typing a keyword.

```
sunset.png  ->  sunset-#orange-#blue-#warm-#sunset-#sky.png
```

Tags are written into the *filename* on purpose: launchers like lumen feed bare
filenames to rofi, so whatever is in the name is what rofi can search.

## Install

```sh
git clone https://github.com/tungsten-w/wallreco.git
cd wallreco
chmod +x wallpaper_recognition.sh
```

Requires `bash`, `awk` and ImageMagick (`sudo pacman -S imagemagick`). Nothing
else for the default colour stage.

## Usage

```sh
./wallpaper_recognition.sh ~/Pictures/Wallpapers          # tag new wallpapers
./wallpaper_recognition.sh --retag -n ~/Pictures/Wallpapers  # preview a re-tag
./wallpaper_recognition.sh --retag ~/Pictures/Wallpapers/dark
```

Already-tagged files are skipped unless `--retag` is passed. Use `-n` /
`--dry-run` first — it prints every rename without touching a single file.

### Options

| Option | Effect |
| --- | --- |
| `-j, --jobs N` | parallel workers (default: one per core) |
| `-r, --retag` | strip existing `#tags` and recompute |
| `-n, --dry-run` | show what would be renamed, change nothing |
| `-m, --max-tags N` | keep at most N tags per file (default: all) |
| `-g, --grid N` | sampling grid size (default 64, min 24) |
| `-V, --vision` | also name the objects in the image (CLIP, offline) |
| `--vision-setup` | install the object recogniser (~600 MB, one time) |
| `--vision-refresh` | reload the vocabulary after editing the label file |
| `-q, --quiet` | only print errors and the summary |
| `-h, --help` | help |

## How it works

Two independent stages.

**Colours** — pure ImageMagick + awk, no dependencies. One `magick` call per
image renders a small pixel grid, and a single awk pass turns it into colour,
mood and scene tags. Runs one worker per core.

**Objects** (optional, `--vision`) — CLIP scores each image against a
vocabulary of prompts entirely offline on the CPU, naming what is actually in
the picture: cat, castle, city, robot… One model load covers the whole batch,
so it is a separate stage rather than part of the per-file worker.

## Tag vocabulary

| Group | Tags |
| --- | --- |
| colours | `black` `white` `gray` `red` `orange` `yellow` `green` `teal` `cyan` `blue` `purple` `pink` `brown` |
| mood | `dark` `light` `vibrant` `pastel` `muted` `monochrome` `warm` `cool` `colorful` |
| scene | `night` `space` `sunset` `sky` `nature` `forest` `gradient` `minimal` `pattern` |

The scene list is deliberately short. Only rules that survived checking against
hand-labelled wallpapers were kept: city, water, mountain, desert and snow all
collapsed into their backgrounds, so they were dropped rather than guessed at.
A wrong tag is worse than a missing one, because it puts the wrong wallpaper in
front of you in rofi. Use `--vision` for those.

## Supported formats

`jpg` `jpeg` `png` `gif` `webp` `bmp` `avif` — searched recursively in every
directory given on the command line.

## Status

The colour stage is complete and used daily. The `--vision` stage needs
`wallpaper_vision.py` and `wallpaper_labels.txt` next to the script; those are
not in this repo yet.
