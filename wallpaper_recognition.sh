#!/usr/bin/env bash
#
# wallpaper_recognition.sh — tag wallpapers with English #hashtags describing
# their dominant colours, mood and content, so the rofi picker can filter them
# by typing a keyword.
#
#   sunset.png  ->  sunset-#orange-#blue-#warm-#sunset-#sky.png
#
# Tags are written into the filename because lumen feeds bare filenames to
# rofi: whatever is in the name is what rofi can search.
#
# Two independent stages:
#
#   colours   pure ImageMagick + awk, no dependencies. One `magick` call per
#             image renders a small pixel grid and a single awk pass turns it
#             into colour, mood and scene tags. Runs one worker per core.
#   objects   optional (--vision). CLIP scores each image against a vocabulary
#             of prompts entirely offline on the CPU, naming what is actually
#             in the picture: cat, castle, city, robot... One model load covers
#             the whole batch, so this is a separate stage rather than part of
#             the per file worker.
#
# Usage: wallpaper_recognition.sh [options] DIR...

set -uo pipefail

readonly PROG=${0##*/}

# ---------------------------------------------------------------------------
# defaults
# ---------------------------------------------------------------------------

JOBS=$(nproc 2>/dev/null || echo 4)
GRID=64            # sampling grid; 64x64 = 4096 px, enough for region + edge stats
MAX_TAGS=0         # 0 = keep every tag found
RETAG=0
DRY_RUN=0
QUIET=0
VISION=0
VISION_SETUP=0
VISION_REFRESH=0
VISION_DIR=${LUMEN_VISION_DIR:-$HOME/.local/share/lumen/vision}
VISION_PY=$VISION_DIR/venv/bin/python
VISION_SCRIPT=${0%/*}/wallpaper_vision.py
CLIP_REPO=https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main

usage() {
  cat <<EOF
$PROG — tag wallpapers with English #hashtags for rofi search.

Usage: $PROG [options] DIR...

Options:
  -j, --jobs N        parallel workers (default: $JOBS, one per core)
  -r, --retag         strip existing #tags and recompute (default: skip tagged)
  -n, --dry-run       show what would be renamed, change nothing
  -m, --max-tags N    keep at most N tags per file (default: all)
  -g, --grid N        sampling grid size (default: $GRID, min 24)
  -V, --vision        also name the objects in the image (CLIP, offline)
      --vision-setup  install the object recogniser (~600 MB, one time)
      --vision-refresh  reload the vocabulary after editing the label file
  -q, --quiet         only print errors and the summary
  -h, --help          this help

Tag vocabulary:
  colours   black white gray red orange yellow green teal cyan blue purple
            pink brown
  mood      dark light vibrant pastel muted monochrome warm cool colorful
  scene     night space sunset sky nature forest gradient minimal pattern

  Those come from colour statistics, so only scenes separable that way are
  listed. --vision adds what is actually in the picture — anime, character,
  cat, dragon, city, castle, robot, forest, rain... from a vocabulary you can
  edit in wallpaper_labels.txt.

Examples:
  $PROG ~/Pictures/Wallpapers                 # tag new wallpapers
  $PROG --retag -n ~/Pictures/Wallpapers      # preview a full re-tag
  $PROG --vision --retag ~/Pictures/Wallpapers  # colours + objects
  $PROG --retag ~/Pictures/Wallpapers/dark    # rewrite tags in one folder
EOF
}

# ---------------------------------------------------------------------------
# arguments
# ---------------------------------------------------------------------------

DIRS=()
while [ $# -gt 0 ]; do
  case $1 in
    -j|--jobs)         JOBS=${2:-}; shift 2 ;;
    -r|--retag)        RETAG=1; shift ;;
    -n|--dry-run)      DRY_RUN=1; shift ;;
    -m|--max-tags)     MAX_TAGS=${2:-}; shift 2 ;;
    -g|--grid)         GRID=${2:-}; shift 2 ;;
    -V|--vision)       VISION=1; shift ;;
    --no-vision)       VISION=0; shift ;;
    --vision-setup)    VISION_SETUP=1; shift ;;
    --vision-refresh)  VISION_REFRESH=1; shift ;;
    -q|--quiet)        QUIET=1; shift ;;
    -h|--help)         usage; exit 0 ;;
    --)                shift; DIRS+=("$@"); break ;;
    -*)                echo "$PROG: unknown option '$1'" >&2; usage >&2; exit 2 ;;
    *)                 DIRS+=("$1"); shift ;;
  esac
done

[ ${#DIRS[@]} -eq 0 ] && { usage >&2; exit 2; }

case $JOBS in ''|*[!0-9]*) echo "$PROG: --jobs must be a number" >&2; exit 2 ;; esac
case $GRID in ''|*[!0-9]*) echo "$PROG: --grid must be a number" >&2; exit 2 ;; esac
case $MAX_TAGS in ''|*[!0-9]*) echo "$PROG: --max-tags must be a number" >&2; exit 2 ;; esac
[ "$JOBS" -lt 1 ] && JOBS=1
[ "$GRID" -lt 24 ] && GRID=24

command -v magick >/dev/null 2>&1 || {
  echo "$PROG: ImageMagick required — sudo pacman -S imagemagick" >&2; exit 1; }
command -v awk >/dev/null 2>&1 || {
  echo "$PROG: awk required" >&2; exit 1; }

for d in "${DIRS[@]}"; do
  [ -d "$d" ] || { echo "$PROG: not a directory: $d" >&2; exit 1; }
done

# ---------------------------------------------------------------------------
# optional vision backend
# ---------------------------------------------------------------------------
# Dormant unless --vision is passed AND ollama is installed with a vision model.
# Everything above works with zero extra dependencies.

if [ "$VISION" = 1 ]; then
  if ! command -v ollama >/dev/null 2>&1; then
    echo "$PROG: --vision needs ollama (not installed) — continuing without it" >&2
    VISION=0
  else
    if [ -z "$VISION_MODEL" ]; then
      VISION_MODEL=$(ollama list 2>/dev/null \
        | awk 'NR>1{print $1}' \
        | grep -m1 -iE 'llava|vision|moondream|qwen.*vl|gemma3|minicpm-v') || true
    fi
    if [ -z "$VISION_MODEL" ]; then
      echo "$PROG: no vision model found in 'ollama list' — continuing without it" >&2
      echo "$PROG: try 'ollama pull llava' or pass --vision-model NAME" >&2
      VISION=0
    fi
  fi
fi

# ---------------------------------------------------------------------------
# image analyser (awk) — written once, shared by every worker
# ---------------------------------------------------------------------------

TMPDIR_RUN=$(mktemp -d "${TMPDIR:-/tmp}/wallrecog.XXXXXX") || exit 1
trap 'rm -rf "$TMPDIR_RUN"' EXIT INT TERM
AWK_PROG=$TMPDIR_RUN/analyse.awk

cat >"$AWK_PROG" <<'AWKEOF'
# Reads a `magick ... txt:-` pixel dump, prints a space separated tag list.
# Expects -v G=<grid size>. Set -v DEBUG=1 to dump the raw metrics first.

function mx3(a,b,c){ return a>b ? (a>c?a:c) : (b>c?b:c) }
function mn3(a,b,c){ return a<b ? (a<c?a:c) : (b<c?b:c) }
function absf(x){ return x<0 ? -x : x }

function hue_of(r,g,b,   M,m,c,h){
  M=mx3(r,g,b); m=mn3(r,g,b); c=M-m
  if (c<=0) return -1
  if      (M==r) h=60*((g-b)/c)
  else if (M==g) h=60*(2+(b-r)/c)
  else           h=60*(4+(r-g)/c)
  if (h<0) h+=360
  return h
}

# Map a hue onto the short, searchable colour vocabulary. Deliberately small:
# these are the words someone actually types into rofi. Brown and pink are not
# hues of their own, they are dark orange/red and pale red, hence the value and
# saturation tests.
function color_name(h,s,v){
  if (h<0) return ""
  if (h>=345 || h<14){
    if (v<=0.45 && s>=0.25) return "brown"
    if (v>=0.70 && s<=0.65) return "pink"
    return "red"
  }
  if (h<45){  if (v<=0.50) return "brown"; return "orange" }
  if (h<68){  if (v<=0.45) return "brown"; return "yellow" }
  if (h<160)  return "green"
  if (h<188)  return "teal"
  if (h<205)  return "cyan"
  if (h<252)  return "blue"
  if (h<292)  return "purple"
  if (h<330){ if (v>=0.62) return "pink"; return "purple" }
  return "pink"
}

function emit(t){ if (t!="" && !(t in seen)){ seen[t]=1; out[++nout]=t } }

# Sum one hue cluster (peak +/- 25 deg). Sets CS/CV to the representative
# saturation and value, taken from the strongly chromatic pixels when there are
# enough of them, so a dark street lit by vivid signs is not called brown.
function cluster(p,   k,j,mass,sv,vv,ws,sv2,vv2,ws2){
  CS=0; CV=0
  if (p<0) return 0
  for (k=-5; k<=5; k++){
    j=(p+k+NB)%NB
    mass+=HB[j]; sv+=HBs[j]; vv+=HBv[j]; ws+=HB[j]
    sv2+=HBs2[j]; vv2+=HBv2[j]; ws2+=HBw2[j]
  }
  if (ws2>0 && ws>0 && ws2>=0.15*ws){ CS=sv2/ws2; CV=vv2/ws2 }
  else if (ws>0){ CS=sv/ws; CV=vv/ws }
  return mass
}

BEGIN{ NB=72; BW=360/NB }          # 72 hue bins of 5 degrees

/^[0-9]/{
  split($1,P,/[,:]/); x=P[1]+0; y=P[2]+0
  t=$2; gsub(/[()]/,"",t); split(t,C,",")
  r=C[1]/255; g=C[2]/255; b=C[3]/255

  M=mx3(r,g,b); m=mn3(r,g,b)
  ch=M-m                                # chroma: how much colour, absolutely
  v=M; s=(M>0)?ch/M:0                   # HSV saturation: only used for naming
  L=0.2126*r+0.7152*g+0.0722*b
  h=hue_of(r,g,b)

  n++; sumL+=L; sumL2+=L*L; sumC+=ch
  LM[x SUBSEP y]=L
  CK[x SUBSEP y]=C[1]*65536+C[2]*256+C[3]   # exact colour, for flat area test

  # Chroma, not HSV saturation, decides whether a pixel reads as coloured: a
  # near black pixel like (0,17,30) has saturation 1.0 but no visible colour,
  # which is what used to make every dark wallpaper come out "blue".
  if (ch>=0.10 && v>=0.10) cn++
  if (ch<0.07)             an++
  if (L>0.75) brightN++
  if (ch>=0.42 && v>=0.60) vividN++

  Q[int(r*15.999)*256 + int(g*15.999)*16 + int(b*15.999)]=1

  if (h>=0 && ch>=0.06 && v>=0.08){
    w=ch*(0.35+0.65*v)
    bi=int(h/BW); if (bi>=NB) bi=NB-1
    HB[bi]+=w; HBv[bi]+=w*v; HBs[bi]+=w*s
    if (ch>=0.25){ HBv2[bi]+=w*v; HBw2[bi]+=w; HBs2[bi]+=w*s }
    hueMass+=w
    if (h<90 || h>=300) warmW+=w; else coolW+=w
    z=int(y*3/G); if (z>2) z=2
    rHB[z,bi]+=w; rMass[z]+=w
  }

  z=int(y*3/G); if (z>2) z=2
  rn[z]++; rL[z]+=L
  rowL[y]+=L
}

END{
  if (n<16){ print "" ; exit }

  meanL=sumL/n; meanC=sumC/n
  varL=sumL2/n-meanL*meanL; sdL=(varL>0)?sqrt(varL):0
  chromaFrac=cn/n; achromFrac=an/n
  brightFrac=brightN/n; vividFrac=vividN/n
  for (q in Q) distinct++

  # ---- detail, per third, plus flat area fraction -----------------------
  # Neighbouring pixels that are byte for byte identical mean a poster-like
  # flat fill; photographs and painted art almost never have them.
  eN=0; eSum=0; eStrong=0; flatN=0
  for (yy=0; yy<G-1; yy++) for (xx=0; xx<G-1; xx++){
    a=LM[xx SUBSEP yy]
    d=absf(a-LM[(xx+1) SUBSEP yy])+absf(a-LM[xx SUBSEP (yy+1)])
    eSum+=d; eN++; if (d>0.15) eStrong++
    if (CK[xx SUBSEP yy]==CK[(xx+1) SUBSEP yy]) flatN++
    z=int(yy*3/G); if (z>2) z=2
    dS[z]+=d; dN[z]++
  }
  detail=(eN>0)?eSum/eN:0
  edgeFrac=(eN>0)?eStrong/eN:0
  flatFrac=(eN>0)?flatN/eN:0
  for (z=0; z<3; z++) zD[z]=(dN[z]>0)?dS[z]/dN[z]:0

  # ---- vertical gradient: how well row luma fits a straight ramp --------
  sx=0; sy=0; sxx=0; sxy=0
  for (yy=0; yy<G; yy++){ rm=rowL[yy]/G; sx+=yy; sy+=rm; sxx+=yy*yy; sxy+=yy*rm }
  den=G*sxx-sx*sx
  slope=(den!=0)?(G*sxy-sx*sy)/den:0
  icept=(sy-slope*sx)/G
  res=0
  for (yy=0; yy<G; yy++){ rm=rowL[yy]/G; d=rm-(slope*yy+icept); res+=d*d }
  gradRes=sqrt(res/G)
  ramp=absf(slope)*G

  for (z=0; z<3; z++) if (rn[z]>0) zL[z]=rL[z]/rn[z]

  # ---- dominant hue clusters -------------------------------------------
  for (i=0; i<NB; i++){
    sm=0
    for (k=-3; k<=3; k++) sm+=HB[(i+k+NB)%NB]
    SM[i]=sm
  }
  p1=-1; b1=0
  for (i=0; i<NB; i++) if (SM[i]>b1){ b1=SM[i]; p1=i }
  m1=cluster(p1); s1=CS; v1=CV
  p2=-1; b2=0
  for (i=0; i<NB; i++){
    dd=absf(i-p1); if (dd>NB/2) dd=NB-dd
    if (dd>=7 && SM[i]>b2){ b2=SM[i]; p2=i }
  }
  m2=cluster(p2); s2=CS; v2=CV

  h1=(p1>=0)?p1*BW+BW/2:-1
  h2=(p2>=0)?p2*BW+BW/2:-1
  # A faint colour still gets named when it is clearly *the* colour of the
  # image, which is how washed out greens or foggy blues keep their tag.
  c1=(m1/n>=0.030 || (m1/n>=0.012 && hueMass>0 && m1>=0.45*hueMass)) \
       ? color_name(h1,s1,v1) : ""
  c2=(m2/n>=0.015 && m1>0 && m2>=0.25*m1) ? color_name(h2,s2,v2) : ""
  if (c2==c1) c2=""

  nclust=0
  for (i=0; i<NB; i++){
    lead=1
    for (k=-4; k<=4; k++){ j=(i+k+NB)%NB; if (SM[j]>SM[i]) lead=0 }
    if (lead && SM[i]>0 && hueMass>0 && SM[i]/hueMass>=0.10) nclust++
  }

  for (z=0; z<3; z++){
    bz=-1; bw=0
    for (i=0; i<NB; i++) if (rHB[z,i]>bw){ bw=rHB[z,i]; bz=i }
    zH[z]=(bz>=0)?bz*BW+BW/2:-1
    zM[z]=(rMass[z]>0 && rn[z]>0)?rMass[z]/rn[z]:0
  }
  topBlue = (zH[0]>=195 && zH[0]<=255 && zM[0]>=0.04)
  topWarm = (((zH[0]>=0 && zH[0]<50) || zH[0]>=325) && zM[0]>=0.04)
  midWarm = (((zH[1]>=0 && zH[1]<50) || zH[1]>=325) && zM[1]>=0.04)
  greenMass=0
  for (i=int(68/BW); i<=int(159/BW); i++) greenMass+=HB[i]
  greenFrac=greenMass/n

  if (DEBUG)
    printf "meanL=%.3f sdL=%.3f meanC=%.3f chromF=%.3f achromF=%.3f brightF=%.3f vividF=%.3f distinct=%d detail=%.3f edgeF=%.3f flatF=%.3f zD=%.3f/%.3f/%.3f gradRes=%.3f ramp=%.3f nclust=%d m1n=%.3f m2n=%.3f h1=%.0f c1=%s h2=%.0f c2=%s zL=%.2f/%.2f/%.2f zH=%.0f/%.0f/%.0f greenF=%.3f\n", \
      meanL,sdL,meanC,chromaFrac,achromFrac,brightFrac,vividFrac,distinct,detail,edgeFrac,flatFrac,zD[0],zD[1],zD[2],gradRes,ramp,nclust,m1/n,m2/n,h1,(c1==""?"-":c1),h2,(c2==""?"-":c2),zL[0],zL[1],zL[2],zH[0],zH[1],zH[2],greenFrac

  # =======================================================================
  # colours first — they are what people search most
  # =======================================================================
  emit(c1)
  emit(c2)
  if (achromFrac>=0.55 || c1==""){
    if      (meanL<0.16) emit("black")
    else if (meanL>0.82) emit("white")
    else                 emit("gray")
  }

  # ---- mood -------------------------------------------------------------
  mono   = (achromFrac>=0.85)
  pastel = (chromaFrac>=0.25 && meanC<0.25 && meanL>0.62)

  if (meanL<0.26) emit("dark")
  if (meanL>0.70) emit("light")
  if (mono) emit("monochrome")
  if (!mono && meanC>=0.32 && chromaFrac>=0.55) emit("vibrant")
  if (pastel) emit("pastel")
  if (!mono && !pastel && meanC<0.12) emit("muted")
  if (nclust>=3 && chromaFrac>=0.30 && meanC>=0.18 && distinct>=200) emit("colorful")
  if (!mono && hueMass>0){
    if      (warmW/hueMass>=0.72) emit("warm")
    else if (coolW/hueMass>=0.72) emit("cool")
  }

  # =======================================================================
  # content — only the rules that survived checking against hand labelled
  # wallpapers. Scenes that colour statistics cannot separate were dropped
  # rather than guessed at: a wrong tag is worse than a missing one, because
  # it puts the wrong wallpaper in front of you in rofi. Notably city, water,
  # mountain, desert and snow all collapsed into their backgrounds, and skin
  # tone matched amber street lights while missing small faces, so people and
  # anime went too. Use --vision for those.
  # =======================================================================

  night = (meanL<0.24 && zL[0]<0.30)
  space = (meanL<0.18 && brightFrac>=0.002 && brightFrac<=0.05 && detail<0.04 \
           && greenFrac<0.02)
  # a real sunset sky is smooth; a warm lit street is not
  sunset= ((topWarm || midWarm) && ramp>=0.12 && zD[0]<0.09 && meanC>=0.15 \
           && zL[0]>zL[2])
  sky   = (topBlue && zL[0]>=0.35 && zD[0]<0.06 && zL[0]>zL[2])
  # green has to be the main colour, or cover real ground: neon green signs on a
  # dark street were otherwise coming out as forest
  nature= ((c1=="green" || greenFrac>=0.10) && detail>=0.03)
  forest= (nature && meanL<0.45)
  grad  = (gradRes<0.02 && detail<0.02 && ramp>=0.10)
  minim = (flatFrac>=0.35 && detail<0.06 && !grad)
  patt  = (detail>=0.10 && flatFrac<0.20 && absf(zL[0]-zL[2])<0.08 \
           && absf(zL[0]-zL[1])<0.08)

  if (night)  emit("night")
  if (space)  emit("space")
  if (sunset) emit("sunset")
  if (sky)    emit("sky")
  if (forest) emit("forest"); else if (nature) emit("nature")
  if (grad)   emit("gradient")
  if (minim)  emit("minimal")
  if (patt)   emit("pattern")

  line=""
  for (i=1; i<=nout; i++) line=line (i>1?" ":"") out[i]
  print line
}
AWKEOF

# ---------------------------------------------------------------------------
# colour worker
# ---------------------------------------------------------------------------

# Strip every existing #tag (and its separator) from a basename, then any
# separator left dangling at the end. Kept in one sed so the workers do not
# depend on shell options the parent enabled.
strip_tags() {
  printf '%s' "$1" | sed -E 's/[-_ ]*#[A-Za-z0-9]+//g; s/[-_ .]+$//'
}

# Print "path<TAB>tags" for one image. Renaming happens later, once the object
# tags are in too, so that a file is touched exactly once.
analyse_one() {
  local file=$1 ext=${file##*.}

  # ImageMagick reads `path[0]` as "first frame", which is what we want for
  # animated GIFs — but a literal '[' in the path would be parsed as a frame
  # spec too, so those go through a bracket-free symlink instead.
  local src=$file cleanup=""
  case $file in
    *'['*|*']'*)
      cleanup=$TMPDIR_RUN/link.$$.$ext
      ln -sf "$file" "$cleanup" 2>/dev/null && src=$cleanup || cleanup=""
      ;;
  esac

  local tags
  tags=$(magick -limit thread 1 -define jpeg:size="$((GRID*2))x$((GRID*2))" \
            "$src[0]" -auto-orient -background black -alpha remove -alpha off \
            -thumbnail "${GRID}x${GRID}!" -colorspace sRGB -type TrueColor \
            -depth 8 txt:- 2>/dev/null \
         | awk -v G="$GRID" -f "$AWK_PROG" 2>/dev/null)
  [ -n "$cleanup" ] && rm -f "$cleanup"

  [ -z "$tags" ] && { printf 'ERR\t%s\n' "$file" >&2; return 0; }
  printf '%s\t%s\n' "$file" "$tags"
}

export -f strip_tags analyse_one
export AWK_PROG GRID TMPDIR_RUN

# ---------------------------------------------------------------------------
# stage 1 — collect
# ---------------------------------------------------------------------------

LIST=$TMPDIR_RUN/list
COLOUR_TSV=$TMPDIR_RUN/colour.tsv
VISION_TSV=$TMPDIR_RUN/vision.tsv
: >"$VISION_TSV"

find "${DIRS[@]}" -type f \( \
    -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \
    -o -iname '*.gif' -o -iname '*.webp' -o -iname '*.bmp' -o -iname '*.avif' \
  \) -print0 >"$LIST"

# Tags travel as tab separated lines from here on, so a path containing a tab
# or a newline would corrupt the join. Those are dropped loudly rather than
# silently mangled.
weird=0
while IFS= read -r -d '' f; do
  case $f in
    *"$(printf '\t')"*) printf '  skipped (tab in name): %s\n' "$f" >&2; weird=$((weird+1)) ;;
    *) printf '%s\0' "$f" ;;
  esac
done <"$LIST" >"$LIST.clean"
mv "$LIST.clean" "$LIST"

total=$(tr -dc '\0' <"$LIST" | wc -c)
[ "$total" -eq 0 ] && { echo "$PROG: no images found" >&2; exit 0; }

# Skipping already tagged files before the expensive stages, not after.
if [ "$RETAG" != 1 ]; then
  skipped=0
  while IFS= read -r -d '' f; do
    case ${f##*/} in
      *'#'*) skipped=$((skipped+1)) ;;
      *) printf '%s\0' "$f" ;;
    esac
  done <"$LIST" >"$LIST.new"
  mv "$LIST.new" "$LIST"
  total=$(tr -dc '\0' <"$LIST" | wc -c)
else
  skipped=0
fi

if [ "$total" -eq 0 ]; then
  printf '\nnothing to do: %d file(s) already tagged (use --retag to redo them)\n' "$skipped"
  exit 0
fi

# ---------------------------------------------------------------------------
# stage 2 — colour, mood and scene, one worker per core
# ---------------------------------------------------------------------------

[ "$QUIET" = 1 ] || printf 'analysing %d image(s)...\n' "$total"
xargs -0 -r -P "$JOBS" -n 1 bash -c 'analyse_one "$1"' _ <"$LIST" \
  2>"$TMPDIR_RUN/err" >"$COLOUR_TSV"
err_count=$(grep -c '^ERR' "$TMPDIR_RUN/err" 2>/dev/null || echo 0)

# ---------------------------------------------------------------------------
# stage 3 — objects, one model load for the whole batch
# ---------------------------------------------------------------------------

if [ "$VISION" = 1 ]; then
  [ "$QUIET" = 1 ] || printf 'recognising objects (CLIP)...\n'
  if ! "$VISION_PY" "$VISION_SCRIPT" <"$LIST" >"$VISION_TSV" 2>"$TMPDIR_RUN/verr"; then
    sed 's/^/  /' "$TMPDIR_RUN/verr" >&2
    echo "$PROG: object recognition failed — continuing with colour tags only" >&2
    : >"$VISION_TSV"
  fi
fi

# ---------------------------------------------------------------------------
# stage 4 — merge and rename
# ---------------------------------------------------------------------------

ok=0; same=0; dry=0; fail=0

merge_and_rename() {
  awk -F'\t' '
    NR==FNR { if ($2 != "") v[$1]=$2; next }
    { print $1 "\t" $2 ($1 in v ? " " v[$1] : "") }
  ' "$VISION_TSV" "$COLOUR_TSV"
}

while IFS=$'\t' read -r file tags; do
  [ -n "$file" ] || continue
  dir=${file%/*}; name=${file##*/}
  ext=${name##*.}; base=${name%.*}
  [ "$RETAG" = 1 ] && base=$(strip_tags "$base")
  [ -z "$base" ] && base="wallpaper"

  suffix=""; count=0
  for t in $tags; do
    [ "$MAX_TAGS" -gt 0 ] && [ "$count" -ge "$MAX_TAGS" ] && break
    try="$suffix-#$t"
    [ $((${#base} + ${#try} + ${#ext} + 1)) -gt 250 ] && break
    suffix=$try
    count=$((count + 1))
  done
  [ -z "$suffix" ] && { fail=$((fail+1)); continue; }

  new="$dir/$base$suffix.$ext"
  if [ "$new" = "$file" ]; then
    same=$((same+1))
    continue
  fi

  n=2
  while [ -e "$new" ]; do
    new="$dir/$base$suffix-$n.$ext"
    n=$((n + 1))
    [ "$n" -gt 99 ] && break
  done

  if [ "$DRY_RUN" = 1 ]; then
    dry=$((dry+1))
    [ "$QUIET" = 1 ] || printf '  %s\n     -> %s\n' "$name" "${new##*/}"
  elif mv -n -- "$file" "$new" 2>/dev/null; then
    ok=$((ok+1))
    [ "$QUIET" = 1 ] || printf '  %s\n     -> %s\n' "$name" "${new##*/}"
  else
    fail=$((fail+1))
  fi
done < <(merge_and_rename)

printf '\n'
[ "$dry"     -gt 0 ] && printf 'dry-run:   %d file(s) would be renamed\n' "$dry"
[ "$ok"      -gt 0 ] && printf 'tagged:    %d\n' "$ok"
[ "$skipped" -gt 0 ] && printf 'skipped:   %d (already tagged, use --retag)\n' "$skipped"
[ "$same"    -gt 0 ] && printf 'unchanged: %d\n' "$same"
[ "$err_count" -gt 0 ] && printf 'unreadable: %d\n' "$err_count"
[ "$fail"    -gt 0 ] && printf 'failed:    %d\n' "$fail"
exit 0
