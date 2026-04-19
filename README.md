```
░░▒▒▓▓████████████████████████████████▓▓▒▒░░
   _    _                            _
  | | _(_)_ __   ___ _ __ ___   __ _| |
  | |/ / | '_ \ / _ \ '_ ` _ \ / _` | |
  |   <| | | | |  __/ | | | | | (_| |_|
  |_|\_\_|_| |_|\___|_| |_| |_|\__,_(_)
                κίνημα
░░▒▒▓▓████████████████████████████████▓▓▒▒░░
       motion · morph · montage
```

# kinema

Music video generator. Pulls images from the a-u.supply outputs index, audio from releases (or any output index), and renders them through chained FFmpeg transition recipes — each run a different stack of dissolves, masks, glitches, and tweens.

Named after κίνημα — *kinēma*, motion. The thing that turns a stack of stills into something that moves.

## Modes

- **picked** — caller supplies image and audio paths
- **search** — query the a-u.supply outputs index (any index, any filter)
- **random** — pure random fill from the search index
- **release** — audio defaults to a track from the releases catalog

## Recipes

Each render samples transitions from a recipe's pool. See `recipes/*.yaml`. v0 is FFmpeg-only (`xfade`, luma masks, `minterpolate`); v1 may offload neural interpolation to Modal.

## Run

```sh
docker build -t kinema .
docker run --rm \
  -v $PWD/work:/work \
  -e AU_BASE_URL=https://a-u.supply \
  -e AU_API_KEY=$AU_API_KEY \
  kinema --recipe smooth-fade --aspect 16:9 --output /work/out.mp4
```
