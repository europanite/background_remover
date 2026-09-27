# OpenCV Video Background Remover — GPU / Human Matting

## RVM colour / VLC transparency note

RVM is used for the **alpha matte**. By default the visible RGB is copied from the original
video (`--rvm-color-source original`) to avoid colour shifts in RVM's predicted foreground.
Fully transparent pixels are also cleared to black so alpha-unaware previewers do not reveal
hidden garbage/background RGB.

For a quick visual check that does not depend on player alpha support, render an opaque preview:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm service \
  /data/input.mp4 /data/preview.mp4 \
  --method rvm --accelerator cuda --background green --downsample-ratio 0.4
```

Then create the transparent MOV:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm service \
  /data/input.mp4 /data/output.mov \
  --method rvm --accelerator cuda --background transparent --downsample-ratio 0.4
```


GPU-accelerated video background removal in a Docker Compose container.

The default method is now **RVM (Robust Video Matting)** rather than MOG2. OpenCV is used for video decoding, color conversion and compositing; PyTorch/CUDA runs the human-matting model.

This matters for difficult footage such as **a person wearing white clothes in front of a white wall**. MOG2 and chroma-keying depend heavily on color/motion separation, while RVM predicts a human foreground and soft alpha matte.

## Important fix: source display aspect ratio is restored

OpenCV's decoded `frame.shape` is **not sufficient** to reconstruct every video's intended display shape. A source can store non-square pixel metadata (`SAR`), a separate display aspect ratio (`DAR`), and phone/camera rotation metadata. Earlier versions forced `SAR=1` from the OpenCV raster size, which could make the output look square or otherwise stretched.

The program now uses `ffprobe` to read the source **encoded size, SAR, DAR and rotation**, asks OpenCV to honor display rotation, and then physically normalizes frames to the source display aspect using square output pixels. RVM therefore sees the correctly proportioned person, and the output no longer depends on a player honoring unusual pixel-aspect metadata.

At startup you should see geometry diagnostics similar to:

```text
source geometry: encoded=180x144 sar=16:15 display_dar=4:3 rotation=0
decoded frame: 180x144 @ 30.000 fps
output frame: 192x144 (square pixels)
```

If a damaged source has incorrect aspect metadata, bypass this restoration with `--geometry-mode decoded`.

## Requirements

- Docker + Docker Compose
- NVIDIA GPU
- NVIDIA Container Toolkit
- Recent NVIDIA driver compatible with CUDA 12.6

## Build

```bash
export HOST_UID=$(id -u)
export HOST_GID=$(id -g)

docker compose \
  -f docker-compose.yml \
  -f docker-compose.gpu.yml \
  build
```

The GPU image uses `pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime` and downloads the official RVM MobileNetV3 checkpoint during build.

## Check GPU

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.gpu.yml \
  run --rm service --info
```

For RVM, `PyTorch CUDA devices` must be at least `1`.

## Recommended command

Put the input at `service/data/input.mp4` and run:

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.gpu.yml \
  run --rm service \
  /data/input.mp4 /data/output.mov \
  --method rvm \
  --accelerator cuda \
  --background transparent
```

`output.mov` uses qtrle alpha and is much faster to encode than transparent VP9/WebM.

Transparent WebM is also supported:

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.gpu.yml \
  run --rm service \
  /data/input.mp4 /data/output.webm \
  --method rvm \
  --accelerator cuda \
  --background transparent
```

## Full-body footage

RVM internally downsamples the semantic branch. Automatic mode targets a maximum internal side of about 512 px. If fine body boundaries are lost, increase the ratio:

```bash
--downsample-ratio 0.4
```

For 1920x1080 the RVM documentation suggests roughly `0.25` for portrait shots and `0.4` for full-body shots.

## Opaque background + NVENC

If transparency is not required, encoding can also use NVIDIA NVENC:

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.gpu.yml \
  run --rm service \
  /data/input.mp4 /data/output.mp4 \
  --method rvm \
  --accelerator cuda \
  --encoder nvenc \
  --background white
```

## Legacy methods

MOG2 and chroma key remain available:

```text
--method mog2
--method chroma
```

For same-color foreground/background footage, use `--method rvm`.

## License note

This repository's own code can retain its existing license, but the GPU image downloads **Robust Video Matting**, whose current upstream repository is GPL-3.0. If you plan to redistribute a built image or use it in a licensing-sensitive product, review the upstream license obligations first.

## Anamorphic / wrong aspect ratio sources

Some camera formats store a non-square raster such as `1280x1080` but are intended
to be displayed as `16:9`. The application now probes geometry with the broadly
compatible `ffprobe -show_streams` path. If the source metadata is missing or an
older ffprobe still cannot expose it, override the intended display ratio explicitly:

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.gpu.yml \
  run --rm service \
  /data/input.mp4 /data/preview.mp4 \
  --method rvm \
  --accelerator cuda \
  --background green \
  --downsample-ratio 0.4 \
  --display-aspect 16:9
```

For a decoded `1280x1080` raster, `--display-aspect 16:9` produces a square-pixel
`1920x1080` output instead of preserving the distorted `1280x1080` display shape.
Use the actual intended ratio of the source if it is not 16:9.
