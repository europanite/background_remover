# Video Background Remover

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
