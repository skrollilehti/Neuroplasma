# Neuroverkkoplasma (Skrolli 2021.1)

## Riippuvuudet

Python3 ja kirjastot tensorflow, numpy ja PIL. Komentorivityökalu ffmpeg.

## Ruutujen luominen

```
mkdir frames
python3 create_frames.py 
```

## Videon luominen ruuduista

```
ffmpeg -framerate 60 -f image2 -i frames/%d.png -pix_fmt yuv420p -c:v libx264 video.mp4
```
