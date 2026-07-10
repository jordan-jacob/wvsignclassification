# wvsignclassification

WV Roadway Sign Detection — computer vision for detecting roadway signs on West
Virginia dashcam footage, with an inventory-reconciliation map + review app.

## Local inference app (`sign_mapper/`)

```
cd sign_mapper
pip install -r requirements.txt   # app-only deps, smaller than the training env
python app.py                     # opens http://localhost:5000
```

**One-time setup size:** the app itself is small, but `ultralytics` pulls in
`torch` (~2 GB with CUDA, ~500 MB CPU-only). This is expected and unavoidable for
ML inference — a one-time download, not the app's own size. The Flask app uses
`opencv-python-headless` (no GUI libraries) to keep the rest of the footprint
minimal. Generated thumbnails under `sign_mapper/outputs/frames/` are ephemeral
and are not committed (see `.gitignore`).
