# Web UI - Tabs Explained

This section explains each tab in plain English.

## Upload
Use this to create new runs.
- Drop ZIP files.
- Choose options (Autotag, AutoChar, Train, etc.).
- Click Launch.

Common options:
- Autotag: writes tags to images automatically.
- AutoChar: removes unwanted tags using presets.
- TagVerify: extra check for color tags (slower, cleaner).
- Facecap: adds extra face crops for better faces.
- Train: trains a LoRA after tagging.
- GPU: speeds up autotagging and required for training.
- Images Only: skip video capping if your ZIP already has images.
- Manual Tagging: pause so you can edit tags before training.

## Queue
Live progress for active runs.
- Shows current step (rename, select, autotag, training).
- Only one run moves through the pipeline at a time.
- If a run is stuck, check the system status.

## History
Finished and failed runs.
- Download dataset ZIP and LoRA ZIP.
- View errors and sample images (if training ran).
- Delete a run to remove its data.

## AutoChar
Manage tag cleanup rules.
- Create presets with block patterns.
- Apply presets during Upload.
- Changes apply immediately to new runs.

## Settings
Global defaults for future runs.
- Capping FPS, selection size, tag thresholds.
- Training defaults (resolution, batch size, ranks).
- Notification settings (email/discord/slack/webhook if configured).
- Tagger models list and download (if available in your UI build).

## Manual Tagging
Edit tags after Autotag.
- Open a paused run.
- Edit captions per image.
- Remove unwanted tags in bulk.
- Click Commit to resume the run.

## Docs
This guide.

## System Status
Shows background services health.
- OK: service is running.
- Waiting/Busy: service is active.
- Fail: service is down or stale.

## Tagger Models (if shown in your UI)
This tab manages the Autotag model.
- Download a model by repo ID (for example `SmilingWolf/wd-swinv2-tagger-v3`).
- After download, set the model as your default in Settings.
- If a model shows as missing, it is not fully downloaded yet.
