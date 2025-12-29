# Quickstart (first run)

This is the fastest way to get your first dataset.

## 1) Upload
- Open the WebApp.
- Go to Upload.
- Drag and drop one or more ZIP files.

## 2) Choose options
Recommended for a first run:
- Autotag: ON
- AutoChar: ON
- Train: OFF
- GPU: ON (if available)
- Images Only: OFF (unless your ZIP is only images)

## 3) Launch
- Click Launch.
- A Run ID is created. You will see it in Queue.

## 4) Wait for completion
- Queue shows current step (rename, autotag, etc.).
- When done, the run moves to History.

## 5) Download
- Open History.
- Download the dataset ZIP.

## Optional: pick a better tagger model
If you want stronger tags, download a better Autotag model.
- Open the Tagger Models tab (if available).
- Download `SmilingWolf/wd-swinv2-tagger-v3` (good all-rounder).
- Go to Settings and set it as the default tagger model.

## Optional: enable notifications
If your admin configured notifications, you can turn them on in Settings.
- Enable the channels you want (email/discord/slack/webhook).
- Turn on job and queue notifications.

---

# Quickstart (with training)
Use this if you want a LoRA.

1. Upload ZIP(s).
2. Enable Train and GPU.
3. Pick a Train Profile if you see one.
4. Launch.
5. In History, download:
   - dataset ZIP
   - LoRA ZIP
   - sample images (if available)

---

# Manual Tagging flow
Use this if you want to edit tags before training.

1. Upload ZIP(s).
2. Turn on Manual Tagging.
3. Launch.
4. The run pauses after autotagging.
5. Open Manual Tagging tab.
6. Edit tags, then Commit.
7. The run resumes.
