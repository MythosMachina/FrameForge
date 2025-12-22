# Quickstart (creator/reviewer)

1) Open the WebApp.
2) Go to Upload and drop one or more ZIPs.
3) Pick your options (Autotag, AutoChar, Facecap, Train, GPU).
4) Click Launch to create runs.
5) Watch Queue for progress and History for finished downloads.

## Creator path (fast)
- Use Autotag + AutoChar for quick, clean tags.
- Enable Train if you want a LoRA.
- Download from History when done.

## Reviewer path
- Check Queue status and step hints.
- Open History to review downloads and errors.
- Ask for a re-upload if tags or samples look off.

## Manual tagging flow
- Turn on Manual Tagging before Launch.
- The run pauses after autotagging.
- Open Manual Tagging, edit captions, then Commit to resume training.

## What the options mean
- Autotag: adds tags automatically.
- AutoChar: removes unwanted tags using preset rules.
- TagVerify: extra check for color tags (slower, more accurate).
- Facecap: adds more face crops for better faces.
- Train: trains a LoRA after tagging (GPU required).
- GPU: speeds up tagging; required for training.
