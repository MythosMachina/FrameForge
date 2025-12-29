# Workflow Steps (what happens to your run)

FrameForge runs the same pipeline every time. Here is the order and what it means.

1) Rename
- Normalizes file names so every image is easy to track.

2) Cap (videos only)
- Extracts frames from videos.
- Optional Facecap creates extra face crops.

3) Select
- Picks a diverse set of images per character.

4) Crop and Flip
- Creates useful variations (center, top, bottom, mirrored).
- Helps training without exploding size.

5) Autotag
- Writes tags (captions) for each image.

6) AutoChar (if enabled)
- Removes unwanted tags using presets.

7) Manual Tagging (if enabled)
- Pipeline pauses.
- You edit tags in the Manual Tagging tab.
- Commit to resume.

8) Finalize
- Moves the cleaned dataset to final output.

9) Training (optional)
- Trains a LoRA using your dataset.
- Writes sample images during training.

10) Packaging
- Creates ZIP downloads for History.

## Images Only mode
If you turn on Images Only:
- Step 2 (Cap) is skipped.
- The rest of the pipeline is the same.
