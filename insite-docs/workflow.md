# Workflow steps

- Rename: normalizes file names in the upload.
- Capping: extracts frames from videos (optional face crops).
- Select: picks a diverse set of images per character.
- Crop/Flip: creates useful variations without exploding size.
- Autotag: writes tags to each image.
- AutoChar: removes unwanted tags using presets.
- Finalize: moves the dataset to final output.
- Training (optional): trains a LoRA and saves samples.
- Packaging: zips downloads for History.

Queue and History show these steps as short hints.
