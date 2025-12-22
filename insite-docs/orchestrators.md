# Orchestrators

FrameForge runs three background services (orchestrators). Together they move each run through the pipeline from upload to download.

## Initiator
- Watches the queue for new uploads.
- Unzips the ZIP into the input folder.
- Creates the run plan so progress can be tracked in the UI.

## Orchestrator
- Runs the main workflow (rename, capping, selection, crop/flip, autotag, training).
- Updates progress steps shown in Queue and History.
- Pauses the run if Manual Tagging is enabled, then resumes after you commit.

## Finisher
- Packages datasets and LoRAs into downloadable ZIPs.
- Collects training outputs and sample images.
- Cleans staging folders after completion.

## What users should know
- You do not need to start these manually.
- If a run looks stuck, check Queue status first, then ask an admin to check the services.
- Only one run moves through the workflow at a time to keep results stable.
