# Background workers (simple explanation)

FrameForge uses three workers so your run can move through the pipeline smoothly.

## Initiator
- Takes a new upload from the Queue.
- Unzips it and prepares the run plan.

## Orchestrator
- Runs the main workflow (rename, select, crop, autotag, training).
- Updates progress for the Queue.

## Finisher
- Packages the dataset and LoRA into ZIPs.
- Cleans up training staging files.

## What you should know
- You do not start these manually.
- If something looks stuck, check System Status first.
- Only one run is processed at a time to keep results stable.
