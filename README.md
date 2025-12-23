# FrameForge

## Overview
FrameForge is a full, pipeline-driven training application with three orchestrators. It turns ZIP uploads into clean, tagged image datasets, can train LoRAs, and packages downloads automatically. A web UI is included for non-technical users, with optional CLI workflows for power users.

„FrameForge is designed for fictional, stylized, and synthetic content.
Use on real individuals without consent is explicitly discouraged.“

## Highlights
- End-to-end pipeline: import → select → crop → tag → (optional) train → package.
- Orchestrated services: initiator, orchestrator, finisher.
- Web-first usage: queue monitoring, manual tagging, downloads.
- Training is fully integrated with **Kohya_ss** scripts (based on the Kohya_ss project: https://github.com/bmaltais/kohya_ss).

## What it does
- Ingests ZIP uploads (images or videos).
- Extracts frames, selects diverse images, and generates crops/flips.
- Autotags and cleans tags with AutoChar presets.
- Trains LoRAs when enabled and saves samples.
- Packages datasets and LoRAs for download.

## Requirements
- Python 3.x
- Node.js + npm
- ffmpeg in PATH
- GPU recommended for tagging; required for training

## Security notice (local-only)
FrameForge is intended for local use only. Do not expose it to the public
internet. At most, run it on a trusted LAN; never on WAN.

## Install (single venv)
Use the consolidated setup script:
```bash
./scripts/setup_all.sh
```
This will:
- Create a single venv and install Python deps
- Install WebApp dependencies and set up the database
- Write systemd unit files under `./systemd`
- Start model downloads in the background

## Services (systemd)
After running setup, install the unit files:
```bash
sudo cp systemd/frameforge-*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now frameforge-db-broker frameforge-webapp frameforge-initiator frameforge-orchestrator frameforge-finisher
```

## Web UI usage
- Upload ZIPs, choose options (Autotag, AutoChar, Facecap, Train, GPU), and Launch.
- Track status in Queue and download results in History.
- Manual Tagging pauses a run until captions are approved.

## CLI usage
```bash
./.venv/bin/python workflow.py --autotag --gpu
```
Add `--train` to run LoRA training.

## Paths
- Inputs: `INBOX/`
- Outputs: `OUTPUTS/datasets/`, `OUTPUTS/loras/`
- Downloads: `ARCHIVE/zips/`
- Internals: `_system/workflow/*`, `_system/trainer/*`, `_system/webapp/*`, `_system/db/db.sqlite`

## Credits
Training pipeline uses **Kohya_ss** scripts (https://github.com/bmaltais/kohya_ss).
