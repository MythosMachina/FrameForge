# FrameForge

FrameForge is a “drop in your clips, get a clean dataset out” toolchain. You feed it ZIPs or videos and it handles the chores: renaming, frame grabs, smart picks, crops/flips, autotagging, and optional LoRA training (trainer runs but still needs fine-tuning). A simple web UI keeps the steps, queue, and cleanup rules (AutoChar) in one place. Everything except training works in CPU mode; GPU greatly speeds things up and the trainer requires a GPU (optimized for Nvidia).

## Features
- Drop-in pipeline (`workflow.py`) with optional autotagging (`--autotag`, `--autochar`, `--gpu`) and training (`--train`).
- Web UI with upload queue, history, multi-zip support, and AutoChar preset editor (stored in SQLite).
- Systemd units for webapp + queue watcher.
- AutoChar presets seeded (`default`, `human`, `furry`, `dragon`, `daemon`), editable in-browser.
- Fallback promotion: tagged sets in `50_ready_autotag` are auto-moved to `60_final_output` after tagging.

## Prerequisites
- Ubuntu/Debian with systemd
- Python 3.10+ and Node.js 18/20
- npm
- ffmpeg on PATH
- sudo/root to install to `/opt/FrameForge` and register services

## Install
```bash
git clone https://github.com/MythosMachina/FrameForge
cd FrameForge
cp webapp/.env.example webapp/.env   # fill in before install (ports, DATABASE_URL, secrets)
sudo scripts/install.sh
```
What the installer does:
- Rsyncs the repo to `/opt/FrameForge`
- Creates a Python venv and installs `requirements.txt`
- Runs `npm ci`, Prisma generate, and `prisma db push`
- Seeds AutoChar presets into `webapp/storage/db.sqlite`
- Writes and enables systemd units: `frameforge-webapp.service` and `frameforge-queue.service`
- Expects `webapp/.env` to be present (copied from `.env.example`); missing env will block the web app/db

Logs: `/opt/FrameForge/webapp/logs`

## Usage
- Web UI: browse to the host on port 3005 (health: `GET /health`). Upload one or more ZIPs; each ZIP becomes a run. Select AutoChar presets to combine cleanup rules.
- CLI (optional): `python workflow.py --autotag --gpu` from `/opt/FrameForge`.
- Queue watcher pulls from `00_import_queue` every 5 minutes when not training/tagging.
- Outputs land in `60_final_output/<runName>`; autotag staging is `50_ready_autotag`.

## Runtime layout
- Input/staging: `00_import_queue`, `10_input`, `20_capped_frames`, `30_work`
- Autotag staging/final: `50_ready_autotag`, `60_final_output`, `70_archive_mp4`
- Training: `trainer/jobs`, `trainer/output`, `90_final_lora`
- Web UI: `webapp` (Prisma DB at `webapp/storage/db.sqlite`, see `.env.example`)

## Development notes
- Node deps are not committed (`npm ci` during install). Python deps live in the venv.
- Runtime folders are kept empty; `.gitkeep` files preserve the structure.
- Default AutoChar presets are seeded via `scripts/seed_autochar.py` (lean defaults, no external files). Extend/tweak them in the UI; they’re stored in the Prisma DB.
- Model weights are not bundled (too large for GitHub); download/place required models under `models/` before training.

## Services
- `frameforge-webapp.service`: runs the Node web UI from `/opt/FrameForge/webapp`, logs to `webapp/logs/webapp.service.log`.
Manage with `systemctl status|logs|restart frameforge-webapp.service`.

## Cleaning helpers
`python clean.py --import|--work|--output|--train|--all` to prune inputs, workspace, outputs, or training artifacts. The webapp retains history in SQLite; delete via the UI or `DELETE /api/run/:id`.
