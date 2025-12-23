# FrameForge

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/)
[![Node](https://img.shields.io/badge/node.js-required-green.svg)](https://nodejs.org/)

End-to-end dataset and LoRA training pipeline that turns ZIP uploads into clean, tagged image datasets with a web-first workflow.

> FrameForge is designed for fictional, stylized, and synthetic content.  
> Use on real individuals without consent is explicitly discouraged.

>Importrant Notice: FrameForge is currently only for PonyXL Training calibrated. More will follow if Demand comes.

## Features
- Pipeline stages: import → select → crop → tag → (optional) train → package.
- Orchestrated services: initiator, orchestrator, finisher, DB broker, webapp.
- Web UI for uploads, queue monitoring, manual tagging, and downloads.
- Training integration with ([**Kohya_ss** scripts](https://github.com/kohya-ss/sd-scripts)).

## Quick Start
```bash
./scripts/setup_all.sh
sudo cp systemd/frameforge-*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now frameforge-db-broker frameforge-webapp frameforge-initiator frameforge-orchestrator frameforge-finisher
```

Open the Web UI at `http://localhost:3005`.

Prerequisites: Python 3, Node.js + npm, ffmpeg in PATH. GPU recommended for tagging; required for training.

## Configuration

| Variable | Default | Notes |
| --- | --- | --- |
| `DB_BROKER_URL` | (set by systemd) | Required for webapp DB access. |
| `PORT` | `3005` | Web UI port. |
| `HOST` | `0.0.0.0` | Web UI bind address. |

## Usage
```bash
./.venv/bin/python workflow.py --autotag --gpu
```
Add `--train` to run LoRA training.

For UI usage and workflows, see the docs:
- `insite-docs/quickstart.md`
- `insite-docs/ui.md`
- `insite-docs/workflow.md`

## Development
- Setup (all-in-one): `./scripts/setup_all.sh`
- Services: `systemd/frameforge-*.service`

## Security
FrameForge is intended for local use only. Do not expose it to the public internet.

## License
MIT
