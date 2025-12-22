# What is FrameForge?

FrameForge is a full pipeline-driven training application with three orchestrators. It turns ZIP uploads into clean, tagged image datasets, can train a LoRA, and packages downloads for you.

## Who it serves
- Creators: upload, configure, and download datasets or LoRAs.
- Reviewers: monitor runs, check results, and request fixes.

## RunID
Every upload gets a 6-digit RunID. You will see it in the queue and filenames, but it is removed from final tags and downloads.

## Staged uploads
Uploads go into a short staging area. Click Launch to create runs. Staged items expire after a few minutes if not launched.
