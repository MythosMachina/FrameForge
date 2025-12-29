# Tagger Models (Autotag)

Autotag uses a tagger model to write tags for each image. You can pick which model to use.

## Recommended model
- `SmilingWolf/wd-swinv2-tagger-v3` is a strong all-rounder.
- It is a good default for most datasets.

## Download a tagger model
1) Open the Tagger Models tab (if available in your UI).
2) Click Download.
3) Paste the repo ID (example: `SmilingWolf/wd-swinv2-tagger-v3`).
4) Wait for the status to show Ready.

## Set the default model
1) Open Settings.
2) Find `autotag_model_id`.
3) Paste the repo ID you want as default.
4) Save.

## When to use a different model
- If tags feel too noisy, try a different model.
- If tags miss details, try a larger model.

## If the model shows as missing
- The download did not complete.
- Re-download the model or ask an admin to check network access.
