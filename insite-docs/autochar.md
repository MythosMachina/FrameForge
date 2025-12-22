# AutoChar

AutoChar removes unwanted tags after autotagging. Presets are shared in the WebApp.

## How presets work
- You can select multiple presets during upload.
- All rules from selected presets are combined.
- The trigger tag (the folder name without RunID) is always kept.

## Simple rules (recommended)
- Add one tag or phrase per line.
- Example: `blonde hair` removes any tag containing that text.

## Advanced rules (optional)
Rules use regex (case-insensitive).
- Exact match: `^blonde_hair$`
- Any word: `\b(bikini|swimsuit)\b`
- Ends with: `.*hair`

Edit presets in the AutoChar editor. Changes apply immediately.
