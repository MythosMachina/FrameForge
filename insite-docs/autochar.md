# AutoChar

AutoChar removes unwanted tags after autotagging. Presets are shared in the WebApp.

## How presets work
- You can select multiple presets during upload.
- All rules from selected presets are combined.
- The trigger tag (the folder name without RunID) is always kept.

## Rules (clear words + wildcards)
- Add one tag or wildcard per line (or comma-separated).
- Wildcards use `*` and are case-insensitive.
- `_` and spaces are treated the same (e.g., `blue_hair` matches `blue hair`).
- The trigger tag (first tag) is always preserved.

Examples:
- `*_hair` removes any hair color tag.
- `blue_hair` removes only blue hair.
- `bikini, swimsuit` removes both tags.

Edit presets in the AutoChar editor. Changes apply immediately.
