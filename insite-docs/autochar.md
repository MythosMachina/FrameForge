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

## Macros (file-based)
Macros expand to multiple tags using preset macro files in `preset/macros/`.
Macro names come from the filename (e.g., `colors.txt` -> `@colors`).

Examples:
- `@colors:hair` removes all color hair tags (e.g., `blonde_hair`, `blue_hair`).
- `@colors:eyes` removes all color eye tags.
- `@colors:fur` removes all color fur tags.
- `@animals:girl` removes `cat girl`, `wolf girl`, etc. if `animals.txt` exists.

Macro sources:
- `preset/macros/colors.txt`

Edit presets in the AutoChar editor. Changes apply immediately.
