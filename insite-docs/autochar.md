# AutoChar (tag cleanup)

AutoChar removes unwanted tags after Autotag. You control it with presets.

## What it does
- Keeps the first tag (the trigger) always.
- Removes any tag that matches a preset rule.

## Presets
- Create or edit presets in the AutoChar tab.
- You can select one or more presets during Upload.
- If AutoChar is ON and no preset is selected, a default preset is used (if available).

## Rules (simple)
- One pattern per line (or comma-separated).
- Wildcards use `*` and are case-insensitive.
- Spaces and underscores are treated the same.

Examples:
- `blue_hair` removes only blue hair tags.
- `*_hair` removes any hair color tag.
- `bikini, swimsuit` removes both tags.

## Macros (optional)
Macros expand to many tags at once.
- Use `@macro` in a preset rule.
- Example: `@colors:hair` expands to all hair color tags.
- If a macro is unknown, it is ignored.

Tip: Use AutoChar to remove noisy or repetitive tags so your dataset stays clean.
