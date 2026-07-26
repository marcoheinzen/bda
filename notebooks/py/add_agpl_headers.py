# Copyright (C) 2024-2026 Marco Heinzen
# SPDX-License-Identifier: AGPL-3.0-or-later
# Part of the Master Thesis "Building Damage Assessment with Multimodal
# Satellite Time Series and Machine Learning in the Russia-Ukraine War 2022-2026"
# Code hosted at https://github.com/marcoheinzen/bda
# Parts of this code were written or improved with the assistance of
# Claude (Anthropic); all other code, and the concept, research, architecture,
# design, execution, testing and validation throughout, are the author's work.

#!/usr/bin/env python3
"""
add_agpl_headers.py
Run from notebooks/ directory to add AGPL headers to all .py files.
Skips files that already have AGPL header. Removes old MIT/CC headers.

Usage:
    cd F:\PROJECTS\masterthesis\gdrive\masterthesis\notebooks
    python add_agpl_headers.py
"""
import sys
from pathlib import Path

AGPL_HEADER = (
    "# bda -- Building Damage Assessment using Sentinel-1/2 satellite imagery\n"
    "# Copyright (C) 2024-2026 Marco Heinzen\n"
    "# SPDX-License-Identifier: AGPL-3.0-or-later\n"
    "#\n"
    "# This program is free software: you can redistribute it and/or modify\n"
    "# it under the terms of the GNU Affero General Public License as published by\n"
    "# the Free Software Foundation, either version 3 of the License, or\n"
    "# (at your option) any later version.\n"
    "\n"
)

OLD_MARKERS = ["SPDX-License-Identifier: MIT", "CC BY 4.0", "MIT License"]

py_files = sorted(Path(".").glob("*.py"))
print(f"Found {len(py_files)} .py files in {Path('.').resolve()}")

updated = 0
skipped = 0
for f in py_files:
    if f.name == "add_agpl_headers.py":
        continue
    content = f.read_text(encoding="utf-8")

    if "AGPL-3.0-or-later" in content[:500]:
        print(f"  SKIP (already AGPL): {f.name}")
        skipped += 1
        continue

    # Remove old MIT/CC header lines if present
    lines = content.split("\n")
    cleaned = []
    in_old_header = False
    for line in lines:
        if any(marker in line for marker in OLD_MARKERS):
            in_old_header = True
            continue
        if in_old_header and line.strip().startswith("#") and len(line.strip()) < 80:
            # skip continuation of old header
            if any(kw in line for kw in ["MIT", "CC BY", "Published", "Author:", "License"]):
                continue
        in_old_header = False
        cleaned.append(line)

    content = "\n".join(cleaned)

    # Insert AGPL header after docstring if present, else at top
    if content.lstrip().startswith('"""') or content.lstrip().startswith("'''"):
        quote = '"""' if content.lstrip().startswith('"""') else "'''"
        first = content.find(quote)
        second = content.find(quote, first + 3)
        if second > 0:
            insert_pos = content.find("\n", second) + 1
            content = content[:insert_pos] + "\n" + AGPL_HEADER + content[insert_pos:]
        else:
            content = AGPL_HEADER + content
    else:
        content = AGPL_HEADER + content

    f.write_text(content, encoding="utf-8")
    print(f"  UPDATED: {f.name}")
    updated += 1

print(f"\nDone: {updated} updated, {skipped} skipped")
