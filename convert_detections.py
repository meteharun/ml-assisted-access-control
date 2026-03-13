"""
convert_detections.py
Converts old-format .txt detection files to JSON.

Run from the project root:
    python3 convert_detections.py
"""

import os
import json
import ast
import glob
import re

detections_dir = os.path.join(os.path.dirname(__file__), "data", "detections")

txt_files = sorted(glob.glob(os.path.join(detections_dir, "*.txt")))
print(f"Found {len(txt_files)} .txt files to convert\n")

converted = 0
skipped   = 0
failed    = 0

for txt_path in txt_files:
    json_path = txt_path.replace(".txt", ".json")

    if os.path.exists(json_path):
        print(f"  [SKIP]  {os.path.basename(txt_path)}  (JSON already exists)")
        skipped += 1
        continue

    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    # Each detection is: label, confidence, privacy_score, [[coords]], type
    # Visual coords span many lines so we rejoin and split on detection boundaries
    entries = []
    current = ""
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        if re.match(r'^[a-zA-Z_\-]+,\s*[\d.]+,\s*[\d.]+,\s*\[', line) and current:
            entries.append(current)
            current = line
        else:
            current = (current + " " + line).strip() if current else line
    if current:
        entries.append(current)

    detections = []
    errors     = []

    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue
        try:
            match = re.match(
                r'^([a-zA-Z_\-]+),\s*([\d.]+),\s*([\d.]+),\s*(\[.*\]),\s*(\w+)$',
                entry,
                re.DOTALL
            )
            if not match:
                errors.append(f"No regex match: {entry[:80]}")
                continue

            label         = match.group(1).strip()
            confidence    = float(match.group(2))
            privacy_score = float(match.group(3))
            coords_raw    = match.group(4).strip()
            pso_type      = match.group(5).strip()
            coords        = ast.literal_eval(coords_raw)

            detection = {
                "label":         label,
                "type":          pso_type,
                "confidence":    confidence,
                "privacy_score": privacy_score,
            }
            if pso_type == "visual":
                detection["pixels"] = coords
            else:
                detection["box"] = coords

            detections.append(detection)

        except Exception as e:
            errors.append(f"{entry[:80]} — {e}")

    if errors:
        print(f"  [WARN] {os.path.basename(txt_path)}: {len(errors)} parse error(s), {len(detections)} detections saved")
        for err in errors[:3]:
            print(f"         {err}")
        failed += 1
    else:
        print(f"  [OK]   {os.path.basename(txt_path)} → {os.path.basename(json_path)}  ({len(detections)} detections)")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(detections, f, indent=2)

    converted += 1

print(f"\n{'='*50}")
print(f"  Converted : {converted}")
print(f"  Skipped   : {skipped}  (JSON already existed)")
print(f"  With warns: {failed}   (saved with partial data)")
print(f"{'='*50}")