#!/usr/bin/env python3
"""
prepare_subblock_dataset.py

1) Converts LabelMe JSONs → YOLO .txt (for JSONs with shapes matching classes)
2) Splits labeled images into train/val/test (80/10/10) if any labels exist
3) Generates subblock_data.yaml for YOLOv8
"""
import json
import random
import shutil
from pathlib import Path

# === Configuration ===
PROJECT_ROOT    = Path(__file__).parent.parent.resolve()
DATA_JSON_DIR   = PROJECT_ROOT / "data" / "to_label_images"
LABELS_DIR      = PROJECT_ROOT / "labels" / "to_label"
OUTPUT_ROOT     = PROJECT_ROOT / "subblock_dataset"
CONFIG_DIR      = PROJECT_ROOT / "config"
CLASSES_FILE    = CONFIG_DIR / "subblock_classes.txt"
CONFIG_YAML     = CONFIG_DIR / "subblock_data.yaml"
TRAIN_FRAC, VAL_FRAC = 0.8, 0.1
SEED = 42
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

# Load class names
def load_classes(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Classes file not found: {path}")
    return [l.strip() for l in path.read_text().splitlines() if l.strip()]

# Convert a single LabelMe JSON to YOLO .txt
def convert_json_to_yolo(json_fp: Path, classes, out_dir: Path):
    data = json.loads(json_fp.read_text())
    w, h = data.get("imageWidth"), data.get("imageHeight")
    yolo_lines = []
    for shape in data.get("shapes", []):
        raw_lbl = shape.get("label", "").strip()
        # normalize label: lowercase, replace spaces with underscores
        lbl = raw_lbl.lower().replace(" ", "_")
        if lbl not in classes:
            continue
        cid = classes.index(lbl)
        xs = [p[0] for p in shape["points"]]
        ys = [p[1] for p in shape["points"]]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        x_c = ((x1 + x2) / 2) / w
        y_c = ((y1 + y2) / 2) / h
        bw  = (x2 - x1) / w
        bh  = (y2 - y1) / h
        yolo_lines.append(f"{cid} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}")
    if yolo_lines:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{json_fp.stem}.txt").write_text("".join(yolo_lines))
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{json_fp.stem}.txt").write_text("\n".join(yolo_lines))

# Find image filenames that have been labeled
def find_labeled_images(img_dir: Path, lbl_dir: Path):
    stems = {txt.stem for txt in lbl_dir.glob("*.txt")}
    images = []
    for stem in stems:
        for ext in IMAGE_EXTS:
            img_fp = img_dir / f"{stem}{ext}"
            if img_fp.exists():
                images.append(img_fp.name)
                break
    return images

# Split based on fractions
def split_filenames(names):
    random.shuffle(names)
    n = len(names)
    n_train = int(n * TRAIN_FRAC)
    n_val   = int(n * (TRAIN_FRAC + VAL_FRAC))
    return names[:n_train], names[n_train:n_val], names[n_val:]

# Copy images and txts into split folders
def copy_split(names, split, img_src: Path, lbl_src: Path, out_root: Path):
    img_out = out_root / split / "images"
    lbl_out = out_root / split / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    for name in names:
        for ext in IMAGE_EXTS:
            src_img = img_src / f"{Path(name).stem}{ext}"
            if src_img.exists():
                shutil.copy(src_img, img_out / src_img.name)
                break
        shutil.copy(lbl_src / f"{Path(name).stem}.txt", lbl_out / f"{Path(name).stem}.txt")

# Write YOLOv8 data yaml
def write_data_yaml(path: Path, base: Path, splits, classes):
    lines = [f"path: {base}",
             f"train: {splits[0]}/images",
             f"val:   {splits[1]}/images",
             f"test:  {splits[2]}/images", "",
             f"nc: {len(classes)}", "names:"]
    lines += [f"  - {c}" for c in classes]
    path.write_text("\n".join(lines))

# Main workflow
def main():
    random.seed(SEED)
    classes = load_classes(CLASSES_FILE)

    # Step 1: Convert all JSONs to YOLO labels
    json_files = list(DATA_JSON_DIR.glob("*.json"))
    print(f"Found {len(json_files)} JSON files in {DATA_JSON_DIR}")
    for jp in json_files:
        convert_json_to_yolo(jp, classes, LABELS_DIR)

    # Count resulting .txt files
    txt_files = list(LABELS_DIR.glob("*.txt"))
    print(f"Generated {len(txt_files)} YOLO label files in {LABELS_DIR}")
    if not txt_files:
        print("⚠️  No YOLO labels generated. Make sure your JSONs have shapes with labels matching the classes file.")
        return

    # Step 2: Gather and split labeled images
    labeled = find_labeled_images(DATA_JSON_DIR, LABELS_DIR)
    print(f"Preparing splits for {len(labeled)} labeled images")
    train, val, test = split_filenames(labeled)

    # Step 3: Copy to train/val/test directories
    for split_name, split_list in zip(["train","val","test"], [train,val,test]):
        copy_split(split_list, split_name, DATA_JSON_DIR, LABELS_DIR, OUTPUT_ROOT)
        print(f" -> {split_name}: {len(split_list)} images")

    # Step 4: Write data.yaml
    write_data_yaml(CONFIG_YAML, OUTPUT_ROOT, [OUTPUT_ROOT/n for n in ("train","val","test")], classes)
    print(f"✅ Dataset preparation complete. YAML written to {CONFIG_YAML}")

if __name__ == "__main__":
    main()
