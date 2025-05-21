# Deep Learning for Circuit Intelligence: A Modular Pipeline for Subblock-Aware Interpretation of Analog Circuit Schematics

A modular pipeline for schematic image analysis and circuit topology extraction. The repository is organized into distinct stages, from raw data ingestion through component/text removal, wire detection, graph construction, and final HTML reporting.

---

## Directory Structure

```
├── .venv/                          # Python virtual environment
├── data/
│   ├── raw/                        # Original schematic images
│   ├── preprocessed/               # Binarized or grayscale inputs for detection
│   ├── processed/                  # Per-image folders with cleaned images and line results
│   ├── yolo_subblocks/             # Images for YOLO subblock labeling
│   └── metadata.csv                # Optional metadata for each schematic
├── external/
│   ├── Deep-Hough-Transform-Li...  # External repos (e.g. alternate line detectors)
│   └── lcnn/                       # External line CNN implementations
├── line_processing/
│   └── scripts/                    # Hough + net/node labeling utilities
│       ├── component_mapping.py
│       ├── hough_detection.py
│       ├── intersections.py
│       ├── net_labeling.py
│       ├── node_clustering.py
│       ├── segment.py
│       └── terminal_detection.py
├── notebooks/
│   ├── main.ipynb                  # End-to-end execution orchestrator
│   ├── line_processing.ipynb       # Detailed wire extraction and grouping
│   ├── outputs.ipynb               # Preview & sample visualizations
│   └── training_subblocks.ipynb    # YOLO subblock labeling & training workflow
├── outputs/                        # Generated HTML detail pages and artifacts
├── runs/                           # Ultralytics detect/train logs
│   └── detect/
├── scripts/                        # Core image + OCR + component helper scripts
│   ├── __init__.py
│   ├── combine_yolo_ocr_data.py
│   ├── component_detector.py
│   ├── graph_builder.py
│   ├── image_utils.py
│   ├── ocr_utils.py
│   ├── remove_functions.py
│   └── text_to_component.py
├── subblock_detect/                # YOLO subblock detection dataset & config
│   ├── config/
│   ├── data/                       # raw_images/, to_label_images/
│   ├── labelImg/                   # LabelImg configs
│   ├── labels/
│   ├── scripts/
│   └── subblock_dataset/
├── trained_models/                 # YOLO weights, e.g. exp_stage2_best.pt
├── .gitignore
├── README.md                       # ← this file
└── requirements.txt                # Python dependencies
```

---

## Pipeline Overview & Notebook Order

1. **`notebooks/main.ipynb`**

   * High-level orchestration: calls component detection, OCR, cleaning, line detection, and report scripts.

2. **Component & OCR Preprocessing**

   * **Script:** `scripts/component_detector.py` & `scripts/ocr_utils.py`
   * **Notebook:** invoked inside `main.ipynb`
   * **Outputs:** `data/processed/<img_id>/_detected.png`, `components.json`, `ocr_output.json`

3. **Image Cleaning**

   * **Script:** `scripts/remove_functions.py`
   * **Notebook:** `notebooks/main.ipynb` cleaning section
   * **Output:** `data/processed/<img_id>/cleaned.png`

4. **Line Detection & Net Labeling**

   * **Scripts:** `line_processing/scripts/hough_detection.py`, `net_labeling.py`, `node_clustering.py`
   * **Notebook:** `notebooks/line_processing.ipynb`
   * **Output:** `data/processed/<img_id>/line_results/segments.json`, `nets.json`, `nodes.json`

5. **Visualization & Sampling**

   * **Notebook:** `notebooks/outputs.ipynb`
   * Random sampling of cleaned images, component overlays, and line-segment plots for QA.

6. **HTML Report Generation**

   * **Scripts:** `make_detail_pages.py`, `make_gallery.py`
   * **Output:** `outputs/<img_id>/<img_id>.html`, `outputs/gallery.html`

---

## Current Status & Next Steps

* **Completed:** component detection, OCR cleanup, Hough-based line detection, basic net labeling, README & HTML reports.
* **To-do:** graph-based net merging, terminal-to-node connectivity, advanced subblock detection integration.

*Last updated: May 21, 2025*
