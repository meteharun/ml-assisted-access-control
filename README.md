# ML-Assisted Access Control

This is the official repository for the paper:  
**[From See to Shield: ML-Assisted Fine-Grained Access Control for Visual Data](https://arxiv.org/abs/2510.19418)** (Akcay et al., 2025)

Full implementation details, experimental setup, and results are described in the paper.

---

## How it works

The pipeline has two stages that run in separate Docker containers:

```
[Input Image]
      │
      ▼
┌─────────────────────┐        shared volume
│   ml-container      │   data/detections/<id>.json
│   detect.py         │ ──────────────────────────────►
│                     │
│  OCR → DeBERTa      │   textual PSOs   (bounding boxes)
│  YOLOv8 → CAPC      │   multimodal PSOs (bounding boxes)
│  YOLOv8-Seg         │   visual PSOs    (pixel masks)
└─────────────────────┘
                                          │
                                          ▼
                               ┌─────────────────────┐
                               │   abe-container      │
                               │   enc_dec.py         │
                               │                      │
                               │  Fernet (AES) per    │
                               │  sensitivity group   │
                               │  wrapped under ABE   │
                               └─────────────────────┘
                                          │
                                          ▼
                               outputs/<id>_encrypted.png
                               outputs/<id>_decrypted_abekey<N>.png
```

The two stages are intentionally separated because `charm-crypto` (ABE) and the ML stack (PyTorch, PaddleOCR, YOLO) have incompatible dependencies and cannot share a single environment.

---

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed and running
- **WSL (Ubuntu)** if you are on Windows — all commands below should be run inside WSL, not PowerShell or CMD
- The `models/` folder populated with fine-tuned weights (see Datasets & Training below)

---

## Project structure

```
ml-assisted-access-control/
│
├── src/                        # All Python source code
│   ├── __init__.py
│   ├── abe.py                  # ABE wrapper (charm-crypto)
│   ├── detect.py               # Detection & classification pipeline
│   └── enc_dec.py              # Encryption / decryption pipeline
│
├── data/
│   ├── samples/                # Input images go here
│   └── detections/             # JSON output from detect.py (auto-created)
│
├── models/                     # Fine-tuned model weights (not in Git)
│   ├── multimodal/
│   │   ├── yolov8l_multimodal/weights/best.pt
│   │   └── deberta_multimodal/
│   ├── visual/
│   │   └── yolo_only_visuals_augmented_100/weights/best.pt
│   └── textual/
│       └── deberta/
│
├── notebooks/                  # Data analysis & experimentation notebooks
├── train/                      # Kaggle notebooks used to train the models
│                               # (segmentation, object detection, text classification)
├── outputs/                    # Encrypted & decrypted images (auto-created)
│
├── run.sh                      # One-command pipeline runner
├── Dockerfile.ml               # Container for detect.py
├── Dockerfile.abe              # Container for enc_dec.py
├── docker-compose.yml          # Defines both services and shared volumes
│
├── reqs_ml.txt                 # ML environment dependencies (reference)
├── reqs_abe.txt                # ABE environment dependencies (reference)
└── .gitignore
```

---

## Datasets & Training

The models used in this project were trained on three datasets:

| Modality | Dataset | Link |
|---|---|---|
| Visual (segmentation) | VISPR-Redactions | [Kaggle](https://www.kaggle.com/datasets/meteharunakcay/visual-redactions) |
| Textual (classification) | OCR-extracted from VISPR-Redactions | [Google Drive](https://drive.google.com/file/d/1g4uV7fLXCiVSXBlFK8Mf4G0O52S3XCSk/view?usp=drive_link) |
| Multimodal (object detection) | VISPR-Redactions (multimodal subset) | [Kaggle](https://www.kaggle.com/datasets/harunakay/multimodal) |

Training notebooks for all three modalities are in the `train/` folder. These are Kaggle notebooks and were run on Kaggle's T4x2 GPUs.

---

## Running the pipeline

### Step 0: Navigate to the project root in WSL

```bash
cd /mnt/c/Users/<your_username>/Desktop/ml-assisted-access-control
```

### Step 1: Build the Docker images

Only needed once, or after you change a Dockerfile or anything in `src/`:

```bash
docker-compose build
```

This builds two images:
- `pesto-pacman-ml` — installs PyTorch, PaddleOCR, YOLO, DeBERTa
- `pesto-pacman-abe` — compiles `libpbc` and `charm-crypto` from source

The first build will take several minutes. Subsequent builds use the Docker cache and are near-instant.

### Step 2: Make the run script executable

Only needed once:

```bash
chmod +x run.sh
```

### Step 3: Run the pipeline

```bash
./run.sh your_image.jpg
```

The image must be inside `data/samples/`. The script will:
1. Run detection and write `data/detections/your_image.json`
2. Encrypt the image and save `outputs/your_image_encrypted.png`
3. Prompt you for an ABE key to decrypt with

When prompted:
```
Which key do you have? (abekey1, abekey2, abekey3, abekey4):
```

Type one of the four keys and press Enter. Higher keys unlock more sensitive content:

| Key | Sensitivity threshold | Decrypts |
|---|---|---|
| `abekey1` | score ≤ 0.25 | Lowest sensitivity only |
| `abekey2` | score ≤ 0.45 | + faces, license plates |
| `abekey3` | score ≤ 0.75 | + names, locations, signatures |
| `abekey4` | score ≤ 1.00 | Everything |

To use real privacy scores from the user study instead of synthetic ones:
```bash
./run.sh your_image.jpg real
```

> **Note on scores:** Synthetic scores (default) spread objects across all four sensitivity groups, producing better visualisations. Real scores from the user study cluster most objects at high sensitivity.

### Step 4: Find your output images

```
outputs/
├── your_image_encrypted.png           ← all PSO regions scrambled
└── your_image_decrypted_abekey<N>.png ← regions you have access to restored
```

On Windows these appear at:
```
C:\Users\<your_username>\Desktop\ml-assisted-access-control\outputs\
```

---

## Running a different image

Just pass a different filename — detection results are cached per image in `data/detections/` so if you have already run detection on an image before, the script skips straight to encryption:

```bash
./run.sh another_image.jpg
```

---

## Troubleshooting

**`Network needs to be recreated` error**
```bash
docker-compose down
./run.sh your_image.jpg
```

**`No such file or directory` for the detection JSON**  
Detection did not finish before encryption started. This should not happen when using `run.sh` since it runs the two stages sequentially. If running stages manually, always wait for `[SAVED] N detections →` to appear in the detect logs before running encryption.

**`libpbc.so.1` not found**  
This should not happen inside Docker as `libpbc` is compiled at image build time. If it occurs when running scripts directly outside Docker:
```bash
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
```

**PaddleOCR model download is slow**  
PaddleOCR downloads its weights on first run (~16 MB total). These are cached inside the container's filesystem. If you want to persist them across container restarts, add a volume mount for `/root/.paddleocr` in `docker-compose.yml`.

**Shell lost current directory (`No such file or directory` on `os.getcwd()`)**  
```bash
cd /mnt/c/Users/<your_username>/Desktop/ml-assisted-access-control
```

**`libgl1-mesa-glx` not found during build**  
This package was removed in Debian trixie. Make sure your `Dockerfile.ml` uses `libgl1` instead of `libgl1-mesa-glx`.

---

## Running stages manually (advanced)

If you need to run detection and encryption separately without using `run.sh`:

**Detection only:**
```bash
IMAGE=your_image.jpg docker-compose up detect
```

**Encryption only** (after detection has finished):
```bash
docker run -it --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/outputs:/app/outputs" \
  pesto-pacman-abe \
  python src/enc_dec.py \
    --image /app/data/samples/your_image.jpg \
    --detections /app/data/detections \
    --output-dir /app/outputs \
    --real-scores false
```

Note: `docker-compose up encrypt` is not used for the encryption stage because `docker-compose up` does not support interactive terminal prompts. The `docker run -it` command is required.

---

## Script reference

### `detect.py`

```
python src/detect.py --image PATH [--output-dir DIR] [--model-dir DIR] [--real-scores BOOL]

  --image         Path to the input image (required)
  --output-dir    Where to write the detection JSON (default: data/detections/)
  --model-dir     Root dir containing the models/ folder (default: project root)
  --real-scores   true | false  (default: false)
```

### `enc_dec.py`

```
python src/enc_dec.py --image PATH [--detections DIR] [--output-dir DIR]
                      [--real-scores BOOL] [--privacy-levels N]

  --image           Path to the input image (required)
  --detections      Directory containing the JSON from detect.py (default: data/detections/)
  --output-dir      Where to write output images (default: outputs/)
  --real-scores     true | false  (default: false) — must match what was used in detect.py
  --privacy-levels  Number of sensitivity groups (default: 4)
```

---

## Detection metadata format

`detect.py` writes one JSON file per image to `data/detections/`. Each entry represents one detected PSO:

```json
[
  {
    "label": "name",
    "type": "textual",
    "confidence": 0.94,
    "privacy_score": 0.7,
    "box": [[120, 45], [310, 45], [310, 68], [120, 68]]
  },
  {
    "label": "face",
    "type": "visual",
    "confidence": 1.0,
    "privacy_score": 0.5,
    "pixels": [[102, 88], [102, 89], "..."]
  }
]
```

---

## Citation

If you use this code or the datasets in your research, please cite:

```bibtex
@article{akcay2025shield,
  title     = {From See to Shield: ML-Assisted Fine-Grained Access Control for Visual Data},
  author    = {Akcay, Mete Harun and Atli, Buse Gul and Rao, Siddharth Prakash and Bakas, Alexandros},
  journal   = {arXiv preprint arXiv:2510.19418},
  year      = {2025}
}
```

---

## OS compatibility

- **WSL (Ubuntu) on Windows:** Fully supported and recommended. All commands in this README assume WSL.
- **Linux:** Fully supported. Commands are identical.
- **Windows (PowerShell/CMD):** Not recommended. Docker commands differ and path handling is error-prone. Use WSL instead.
- **macOS:** Docker works. `charm-crypto` compilation on Apple Silicon may require Homebrew-installed `gmp` and `pbc` if running outside Docker.