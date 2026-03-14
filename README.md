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
│   ml-container      │   data/detections/<id>.txt
│   detect.py         │ ──────────────────────────────►
│                     │
│  OCR → DeBERTa      │   textual PSOs   (bounding boxes)
│  YOLOv8 → CAPC      │   multimodal PSOs (bounding boxes)
│  YOLOv8-Seg         │   visual PSOs    (pixel masks)
└─────────────────────┘
                                          │
                                          ▼
                               ┌─────────────────────┐
                               │   abe-container     │
                               │   enc_dec.py        │
                               │                     │
                               │  Fernet (AES) per   │
                               │  sensitivity group  │
                               │  wrapped under ABE  │
                               └─────────────────────┘
                                          │
                                          ▼
                               outputs/<id>_encrypted.png
                               outputs/<id>_decrypted_abekey<N>.png
```

The two stages are intentionally separated because `charm-crypto` (ABE) and the ML stack (PyTorch, PaddleOCR, YOLO) have incompatible dependencies and cannot share a single environment.

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
├── models/                     # Fine-tuned model weights (not in Git, see below)
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

## Prerequisites

### 1. Docker

Install [Docker](https://docs.docker.com/get-docker/) and make sure it is running.

If you are on **Windows**, all commands must be run inside **WSL (Ubuntu)**, not PowerShell or CMD. Open WSL and navigate to the project:

```bash
cd /mnt/c/Users/<your_username>/Desktop/ml-assisted-access-control
```

### 2. Model weights

The `models/` folder must be populated with fine-tuned weights before running the pipeline. The models are **not included in this repository** due to file size. Download them and place them in the correct paths:

| Model | Purpose | Expected path |
|---|---|---|
| YOLOv8-Seg (visual) | Detects faces, persons, signatures, etc. | `models/visual/yolo_only_visuals_augmented_100/weights/best.pt` |
| YOLOv8 (multimodal) | Detects ID cards, passports, receipts, etc. | `models/multimodal/yolov8l_multimodal/weights/best.pt` |
| DeBERTa (multimodal) | CAPC post-correction for multimodal PSOs | `models/multimodal/deberta_multimodal/` |
| DeBERTa (textual) | Classifies OCR-extracted text regions | `models/textual/deberta/` |

The DeBERTa folders must contain the standard HuggingFace model files: `config.json`, `pytorch_model.bin`, `tokenizer_config.json`, `tokenizer.json`, `special_tokens_map.json`.

The models were trained on the datasets below. Training notebooks are in the `train/` folder and were run on Kaggle's T4x2 GPUs.

| Modality | Dataset | Link |
|---|---|---|
| Visual | VISPR-Redactions | [Kaggle](https://www.kaggle.com/datasets/meteharunakcay/visual-redactions) |
| Textual | OCR-extracted from VISPR-Redactions | [Google Drive](https://drive.google.com/file/d/1g4uV7fLXCiVSXBlFK8Mf4G0O52S3XCSk/view?usp=drive_link) |
| Multimodal | VISPR-Redactions (multimodal subset) | [Kaggle](https://www.kaggle.com/datasets/harunakay/multimodal) |

---

## Running the pipeline

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
sed -i 's/\r//' run.sh
chmod +x run.sh
```

The first line strips Windows line endings from the script, which cause a `bad interpreter: /bin/bash^M` error when running from WSL. The second makes it executable.

### Step 3: Run the pipeline

```bash
./run.sh your_image.jpg
```

The image must be inside `data/samples/`. The script will:
1. Run detection and write `data/detections/your_image.txt`
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

Just pass a different filename. Detection results are cached per image in `data/detections/` so if you have already detected an image before, the script skips straight to encryption:

```bash
./run.sh another_image.jpg
```

---

## Troubleshooting

**Detection fails with `No file named pytorch_model.bin found in directory`**  
The model weights are missing. See Prerequisites → Model weights above and make sure all four models are downloaded and placed in the correct paths before running.

**`Network needs to be recreated` error**  
`run.sh` handles this automatically by running `docker-compose down` before each detection stage. If you see it outside of `run.sh`, run:
```bash
docker-compose down
```

**`No such file or directory` for the detection file**  
Detection failed before writing its output. Check the detection logs above the error message — it will tell you exactly what went wrong (usually missing model weights).

**`libpbc.so.1` not found**  
This should not happen inside Docker as `libpbc` is compiled at image build time. If it occurs when running scripts directly outside Docker:
```bash
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
```

**PaddleOCR downloads models on every run**  
PaddleOCR caches weights inside the container filesystem, which is discarded when the container stops. To persist the cache across runs, add this volume to the `detect` service in `docker-compose.yml`:
```yaml
- paddleocr-cache:/root/.paddleocr
```
And add at the bottom of the file:
```yaml
volumes:
  paddleocr-cache:
```

**Shell lost current directory (`No such file or directory` on `os.getcwd()`)**
```bash
cd /mnt/c/Users/<your_username>/Desktop/ml-assisted-access-control
```

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
  --output-dir    Where to write the detection file (default: data/detections/)
  --model-dir     Root dir containing the models/ folder (default: project root)
  --real-scores   true | false  (default: false)
```

### `enc_dec.py`

```
python src/enc_dec.py --image PATH [--detections DIR] [--output-dir DIR]
                      [--real-scores BOOL] [--privacy-levels N]

  --image           Path to the input image (required)
  --detections      Directory containing the .txt file from detect.py (default: data/detections/)
  --output-dir      Where to write output images (default: outputs/)
  --real-scores     true | false  (default: false) — must match what was used in detect.py
  --privacy-levels  Number of sensitivity groups (default: 4)
```

---

## Detection metadata format

`detect.py` writes one `.txt` file per image to `data/detections/`. Each line represents one detected PSO in the format:

```
label, confidence, privacy_score, [[coords]], type
```

For textual and multimodal PSOs, `coords` is a list of 4 corner points. For visual PSOs, `coords` is the full list of pixel coordinates. For example:

```
name, 0.94, 0.7, [[120, 45], [310, 45], [310, 68], [120, 68]], textual
face, 1.0, 0.5, [[102, 88], [102, 89], ...], visual
passport, 0.98, 0.6, [[50, 30], [200, 30], [200, 150], [50, 150]], multimodal
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
