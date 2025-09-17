# Partial Encryption via Segmentation

This branch of the repository contains the full pipeline for **ML-assisted access control**, consisting of two major components:

- **Detection of private regions(`ml-env`)**: Used for running segmentation, object detection and OCR + text classification.  
  - script: `scripts/detect.py`

- **Attribute-Based Encryption (`abe-env`)**: Used for encrypting/decrypting detected private regions using Charm-Crypto.  
  - script: `scripts/enc-dec.py`

⚠️ **Important:** You must use the correct environment depending on which script you want to run.  
- For detection/classification: activate **`ml-env`**  
- For encryption/decryption: activate **`abe-env`**

## Prerequisites

- **Operating System:** A Linux-based environment is highly recommended for this project, especially due to `charm-crypto`'s compilation requirements.
- **If on Windows:** Please ensure you have Windows Subsystem for Linux (WSL) with an Ubuntu distribution properly installed and configured. All commands below should be run within your WSL Ubuntu terminal.
- **Git:** Git must be installed on your system (or within your WSL environment).

---

## Project Folder Structure

The repository is organized as follows:

```plaintext
partial-encryption-via-segmentation/
│
├── abe-env/              # Virtual environment for encryption/decryption (not pushed to Git)
├── ml-env/               # Virtual environment for detection/classification (not pushed to Git)
├── charm/                # Source code for Charm-Crypto (needed to compile/install)
│
├── data/                 # Dataset files (images, annotations, etc.)
├── detections/           # Output detections from YOLO/PaddleOCR pipeline
├── models/               # Pretrained or fine-tuned models (YOLO, DeBERTa, etc.)
├── notebooks/            # Jupyter notebooks for experiments and prototyping
├── outputs/              # Generated results, logs, visualizations
├── scripts/              # Main scripts for running the pipeline
│   ├── detect.py         # Detection + OCR + classification pipeline (uses `ml-env`)
│   ├── enc-dec.py        # Encryption/decryption pipeline (uses `abe-env`)
│   └── ...               # Other helper scripts
├── train/                # Training scripts and configurations
│
├── reqs_abe.txt      # Dependency list for `abe-env`
├── reqs_ml.txt  # (Optional) Dependency list for `ml-env`
├── README.md             # Project documentation
└── .gitignore            # Ignored files and folders (envs, cache, etc.)
```

## 1. Setting up the repository


Follow these steps in your Linux terminal (or WSL Ubuntu terminal):

### Step 1: Install Essential System Dependencies

These packages provide build tools, Python utilities, and cryptographic libraries necessary for the project, especially for compiling `charm-crypto`.

```bash
sudo apt update
sudo apt install python3 python3-venv python3-pip git build-essential libgmp-dev libssl-dev libsodium-dev dos2unix
```

- `build-essential`: Provides core C/C++ compilation tools (`gcc`, `g++`, `make`).
- `libgmp-dev`, `libssl-dev`, `libsodium-dev`: Cryptographic and general development libraries required by `charm-crypto`.
- `dos2unix`: A utility to convert Windows-style line endings to Unix-style, crucial for shell scripts.

---

### Step 2: Clone the Project Repository

Clone the specific `full_pipeline` branch of the repository.

```bash
git clone -b full_pipeline https://gitlabe2.ext.net.nokia.com/network-security/ai-ml-security/partial-encryption-via-segmentation.git

```

---

### Step 3: Navigate to the Project Directory

Enter the newly cloned project folder.

```bash
cd partial-encryption-via-segmentation
```

## 2. Setting up `abe-env`


### Step 1: Create and Activate a New Virtual Environment


```bash
python3 -m venv abe-env
source abe-env/bin/activate
```

You should now see `(abe-env)` at the beginning of your terminal prompt, indicating that your virtual environment is active.

---

### Step 2: Install Required Python Packages

First, install the standard Python libraries. Then, install `charm-crypto` from the source code included in the repository.

#### a. Install General Python Packages:

```bash
pip install numpy Pillow cryptography
```

#### b. Install charm-crypto from Source:

The `charm-crypto` source code is included directly within this repository. You'll compile and install it within your virtual environment.

**i. Navigate into the charm directory:**

```bash
cd charm/
```

**ii. Ensure `configure.sh` is executable and has correct line endings:**

```bash
chmod +x configure.sh
dos2unix configure.sh
```

**iii. Run the configuration script and install `charm-crypto`:**

This process involves compilation and may take several minutes.

```bash
./configure.sh
pip install .
```

**iv. Return to the main project directory:**

```bash
cd ..
```

---

### Step 3: Run the `enc-dec.py` Script

With all dependencies installed and the environment correctly set up, you can now execute the encryption/decryption script.

```bash
python3 scripts/enc-dec.py
```

#### Important Parameters for `enc-dec.py`

- **`is_real`**  
  - If set to `true`, the program will use scores from the user study.  
  - ⚠️ These scores are **not ideal for visualizations**.  
  - For testing whether the program is functioning correctly, it is recommended to set this to `false`.

- **`IMAGE_PATH`**  
  - Path to the image that will be encrypted.

- **`DET_PATH`**  
  - Path to the detections file corresponding to the given image.

Exit the virtual environment.
```bash
deactivate
```

## 3. Setting up `ml-env`


Follow these steps in your Linux terminal (or WSL Ubuntu terminal):

### Step 1: Create and Activate a New Virtual Environment


```bash
python3 -m venv ml-env
source ml-env/bin/activate
```

You should now see `(ml-env)` at the beginning of your terminal prompt, indicating that your virtual environment is active.

---

### Step 2: Install Required Python Packages

#### a. Upgrade pip:

```bash
python -m pip install --upgrade pip
```

#### b. Install Dependencies:

```bash
pip install matplotlib==3.10.3 opencv-python==4.6.0.66 transformers==4.39.3 sentencepiece==0.2.0 ultralytics==8.3.148 paddlepaddle==2.6.1

```

```bash
pip install --only-binary=:all: --index-url https://pypi.org/simple "PyMuPDF==1.24.10" "pdf2docx==0.5.8"
```

```bash
pip install shapely==2.1.1 imgaug==0.4.0 scikit-image==0.25.2 lmdb==1.7.3 pyclipper==1.3.0.post6 protobuf==3.20.3 visualdl==2.5.3 bce-python-sdk==0.9.46

```
```bash
pip install --no-deps paddleocr==2.6.1.3
```
```bash
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0

```

```bash
pip install --force-reinstall numpy==1.26.4 imgaug==0.4.0
```

### Step 3. Run the Detection Script:

```bash
python3 scripts/detect.py
```

#### Important Parameters for `detect.py`

- **`IMAGE_PATH`**  
  - Path to the image that will be processed for detection.

- **`assign_privacy_scores` function**  
  - Accepts a boolean parameter (`True` or `False`).  
  - If set to `True`, privacy scores are assigned based on the user study.  
  - ⚠️ These scores are **not suitable for visualizations**, so using `False` is recommended for testing and debugging.

- **Consistency Across Scripts**  
  - Ideally, the same boolean values should be used in both scripts.  
  - For example:  
    - If `assign_privacy_scores(True)` is used in `detect.py`, then `is_real` in `enc-dec.py` should also be set to `True`.  
    - If `assign_privacy_scores(False)` is used, then `is_real` should also be `False`.


Exit the virtual environment.
```bash
deactivate
```


