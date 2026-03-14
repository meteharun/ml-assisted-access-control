#!/bin/bash
# =============================================================
# run.sh — Run the full PSO detection + encryption pipeline
#
# Usage:
#   ./run.sh <image_filename> [real|synthetic]
#
# Examples:
#   ./run.sh 2017_10130708.jpg
#   ./run.sh 2017_10130708.jpg real
# =============================================================

# Always run from the directory where run.sh lives
cd "$(dirname "$(readlink -f "$0")")"

# ── Arguments ─────────────────────────────────────────────────
IMAGE_FILENAME="${1}"
SCORE_MODE="${2:-synthetic}"

if [ -z "$IMAGE_FILENAME" ]; then
    echo "Usage: ./run.sh <image_filename> [real|synthetic]"
    echo ""
    echo "Examples:"
    echo "  ./run.sh 2017_10130708.jpg"
    echo "  ./run.sh 2017_10130708.jpg real"
    exit 1
fi

if [ "$SCORE_MODE" = "real" ]; then
    REAL_SCORES="true"
else
    REAL_SCORES="false"
fi

IMAGE_PATH="data/samples/${IMAGE_FILENAME}"
IMAGE_ID="${IMAGE_FILENAME%.*}"
DETECTION_FILE="data/detections/${IMAGE_ID}.txt"

# ── Cleanup stale Docker networks ─────────────────────────────
docker-compose down --remove-orphans 2>/dev/null || true

# ── Checks ────────────────────────────────────────────────────
if [ ! -f "$IMAGE_PATH" ]; then
    echo "Error: image not found at ${IMAGE_PATH}"
    echo "Make sure the file is inside the data/samples/ folder."
    exit 1
fi

echo "============================================"
echo "  Image     : $IMAGE_FILENAME"
echo "  Scores    : $SCORE_MODE"
echo "============================================"

# ── Stage 1: Detection (skip if .txt already exists) ──────────
if [ -f "$DETECTION_FILE" ]; then
    echo ""
    echo "[ Stage 1 / 2 ] Detection file already exists, skipping."
    echo "  → ${DETECTION_FILE}"
else
    echo ""
    echo "[ Stage 1 / 2 ] Running detection..."
    echo ""

    IMAGE="$IMAGE_FILENAME" USE_REAL_SCORES="$REAL_SCORES" docker-compose up detect

    if [ ! -f "$DETECTION_FILE" ]; then
        echo ""
        echo "============================================"
        echo "  ERROR: Detection failed."
        echo "  No detections file found at:"
        echo "  ${DETECTION_FILE}"
        echo ""
        echo "  Model weights are required to run detection."
        echo "  See README.md -> Prerequisites -> Model weights"
        echo "============================================"
        exit 1
    fi

    echo ""
    echo "[ Stage 1 / 2 ] Detection complete."
fi

# ── Stage 2: Encryption + Decryption ─────────────────────────
echo ""
echo "[ Stage 2 / 2 ] Running encryption..."
echo ""

docker run -it --rm \
    -v "$(pwd)/data:/app/data" \
    -v "$(pwd)/outputs:/app/outputs" \
    pesto-pacman-abe \
    python src/enc_dec.py \
        --image "/app/data/samples/${IMAGE_FILENAME}" \
        --detections /app/data/detections \
        --output-dir /app/outputs \
        --real-scores "$REAL_SCORES"

echo ""
echo "============================================"
echo "  Done. Output images:"
echo "  outputs/${IMAGE_ID}_encrypted.png"
echo "  outputs/${IMAGE_ID}_decrypted_abekey<N>.png"
echo "============================================"