import re
import os
import cv2
import torch
import numpy as np
from paddleocr import PaddleOCR
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification
from ultralytics import YOLO
import torch.nn.functional as F
from difflib import SequenceMatcher
# === Configuration ===
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # one level up from scripts/

IMAGE_PATH = os.path.join(BASE_DIR, "data", "samples","2017_13079119.jpg")
image_id = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "detections", f"{image_id}.txt")
YOLO_MULTIMODAL_PATH = os.path.join(BASE_DIR, "models", "multimodal", "yolov8l_multimodal", "weights", "best.pt")
YOLO_VISUAL_PATH = os.path.join(BASE_DIR, "models", "visual", "yolo_only_visuals_augmented_100", "weights", "best.pt")
DEBERTA_MULTIMODAL_PATH = os.path.join(BASE_DIR, "models", "multimodal", "deberta_multimodal")
DEBERTA_TEXTUAL_PATH = os.path.join(BASE_DIR, "models", "textual", "deberta")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# === Label Mappings ===
MULTIMODAL_LABELS = {
    0: "credit_card", 1: "passport", 2: "drivers_license", 3: "student_id",
    4: "mail", 5: "receipt", 6: "ticket"
}
VISUAL_LABELS = {
    0: "face", 1: "license_plate", 2: "person", 3: "nudity",
    4: "handwriting", 5: "disability", 6: "medicine", 7: "fingerprint",
    8: "signature"
}
TEXTUAL_LABELS = {
    0: "name", 1: "phone", 2: "date-time", 3: "email", 4: "location", 5: "safe"
}

KNOWN_PREFIXES = [
    "expires", "expires on", "dob", "date of birth", "date of issue",
    "issued", "valid until", "birthdate", "exp"
]

BIRTH_CUES = ["dob", "date of birth", "birthdate", "birth date", "birth", "born", "Date of Birth"]
NAME_CUES = ["name", "surname", "first name", "last name", "given name", "family name"]

detections = []

def assign_privacy_scores(is_real):
    privacy_scores = {
        "face": 0.5,
        "location": 0.8,
        "license_plate": 0.6,
        "person": 0.35,
        "nudity": 0.9,
        "name": 0.7,
        "birth_date": 0.7,
        "handwriting": 0.4,
        "credit_card": 0.6,
        "passport": 0.3,
        "drivers_license": 0.3,
        "student_id": 0.3,
        "mail": 0.5,
        "receipt": 0.4,
        "ticket": 0.4,
        "disability": 0.5,
        "medicine": 0.7,
        "phone": 0.6,
        "landmark": 0.2,
        "fingerprint": 0.8,
        "date-time": 0.5,
        "username": 0.6,
        "signature": 0.8,
        "email": 0.6        
    }

    real_privacy_scores = {
        "face": 0.38,
        "location": 0.1,
        "license_plate": 0.34,
        "person": 0.3,
        "nudity": 0.85,
        "name": 0.92,
        "birth_date": 0.8,
        "handwriting": 0.85,
        "credit_card": 0.98,
        "passport": 0.98,
        "drivers_license": 1.0,
        "student_id": 0.99,
        "mail": 0.7,
        "receipt": 0.8,
        "ticket": 0.9,
        "disability": 0.5,
        "medicine": 0.87,
        "phone": 0.6,
        "landmark": 0.1,
        "fingerprint": 0.53,
        "date-time": 0.47,
        "username": 0.59,
        "signature": 0.95,
        "email": 0.74
        
    }
    for det in detections:
        label = det.get("label")
        if is_real:
            det["privacy_score"] = real_privacy_scores.get(label, 0.0) 
        else: 
            det["privacy_score"] = privacy_scores.get(label, 0.0) 

def normalize(text):
    return re.sub(r"[^a-zA-Z]", "", text.lower())


def is_fuzzy_match(text, keywords, threshold=0.8):
    parts = re.split(r"[/():\-]", text.lower())
    parts = [normalize(p) for p in parts if p.strip()]
    for part in parts:
        for keyword in keywords:
            if SequenceMatcher(None, part, normalize(keyword)).ratio() >= threshold:
                return True
    return False


def get_center(box):
    return np.mean(box, axis=0)


# === Utilities ===
def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1b, y1b, x2b, y2b = box2
    xi1, yi1 = max(x1, x1b), max(y1, y1b)
    xi2, yi2 = min(x2, x2b), min(y2, y2b)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union = (x2 - x1)*(y2 - y1) + (x2b - x1b)*(y2b - y1b) - inter
    return inter / union if union > 0 else 0


def save_detections_to_txt(detections, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for det in detections:
            label = det.get("label", "unknown")
            confidence = det.get("confidence", 0.0)
            privacy = det.get("privacy_score", 0.0)
            det_type = det.get("type", "unknown")

            if "pixels" in det:
                region = det["pixels"]
            elif "box" in det:
                region = det["box"]
            else:
                region = []

            line = f"{label}, {confidence}, {privacy}, {region}, {det_type}\n"
            f.write(line)


def smart_split(text):
    text_lower = text.lower()

    for phrase in KNOWN_PREFIXES:
        # Match known phrase followed by optional separator(s) and then value
        pattern = rf"({re.escape(phrase)})[:.\s\-_]*([a-z0-9]+.*)"
        match = re.search(pattern, text_lower)

        if match:
            label_raw = match.group(1)
            value_raw = match.group(2)

            label_part = text[:match.start(2)].strip(": .-_").strip()
            value_part = text[match.start(2):].strip(": .-_").strip()

            if value_part:
                return [label_part, value_part]

    return [text]


# === Model Loading ===
def load_models():
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    tokenizer_textual = DebertaV2Tokenizer.from_pretrained(DEBERTA_TEXTUAL_PATH)
    model_textual = DebertaV2ForSequenceClassification.from_pretrained(DEBERTA_TEXTUAL_PATH).to(DEVICE).eval()
    tokenizer_multimodal = DebertaV2Tokenizer.from_pretrained(DEBERTA_MULTIMODAL_PATH)
    model_multimodal = DebertaV2ForSequenceClassification.from_pretrained(DEBERTA_MULTIMODAL_PATH).to(DEVICE).eval()
    yolo_multimodal = YOLO(YOLO_MULTIMODAL_PATH)
    yolo_visual = YOLO(YOLO_VISUAL_PATH)
    return ocr, tokenizer_textual, model_textual, tokenizer_multimodal, model_multimodal, yolo_multimodal, yolo_visual


# === Textual Classification ===
def run_textual_classification(ocr, tokenizer_textual, model_textual):
    print("\n[OCR + TEXTUAL CLASSIFICATION]")
    ocr_results = ocr.ocr(IMAGE_PATH, cls=True)
    texts, text_boxes = [], []
    for line in ocr_results[0]:
        texts.append(line[1][0])
        text_boxes.append(np.array(line[0], dtype=np.int32))

    split_texts, split_boxes = [], []
    for text, box in zip(texts, text_boxes):
        parts = smart_split(text)
        total_len = sum(len(p) for p in parts)
        if total_len == 0:
            continue
        x_coords = [pt[0] for pt in box]
        y_coords = [pt[1] for pt in box]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        x_start = x_min
        for part in parts:
            rel_width = len(part) / total_len
            x_end = x_start + int(rel_width * (x_max - x_min))
            sub_box = np.array([
                [x_start, y_min], [x_end, y_min], [x_end, y_max], [x_start, y_max]
            ], dtype=np.int32)
            split_texts.append(part)
            split_boxes.append(sub_box)
            x_start = x_end

    predicted = []
    if split_texts:
        inputs = tokenizer_textual(split_texts, return_tensors="pt", truncation=True, padding=True).to(DEVICE)
        with torch.no_grad():
            logits = model_textual(**inputs).logits
            probs = F.softmax(logits, dim=1)
            top2 = torch.topk(probs, k=2, dim=1)

        for part, box, p_idx, p_probs in zip(split_texts, split_boxes, top2.indices, top2.values):
            label = TEXTUAL_LABELS.get(p_idx[0].item(), "unknown")
            confidence = p_probs[0].item()
            if label == "safe" and confidence < 0.6:
                label = TEXTUAL_LABELS.get(p_idx[1].item(), "unknown")
                confidence = p_probs[1].item()

            predicted.append({
                "text": part,
                "box": box,
                "label": label,
                "conf": confidence,
                "center": get_center(box),
                "corrected": False
            })

            print(f"BERT: '{part}' → {label} ({confidence:.2f})")

    def correct_next(cue_keywords, target_label, candidate_predicates, new_label):
        for i, cue_pred in enumerate(predicted):
            if not is_fuzzy_match(cue_pred["text"], cue_keywords):
                continue
            if i + 1 >= len(predicted):
                continue

            next_pred = predicted[i + 1]
            if candidate_predicates(next_pred):
                predicted[i + 1]["label"] = new_label
                predicted[i + 1]["corrected"] = True
                print(f"[AUTO-CORRECT] Closest to {target_label} cue '{cue_pred['text']}' → '{next_pred['text']}' → {new_label}")

    # Apply cue-based corrections
    correct_next(BIRTH_CUES, "birth", lambda p: p["label"] == "date-time", "birth_date")
    correct_next(NAME_CUES, "name", lambda p: True, "name")

    # Draw and collect final predictions
    for item in predicted:
        if item["label"] == "safe":
            continue

        detections.append({
            "text": item["text"],
            "type": "textual",
            "label": item["label"],
            "confidence": round(item["conf"], 2),
            "box": item["box"].tolist()
        })

    return texts, text_boxes


# === Multimodal Classification ===
def run_multimodal_classification(yolo_model, tokenizer, deberta, texts, text_boxes):
    print("\n[MULTIMODAL CLASSIFICATION]")
    results = yolo_model(IMAGE_PATH)[0]

    raw_boxes = [(list(map(int, box.xyxy[0])), int(box.cls.item()), float(box.conf.item())) 
                 for box in results.boxes]

    merged_boxes = []
    for box, cls_id, conf in raw_boxes:
        keep = True
        for i, (existing_box, existing_cls, existing_conf) in enumerate(merged_boxes):
            if cls_id == existing_cls and compute_iou(box, existing_box) > 0.6:
                merged_boxes[i] = (
                    [min(box[0], existing_box[0]), min(box[1], existing_box[1]),
                     max(box[2], existing_box[2]), max(box[3], existing_box[3])],
                    cls_id,
                    max(conf, existing_conf)
                )
                keep = False
                break
        if keep:
            merged_boxes.append((box, cls_id, conf))

    for (x1, y1, x2, y2), yolo_class, yolo_conf in merged_boxes:
        # collect texts inside this YOLO box
        region_texts = []
        for text, box in zip(texts, text_boxes):
            box_x = box[:, 0]
            box_y = box[:, 1]
            if np.all(box_x >= x1) and np.all(box_x <= x2) and np.all(box_y >= y1) and np.all(box_y <= y2):
                region_texts.append(text)

        full_text = " ".join(region_texts).strip() or "unknown"
        inputs = tokenizer(full_text, return_tensors="pt", truncation=True, padding=True).to(DEVICE)

        with torch.no_grad():
            logits = deberta(**inputs).logits
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        # --- Hybrid correction logic ---
        if yolo_class in [2, 3]:  # drivers_license or student_id
            final_class = int(np.argmax(probs))
            final_conf = float(np.max(probs))

        elif yolo_class in [5, 6]:  # receipt or ticket
            probs_yolo = np.zeros(len(MULTIMODAL_LABELS))
            probs_yolo[yolo_class] = yolo_conf
            fused = (3 * probs + probs_yolo) / 4
            final_class = int(np.argmax(fused))
            final_conf = float(np.max(fused))

        else:  # keep YOLO
            final_class = yolo_class
            final_conf = yolo_conf

        label = MULTIMODAL_LABELS.get(final_class, "unknown")

        print(f"[PREDICTED] {label} ({final_conf:.2f}) | OCR: {full_text[:100]}{'...' if len(full_text) > 100 else ''}")

        detections.append({
            "text": full_text,
            "type": "multimodal",
            "label": label,
            "confidence": round(final_conf, 2),
            "box": [[x1, y1], [x2, y2]],
            "yolo_class": yolo_class,
            "yolo_conf": round(yolo_conf, 2)
        })


# === Visual Segmentation ===
def run_visual_segmentation(yolo_model):
    print("\n[VISUAL DETECTIONS]")
    results = yolo_model(IMAGE_PATH)[0]

    if results.masks is None:
        return

    masks = results.masks.data.cpu().numpy()
    cls_ids = results.boxes.cls.cpu().numpy().astype(int)
    orig_h, orig_w = results.masks.orig_shape

    for mask, cls_id in zip(masks, cls_ids):
        label = VISUAL_LABELS.get(cls_id, "unknown")
        resized_mask = cv2.resize(mask.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        pixel_indices = np.column_stack(np.where(resized_mask == 1))

        print(f"YOLO: {label}")
        detections.append({
            "label": label,
            "type": "visual",
            "confidence": 1.0,
            "pixels": pixel_indices.tolist(),
        })


# === Main ===
def main():
    ocr, tokenizer_textual, model_textual, tokenizer_multimodal, model_multimodal, yolo_multimodal, yolo_visual = load_models()
    texts, text_boxes = run_textual_classification(ocr, tokenizer_textual, model_textual)
    run_multimodal_classification(yolo_multimodal, tokenizer_multimodal, model_multimodal, texts, text_boxes)
    run_visual_segmentation(yolo_visual)
    assign_privacy_scores(False) ############################ True: Real privacy scores, False: synthetic privacy scores (better for visualization)
    save_detections_to_txt(detections, OUTPUT_PATH)
    print("\n------------FINAL DETECTIONS------------\n")
    for det in detections:
        print(f"Detected: {det['type']} | {det['label']} | Confidence: {det['confidence']} | Privacy: {det['privacy_score']}")


if __name__ == "__main__":
    main()
