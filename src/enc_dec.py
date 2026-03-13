"""
enc_dec.py — Encryption / Decryption Pipeline
==============================================
Reads an image and its corresponding detection JSON produced by detect.py,
encrypts privacy-sensitive regions with a hybrid ABE + AES scheme, and
optionally decrypts them based on a user-supplied ABE key level.

Usage:
    python src/enc_dec.py --image data/samples/img.jpg
    python src/enc_dec.py --image data/samples/img.jpg \\
                          --detections data/detections \\
                          --output-dir outputs \\
                          --real-scores false
"""

import base64
import os
import re
import json
import time
import argparse
from collections import deque

import numpy as np
from PIL import Image
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from abe import ABE

# ── Score-level thresholds ────────────────────────────────────────────────────
# These map a continuous privacy score → one of 4 discrete sensitivity groups.
# SYNTHETIC_LEVELS are better for visualisation; REAL_LEVELS match the paper.

SYNTHETIC_LEVELS = [0.25, 0.45, 0.75, 1.0]
REAL_LEVELS      = [0.35, 0.70, 0.90, 1.0]


# ── Path helper (WSL / Windows dual-boot support) ─────────────────────────────

def to_native_path(p: str) -> str:
    """Convert a Windows-style path to a Linux/WSL mount path if needed."""
    if os.name == "posix" and re.match(r"^[A-Za-z]:[\\/]", p):
        drive = p[0].lower()
        rest  = p[2:].lstrip("\\/").replace("\\", "/")
        return f"/mnt/{drive}/{rest}"
    return p


# ── Core class ────────────────────────────────────────────────────────────────

class PESTOPACMAN:
    """
    Orchestrates the hybrid ABE + symmetric encryption pipeline.

    Responsibilities (mirrors the paper's CryptoCore + AccessPolicy modules):
      - Key generation: one symmetric Fernet key per sensitivity group,
        each wrapped under an ABE policy.
      - Encryption: segments are encrypted in descending sensitivity order
        to avoid redundant re-encryption of overlapping regions.
      - Decryption: user supplies their ABE key level; only segments at or
        below that level are decrypted.
    """

    def __init__(
        self,
        image_path: str,
        detections_path: str,
        privacy_levels: int = 4,
    ):
        self.abe             = ABE()
        self.privacy_levels  = privacy_levels
        self.keys            = []       # Fernet keys (symmetric)
        self.encrypted_keys  = []       # ABE ciphertexts wrapping each Fernet key
        self.salts           = []
        self.data_queue      = deque()  # populated during encrypt_image()

        self._setup_keys()

        self.image          = Image.open(image_path).convert("RGB")
        self.image_array    = np.array(self.image)
        self.detections_path = detections_path

    # ── Key setup ─────────────────────────────────────────────────────────────

    def _derive_fernet_key(self, pairing_element, salt: bytes) -> bytes:
        """Derive a Fernet key from a pairing group element via PBKDF2."""
        serialized = self.abe.pairing_group.serialize(pairing_element)
        kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=480000)
        return base64.urlsafe_b64encode(kdf.derive(serialized))

    def _setup_keys(self) -> None:
        """
        Generate one symmetric key per sensitivity group and wrap each under
        an ABE policy.  Policy for group i: SCORE{i} OR SCORE{i+1} OR ... OR SCORE{n}
        so that higher-privileged users can always decrypt lower-sensitivity content.
        """
        for i in range(self.privacy_levels):
            g              = self.abe.get_random_plaintext()
            policy_string  = " or ".join([f"SCORE{j+1}" for j in range(i, self.privacy_levels)])
            encrypted_key  = self.abe.encrypt(g, policy_string)
            salt           = os.urandom(16)
            fernet_key     = self._derive_fernet_key(g, salt)

            self.encrypted_keys.append(encrypted_key)
            self.salts.append(salt)
            self.keys.append(fernet_key)

    # ── Metadata loading ──────────────────────────────────────────────────────

    def load_detections(self) -> list:
        """
        Load detections from JSON (written by detect.py).
        Returns a list of dicts with keys: label, type, privacy_score, box/pixels.
        """
        with open(self.detections_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # ── Encryption ────────────────────────────────────────────────────────────

    def encrypt_image(self, score_levels: list) -> tuple:
        """
        Encrypt all detected PSO regions and return (encrypted_image, elapsed_seconds).
        Segments are processed in descending sensitivity order so that a pixel
        already encrypted at level N is not re-encrypted at level M < N.
        """
        start       = time.time()
        detections  = self.load_detections()
        all_segments = []

        for det in detections:
            score    = det.get("privacy_score", 0.0)
            seg_type = det.get("type", "unknown")
            label    = det.get("label", "unknown")
            coords   = det.get("pixels") if seg_type == "visual" else det.get("box")

            # Map continuous score → discrete level index
            level = len(score_levels) - 1
            for i, threshold in enumerate(score_levels):
                if score <= threshold:
                    level = i
                    break

            all_segments.append((level, coords, self.keys[level], seg_type, label))

        # Highest sensitivity first to avoid double-encryption of overlapping regions
        all_segments.sort(reverse=True, key=lambda x: x[0])

        for level, coords, key, seg_type, label in all_segments:
            result = self._encrypt_segment(coords, key, seg_type)
            if result is None:
                continue

            if seg_type == "visual":
                ciphertext, shape, valid_coords = result
                self.data_queue.append((label, valid_coords, level, ciphertext, seg_type, shape))
            else:
                ciphertext, shape = result
                self.data_queue.append((label, coords, level, ciphertext, seg_type, shape))

            print(f"  [ENCRYPT] level={level}, type={seg_type}, label={label}")

        return Image.fromarray(self.image_array), time.time() - start

    def _encrypt_segment(self, coords, key: bytes, segment_type: str):
        """Encrypt a single region and paint the scrambled ciphertext over it."""
        f = Fernet(key)

        if segment_type in ("textual", "multimodal"):
            xs = [pt[0] for pt in coords]
            ys = [pt[1] for pt in coords]
            if not xs or not ys:
                return None
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            x0 = max(0, x_min);  y0 = max(0, y_min)
            x1 = min(self.image_array.shape[1], x_max)
            y1 = min(self.image_array.shape[0], y_max)

            sub_img    = self.image_array[y0:y1, x0:x1].copy()
            shape      = sub_img.shape
            ciphertext = f.encrypt(sub_img.tobytes())
            scrambled  = self._fit_to_shape(np.frombuffer(ciphertext, dtype=np.uint8), shape)
            self.image_array[y0:y1, x0:x1] = scrambled
            return ciphertext, shape

        elif segment_type == "visual":
            if not hasattr(self, "_encrypted_mask"):
                self._encrypted_mask = np.zeros(self.image_array.shape[:2], dtype=bool)

            original_pixels, valid_coords = [], []
            for y, x in coords:
                if (0 <= y < self.image_array.shape[0]
                        and 0 <= x < self.image_array.shape[1]
                        and not self._encrypted_mask[y, x]):
                    original_pixels.append(self.image_array[y, x].tolist())
                    valid_coords.append((y, x))
                    self._encrypted_mask[y, x] = True

            if not original_pixels:
                return None

            array      = np.array(original_pixels, dtype=np.uint8)
            shape      = array.shape
            ciphertext = f.encrypt(array.tobytes())
            scrambled  = self._fit_to_shape(np.frombuffer(ciphertext, dtype=np.uint8), shape)
            for i, (y, x) in enumerate(valid_coords):
                self.image_array[y, x] = scrambled[i]

            return ciphertext, shape, valid_coords

        return None

    @staticmethod
    def _fit_to_shape(data: np.ndarray, shape: tuple) -> np.ndarray:
        """Pad or truncate byte array to match target shape, then reshape."""
        needed = int(np.prod(shape))
        if data.size < needed:
            data = np.pad(data, (0, needed - data.size), "constant")
        else:
            data = data[:needed]
        return data.reshape(shape)

    # ── Decryption ────────────────────────────────────────────────────────────

    def decrypt_image(self, user_level_idx: int) -> tuple:
        """
        Decrypt all segments the user has access to (level <= user_level_idx).
        Returns (decrypted_image, elapsed_seconds).
        """
        start      = time.time()
        img_arr    = self.image_array.copy()
        queue_copy = self.data_queue.copy()

        while queue_copy:
            label, coords, seg_level, ciphertext, seg_type, shape = queue_copy.pop()

            if seg_level > user_level_idx:
                print(f"  [SKIPPED]  '{label}' level={seg_level} > user_level={user_level_idx}")
                continue

            f = Fernet(self.keys[seg_level])
            try:
                decrypted_bytes = f.decrypt(ciphertext)
            except Exception as e:
                print(f"  [ERROR] Could not decrypt '{label}': {e}")
                continue

            decrypted_arr = np.frombuffer(decrypted_bytes, dtype=np.uint8).reshape(shape)
            print(f"  [SUCCESS]  Decrypted '{label}' at level {seg_level}")

            if seg_type in ("textual", "multimodal"):
                xs = [pt[0] for pt in coords]
                ys = [pt[1] for pt in coords]
                x0 = max(0, min(xs));  y0 = max(0, min(ys))
                x1 = min(img_arr.shape[1], max(xs))
                y1 = min(img_arr.shape[0], max(ys))
                img_arr[y0:y1, x0:x1] = decrypted_arr

            elif seg_type == "visual":
                for i, (y, x) in enumerate(coords):
                    if i < len(decrypted_arr) and 0 <= y < img_arr.shape[0] and 0 <= x < img_arr.shape[1]:
                        img_arr[y, x] = decrypted_arr[i]

        return Image.fromarray(img_arr), time.time() - start

    # ── ABE key resolution ────────────────────────────────────────────────────

    def resolve_user_level(self, user_abe_key) -> int:
        """
        Try to decrypt each wrapped symmetric key starting from the most
        sensitive group.  The first success determines the user's maximum level.
        Returns -1 if the key satisfies no policy.
        """
        for i in range(self.privacy_levels - 1, -1, -1):
            result = self.abe.decrypt(user_abe_key, self.encrypted_keys[i])
            if result:
                return i
        return -1


# ── CLI helpers ───────────────────────────────────────────────────────────────

def get_user_input(abe_keys: list) -> tuple:
    options   = ", ".join([f"abekey{i+1}" for i in range(len(abe_keys))])
    raw_input = input(f"Which key do you have? ({options}): ").strip()
    try:
        idx = int(raw_input.replace("abekey", "")) - 1
        if 0 <= idx < len(abe_keys):
            return abe_keys[idx], raw_input
    except ValueError:
        pass
    raise ValueError(f"Invalid input: '{raw_input}'")


def parse_args():
    parser = argparse.ArgumentParser(description="PSO Encryption/Decryption Pipeline")
    parser.add_argument("--image",        required=True,  help="Path to input image")
    parser.add_argument("--detections",   default=None,   help="Directory containing detection JSON files")
    parser.add_argument("--output-dir",   default="outputs", help="Directory for encrypted/decrypted images")
    parser.add_argument("--real-scores",  default="false",   help="Use real user-study score thresholds (true/false)")
    parser.add_argument("--privacy-levels", type=int, default=4, help="Number of sensitivity groups (default: 4)")
    return parser.parse_args()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    image_path   = to_native_path(args.image)
    use_real     = args.real_scores.lower() == "true"
    score_levels = REAL_LEVELS if use_real else SYNTHETIC_LEVELS
    output_dir   = args.output_dir

    image_id = os.path.splitext(os.path.basename(image_path))[0]

    det_dir          = args.detections or os.path.join(os.path.dirname(os.path.dirname(image_path)), "detections")
    detections_path  = to_native_path(os.path.join(det_dir, f"{image_id}.json"))

    print(f"[CONFIG] image={image_path}, real_scores={use_real}, detections={detections_path}")

    pestopacman = PESTOPACMAN(
        image_path=image_path,
        detections_path=detections_path,
        privacy_levels=args.privacy_levels,
    )

    # Generate one ABE key per sensitivity level (simulates different users)
    abe_keys = [pestopacman.abe.generate_key([f"SCORE{i+1}"]) for i in range(pestopacman.privacy_levels)]

    # Encrypt
    encrypted_image, enc_time = pestopacman.encrypt_image(score_levels)
    print(f"\nImage encrypted in {enc_time:.2f}s")

    os.makedirs(output_dir, exist_ok=True)
    encrypted_path = os.path.join(output_dir, f"{image_id}_encrypted.png")
    encrypted_image.save(encrypted_path)
    print(f"Encrypted image → {encrypted_path}")

    # Ask which key the user has
    try:
        user_abe_key, label = get_user_input(abe_keys)
    except ValueError as e:
        print(e)
        return

    user_level_idx = pestopacman.resolve_user_level(user_abe_key)
    if user_level_idx == -1:
        print("You do not have access to any decryption level.")
        return

    print(f"\n[INFO] Resolved user level: {user_level_idx}")

    # Decrypt
    decrypted_image, dec_time = pestopacman.decrypt_image(user_level_idx)
    print(f"Image decrypted in {dec_time:.2f}s")

    decrypted_path = os.path.join(output_dir, f"{image_id}_decrypted_{label}.png")
    decrypted_image.save(decrypted_path)
    print(f"Decrypted image → {decrypted_path}")


if __name__ == "__main__":
    main()
