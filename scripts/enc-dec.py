import base64
import os
import time
from PIL import Image
import numpy as np
from cryptography.fernet import Fernet
from collections import deque
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from abe import ABE
import re

is_real = False
SCORE_LEVELS = [0.25, 0.45, 0.75, 1.0 ] # BETTER FOR VISUALIZATIONS
REAL_SCORE_LEVELS = [0.35, 0.7, 0.9, 1.0]

BASE_DIR = os.path.dirname(os.path.dirname(__file__))  
IMAGE_PATH = os.path.join(BASE_DIR, "data", "samples","2017_13079119.jpg") #################### Input Image
image_id = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
DET_PATH = os.path.join(BASE_DIR, "data", "detections", f"{image_id}.txt") #################### Input Image's Detections

def to_native_path(p: str) -> str:
    if os.name == "posix" and re.match(r"^[A-Za-z]:[\\/]", p):
        drive = p[0].lower()
        rest = p[2:].lstrip("\\/").replace("\\", "/")
        return f"/mnt/{drive}/{rest}"
    return p


class PESTOPACMAN:
    def __init__(self, privacy_levels=4, image_path="detections5.jpg", coordinates_file="detections5.txt"):
        self.abe = ABE()
        self.privacy_levels = privacy_levels
        self.keys = []
        self.encrypted_keys = []
        self.salts = []
        self.create_image_encryption_keys()

        self.image = Image.open(image_path).convert("RGB")
        self.image_array = np.array(self.image)
        self.coordinates_file = coordinates_file
        self.data_queue = deque()

    def create_key_from_pairing_group_element(self, element, salt):
        serialized_element = self.abe.pairing_group.serialize(element)
        kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=480000)
        return base64.urlsafe_b64encode(kdf.derive(serialized_element))

    def create_image_encryption_keys(self):
        for i in range(self.privacy_levels):
            g = self.abe.get_random_plaintext()
            policy_string = " or ".join([f"SCORE{j+1}" for j in range(i, self.privacy_levels)])
            encrypted_key = self.abe.encrypt(g, policy_string)
            salt = os.urandom(16)
            key = self.create_key_from_pairing_group_element(g, salt)

            self.encrypted_keys.append(encrypted_key)
            self.salts.append(salt)
            self.keys.append(key)

    def get_image_coordinates(self):
        segments_info = []
        with open(self.coordinates_file, "r") as f:
            for line in f:
                try:
                    main_parts_str, segment_type = line.strip().rsplit(", ", 1)
                    label, _, privacy_score_str, coords_str = main_parts_str.split(", ", 3)
                    privacy_score = float(privacy_score_str)
                    coords = eval(coords_str)
                    segments_info.append((label, coords, privacy_score, segment_type))
                except Exception as e:
                    print(f"Error parsing line: '{line.strip()}'. Error: {e}")
        return segments_info

    def encrypt_image(self):
        encryption_start = time.time()
        score_levels = SCORE_LEVELS
        if is_real == True:
            score_levels = REAL_SCORE_LEVELS
        all_segments = []

        for label, coords, score, seg_type in self.get_image_coordinates():
            for i, threshold in enumerate(score_levels):
                if score <= threshold:
                    level = i
                    break
            else:
                level = len(score_levels) - 1

            all_segments.append((level, coords, self.keys[level], seg_type, label))

        all_segments.sort(reverse=True, key=lambda x: x[0])

        for level, coords, key, seg_type, label in all_segments:
            result = self.encrypt_image_segment(coords, key, seg_type)
            if result:
                if seg_type == "visual":
                    ciphertext, shape, valid_coords = result
                    self.data_queue.append((label, valid_coords, level, ciphertext, seg_type, shape))
                else:
                    ciphertext, shape = result
                    self.data_queue.append((label, coords, level, ciphertext, seg_type, shape))
                print(f"[ENCRYPT] score level={level}, type={seg_type}, label={label}")

        encrypted_image = Image.fromarray(self.image_array)
        return encrypted_image, time.time() - encryption_start


    def encrypt_image_segment(self, coords, key, segment_type):
        f = Fernet(key)

        if segment_type in ["textual", "multimodal"]:
            xs = [pt[0] for pt in coords]
            ys = [pt[1] for pt in coords]
            if not xs or not ys:
                return None
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            x_start, y_start = max(0, x_min), max(0, y_min)
            x_end, y_end = min(self.image_array.shape[1], x_max), min(self.image_array.shape[0], y_max)
            sub_img = self.image_array[y_start:y_end, x_start:x_end].copy()
            shape = sub_img.shape
            ciphertext = f.encrypt(sub_img.tobytes())
            scrambled = np.frombuffer(ciphertext, dtype=np.uint8)
            needed = np.prod(shape)
            if scrambled.size < needed:
                scrambled = np.pad(scrambled, (0, needed - scrambled.size), 'constant')
            else:
                scrambled = scrambled[:needed]
            self.image_array[y_start:y_end, x_start:x_end] = scrambled.reshape(shape)
            return ciphertext, shape

        elif segment_type == "visual":
            if not hasattr(self, "encrypted_mask"):
                self.encrypted_mask = np.zeros(self.image_array.shape[:2], dtype=bool)
            original_pixels = []
            valid_coords = []
            for y, x in coords:
                if 0 <= y < self.image_array.shape[0] and 0 <= x < self.image_array.shape[1]:
                    if not self.encrypted_mask[y, x]:
                        original_pixels.append(self.image_array[y, x].tolist())
                        valid_coords.append((y, x))
                        self.encrypted_mask[y, x] = True
            if not original_pixels:
                return None
            array = np.array(original_pixels, dtype=np.uint8)
            shape = array.shape
            ciphertext = f.encrypt(array.tobytes())
            scrambled = np.frombuffer(ciphertext, dtype=np.uint8)
            needed = np.prod(shape)
            if scrambled.size < needed:
                scrambled = np.pad(scrambled, (0, needed - scrambled.size), 'constant')
            else:
                scrambled = scrambled[:needed]
            scrambled = scrambled.reshape(shape)
            for i, (y, x) in enumerate(valid_coords):
                self.image_array[y, x] = scrambled[i]
            return ciphertext, shape, valid_coords




    def decrypt_image(self, user_level_idx):
        start = time.time()
        img_arr = self.image_array.copy()
        queue_copy = self.data_queue.copy()

        while queue_copy:
            label, coords, segment_level, ciphertext, seg_type, shape = queue_copy.pop()

            #print(f"[DEBUG] Segment: '{label}' | Type: {seg_type} | Level: {segment_level} | User Level: {user_level_idx}")

            if segment_level <= user_level_idx:
                f = Fernet(self.keys[segment_level])
                try:
                    decrypted_bytes = f.decrypt(ciphertext)
                    print(f"[SUCCESS] Decrypted segment: '{label}' at level {segment_level}")
                except Exception as e:
                    #print(f"[ERROR] Failed to decrypt segment '{label}' at level {segment_level}: {e}")
                    continue

                decrypted_arr = np.frombuffer(decrypted_bytes, dtype=np.uint8).reshape(shape)

                if seg_type in ["textual", "multimodal"]:
                    xs = [pt[0] for pt in coords]
                    ys = [pt[1] for pt in coords]
                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)
                    x_start, y_start = max(0, x_min), max(0, y_min)
                    x_end, y_end = min(img_arr.shape[1], x_max), min(img_arr.shape[0], y_max)
                    img_arr[y_start:y_end, x_start:x_end] = decrypted_arr

                elif seg_type == "visual":
                    for i, (y, x) in enumerate(coords): 
                        if i < len(decrypted_arr) and 0 <= y < img_arr.shape[0] and 0 <= x < img_arr.shape[1]:
                            img_arr[y, x] = decrypted_arr[i]
            else:
                print(f"[SKIPPED] Segment: '{label}' at level {segment_level} exceeds user level {user_level_idx}")

        return Image.fromarray(img_arr), time.time() - start



def get_user_input(abe_keys):
    key_options = ", ".join([f"abekey{i+1}" for i in range(len(abe_keys))])
    user_input = input(f"Which key do you have ({key_options}): ")
    try:
        idx = int(user_input.replace("abekey", "")) - 1
        if 0 <= idx < len(abe_keys):
            return abe_keys[idx], user_input
    except:
        pass
    raise Exception(f"Invalid input: '{user_input}'")


def main():
    IMAGE_DIR = to_native_path(IMAGE_PATH)
    DET_DIR   = to_native_path(DET_PATH)

    pestopacman = PESTOPACMAN(image_path=IMAGE_DIR, coordinates_file=DET_DIR)
    abe_keys = [pestopacman.abe.generate_key([f"SCORE{i+1}"]) for i in range(pestopacman.privacy_levels)]

    encrypted_image, enc_time = pestopacman.encrypt_image()
    print(f"Image encrypted in {enc_time:.2f} seconds")

    os.makedirs("outputs", exist_ok=True)
    encrypted_path = os.path.join("outputs", "encrypted_image.png")
    encrypted_image.save(encrypted_path)
    print(f"Encrypted image saved to: {encrypted_path}")

    try:
        user_abe_key, label = get_user_input(abe_keys)
    except Exception as e:
        print(e)
        return

    # Determine the **highest** accessible level 
    user_level_idx = -1
    for i in range(pestopacman.privacy_levels - 1, -1, -1):
        g = pestopacman.abe.decrypt(user_abe_key, pestopacman.encrypted_keys[i])
        if g:
            user_level_idx = i
            break

    if user_level_idx == -1:
        print("You do not have access to any decryption level.")
        return

    #print(f"[INFO] User level determined as: {user_level_idx}")
    decrypted_image, dec_time = pestopacman.decrypt_image(user_level_idx)
    print(f"Image decrypted in {dec_time:.2f} seconds")

    decrypted_path = os.path.join("outputs", f"decrypted_image_{label}.png")
    decrypted_image.save(decrypted_path)
    print(f"Decrypted image saved to: {decrypted_path}")




if __name__ == "__main__":
    main()