import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
from torchvision import transforms as T

# ===============================
# 1. 경로 설정
# ===============================
INPUT_DIR = "./data/cropped_img"
OUTPUT_DIR = "./data/final_processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PATCH_SIZE = 224

# ===============================
# 2. ResNet용 Normalize (학습 시 사용할 것)
# ===============================
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]

model_preprocess = T.Compose([
    T.Resize((PATCH_SIZE, PATCH_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=imagenet_mean, std=imagenet_std)
])

# ===============================
# 3. CLAHE (L 채널만)c
# ===============================
def apply_mild_clahe(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    l = clahe.apply(l)

    merged = cv2.merge((l, a, b))
    img_bgr = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    blurred = cv2.GaussianBlur(img_bgr, (0, 0), 1.0)
    img_bgr = cv2.addWeighted(img_bgr, 1.2, blurred, -0.2, 0)

    return img_bgr

# ===============================
# 4. Scanner 스타일 통합 함수
# ===============================
def scanner_style_preprocess(img_rgb):
    img_rgb = apply_mild_clahe(img_rgb)
    img_rgb = cv2.bilateralFilter(img_rgb, 3, 30, 30)
    return img_rgb

# ===============================
# 5. 메인 루프
# ===============================
for root, dirs, files in os.walk(INPUT_DIR):

    for file in tqdm(files):

        if not file.lower().endswith((".jpg", ".png")):
            continue

        input_path = os.path.join(root, file)

        relative_path = os.path.relpath(root, INPUT_DIR)
        save_dir = os.path.join(OUTPUT_DIR, relative_path)
        os.makedirs(save_dir, exist_ok=True)

        img = cv2.imread(input_path)
        if img is None:
            continue

        img_processed = scanner_style_preprocess(img)  # BGR 그대로 전달

        # 🔹 밝기 극단값 필터링
        tensor_check = T.ToTensor()(Image.fromarray(img_processed))
        mean_val = tensor_check.mean().item()

        if mean_val < 0.1 or mean_val > 0.9:
            continue

        # 🔹 224 리사이즈 후 저장 (Normalize 적용하지 않음)
        img_resized = cv2.resize(img_processed, (PATCH_SIZE, PATCH_SIZE))

        save_path = os.path.join(save_dir, file)
        cv2.imwrite(save_path, img_resized)
