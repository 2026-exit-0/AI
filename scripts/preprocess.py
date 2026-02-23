import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
from torchvision import transforms as T
from torchvision.utils import save_image

# ===============================
# 1. 경로 설정
# ===============================
INPUT_DIR = "./data/cropped_img"   # 256x256 bbox 결과
OUTPUT_DIR = "./data/final_processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PATCH_SIZE = 224

# ===============================
# 2. 모델 입력 전처리
# ===============================
model_preprocess = T.Compose([
    T.Resize((PATCH_SIZE, PATCH_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# ===============================
# 3. CLAHE (질감 유지 목적)
# ===============================
def apply_mild_clahe(img_rgb):
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

# ===============================
# 4. 메인 루프
# ===============================
# 하위 폴더까지 재귀 순회
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

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # CLAHE
        img_processed = apply_mild_clahe(img_rgb)

        # 밝기 극단값 필터링 (Normalize 이전)
        tensor_check = T.ToTensor()(Image.fromarray(img_processed))
        mean_val = tensor_check.mean().item()

        if mean_val < 0.1 or mean_val > 0.9:
            continue

        # 모델 입력 변환
        final_tensor = model_preprocess(Image.fromarray(img_processed))

        save_path = os.path.join(save_dir, file)
        save_image(final_tensor, save_path)