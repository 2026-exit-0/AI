# 결과 이미지 밝기가 밝아 코드 수정 필요

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
RAW_DIR = "./data/raw_images"
SAVE_DIR = "./data/processed_images"
os.makedirs(SAVE_DIR, exist_ok=True)

# ===============================
# 2. 얼굴 검출기 (Haarcascade)
# ===============================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ===============================
# 3. ResNet 전처리
# ===============================
PATCH_SIZE = 224

preprocess = T.Compose([
    T.Resize((PATCH_SIZE, PATCH_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# ===============================
# 4. 부위 비율 좌표 (정면 기준)
# ===============================
PART_RATIOS = {
    "Forehead": (0.25, 0.5),
    "Nose": (0.55, 0.5),
    "L_Cheek": (0.60, 0.35),
    "R_Cheek": (0.60, 0.65)
}

# ===============================
# 5. CLAHE 적용 함수
# ===============================
def apply_clahe(img_rgb):
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    return enhanced

# ===============================
# 6. 부위별 crop 함수
# ===============================
def extract_parts(face_img):
    h, w, _ = face_img.shape
    crops = {}

    for part_name, (ry, rx) in PART_RATIOS.items():
        cx, cy = int(rx * w), int(ry * h)

        x1 = max(0, cx - PATCH_SIZE // 2)
        y1 = max(0, cy - PATCH_SIZE // 2)

        crop = face_img[y1:y1+PATCH_SIZE, x1:x1+PATCH_SIZE]

        if crop.shape[0] != PATCH_SIZE or crop.shape[1] != PATCH_SIZE:
            crop = cv2.resize(crop, (PATCH_SIZE, PATCH_SIZE))

        crops[part_name] = crop

    return crops

# ===============================
# 7. 메인 처리 루프
# ===============================
for img_name in tqdm(os.listdir(RAW_DIR)):
    path = os.path.join(RAW_DIR, img_name)
    img = cv2.imread(path)

    if img is None:
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        continue

    # 가장 큰 얼굴 선택
    x, y, w, h = max(faces, key=lambda b: b[2]*b[3])
    face = img[y:y+h, x:x+w]

    # BGR → RGB
    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    # CLAHE 적용
    face_enhanced = apply_clahe(face_rgb)

    # 부위별 crop
    parts = extract_parts(face_enhanced)

    base = os.path.splitext(img_name)[0]
    label = img_name.split("_")[0]

    save_dir = os.path.join(SAVE_DIR, label)
    os.makedirs(save_dir, exist_ok=True)

    for part_name, crop in parts.items():

        # 밝기 필터링
        crop_tensor = T.ToTensor()(Image.fromarray(crop))
        mean_val = crop_tensor.mean().item()

        if mean_val < 0.15 or mean_val > 0.85:
            continue

        # 학습용 전처리
        tensor_img = preprocess(Image.fromarray(crop))

        save_path = os.path.join(
            save_dir,
            f"{base}_{part_name}.png"
        )

        save_image(tensor_img, save_path)