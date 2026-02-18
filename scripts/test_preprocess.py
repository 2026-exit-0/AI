import os
import cv2
import numpy as np
from tqdm import tqdm   # 반복문 진행 상황을 시각적으로 보여주는 라이브러리

RAW_DIR = "./data/raw_images"
SAVE_DIR = "./data/processed_images"
os.makedirs(SAVE_DIR, exist_ok=True)

# OpenCV에서 제공되는 얼굴 검출 모델로 정면 얼굴 검출
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# 패치 크기와 이미지당 패치 수 설정
PATCH_SIZE = 224
PATCH_PER_IMAGE = 3

# 랜덤 피부 패치 추출 함수
def extract_patches(face_img):
    h, w, _ = face_img.shape
    patches = []
    
    for _ in range(PATCH_PER_IMAGE):
        # 얼굴 크기가 PATCH_SIZE보다 작으면 resize
        if h < PATCH_SIZE or w < PATCH_SIZE:
            resized = cv2.resize(face_img, (PATCH_SIZE, PATCH_SIZE))
            patches.append(resized)
            continue
        
        x = np.random.randint(0, w - PATCH_SIZE)
        y = np.random.randint(0, h - PATCH_SIZE)
        
        # 랜덤 시작점에서 영역 Crop
        patch = face_img[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
        patches.append(patch)
    
    return patches

# 원본 이미지 폴더 안 모든 파일에서 패치 생성
# tqdm으로 진행률 표시
for img_name in tqdm(os.listdir(RAW_DIR)):
    path = os.path.join(RAW_DIR, img_name)
    img = cv2.imread(path)
    
    if img is None:
        continue
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        continue
    
    x, y, w, h = max(faces, key=lambda b: b[2]*b[3])
    face = img[y:y+h, x:x+w]
    
    patches = extract_patches(face)
    
    base = os.path.splitext(img_name)[0]
    
    for i, patch in enumerate(patches):
        save_name = f"{base}_patch{i}.png"
        save_path = os.path.join(SAVE_DIR, save_name)
        cv2.imwrite(save_path, patch)
