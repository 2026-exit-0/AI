import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import torchvision.transforms.functional as F

# ===============================
# 1. 경로 및 설정
# ===============================
# DSLR 폴더만 타겟팅하려면 경로를 './data/cropped_img/D'로 설정하거나, 
# 하위 로직에서 'D' 폴더인 경우에만 특정 필터를 적용하도록 설정할 수 있습니다.
INPUT_DIR = "./data/cropped_img/D" 
OUTPUT_DIR = "./data/final_processed/D"
PATCH_SIZE = 224

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===============================
# 2. DSLR -> 스캐너 시뮬레이션 함수 (BGR 기반)
# ===============================
def apply_dslr_to_scanner_effect(img_bgr):
    """
    DSLR의 부드러운 이미지를 스캐너 특유의 고대비, 고선명, 고밝기로 변환
    """
    # [Step 1] CLAHE (강하게 적용하여 모공/요철 부각)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    # clipLimit을 3.0~4.0으로 주면 스캐너 특유의 대비가 살아납니다.
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img_bgr = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    # [Step 2] Sharpening (경계선을 날카롭게)
    # 주변 노이즈를 억제하면서 경계만 살리기 위해 가우시안 블러와 가중치 합산 사용
    kernel = np.array([[-1, -1, -1], 
                        [-1,  9, -1], 
                        [-1, -1, -1]])
    img_sharp = cv2.filter2D(img_bgr, -1, kernel)
    img_bgr = cv2.addWeighted(img_bgr, 0.7, img_sharp, 0.3, 0)

    # [Step 3] Bilateral Filter (피부결은 유지하면서 미세 잡티 정리)
    img_bgr = cv2.bilateralFilter(img_bgr, d=5, sigmaColor=30, sigmaSpace=30)

    # [Step 4] 밝기 및 색감 조정 (PIL 변환 후 처리)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    
    # 스캐너 LED 조명 모사 (밝기 1.2배, 채도 약간 낮춤)
    pil_img = F.adjust_brightness(pil_img, brightness_factor=1.15)
    pil_img = F.adjust_saturation(pil_img, saturation_factor=0.9)
    
    # 다시 OpenCV용 BGR로 복귀
    final_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return final_bgr

# ===============================
# 3. 메인 루프 (파일 순회 및 저장)
# ===============================
print(f"작업 시작: {INPUT_DIR} -> {OUTPUT_DIR}")

for root, dirs, files in os.walk(INPUT_DIR):
    for file in tqdm(files):
        if not file.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        # 경로 설정
        input_path = os.path.join(root, file)
        relative_path = os.path.relpath(root, INPUT_DIR)
        save_dir = os.path.join(OUTPUT_DIR, relative_path)
        os.makedirs(save_dir, exist_ok=True)

        # 1. 이미지 읽기
        img = cv2.imread(input_path)
        if img is None:
            continue

        # 2. DSLR -> 스캐너 전처리 적용
        img_processed = apply_dslr_to_scanner_effect(img)

        # 3. 밝기 극단값 필터링 (너무 어둡거나 날아간 사진 제외)
        # 0~1 사이 값으로 변환하여 평균 확인
        gray = cv2.cvtColor(img_processed, cv2.COLOR_BGR2GRAY)
        mean_val = gray.mean() / 255.0

        if mean_val < 0.15 or mean_val > 0.85:
            # 너무 어둡거나(0.15미만) 너무 밝은(0.85초과) 이미지는 학습 품질을 위해 제외
            continue

        # 4. ResNet 규격(224x224) 리사이즈
        img_final = cv2.resize(img_processed, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_CUBIC)

        # 5. 저장 (Normalize는 학습 시 DataLoader에서 진행하므로 여기서는 이미지로 저장)
        save_path = os.path.join(save_dir, file)
        cv2.imwrite(save_path, img_final)

print("전처리 및 저장이 완료되었습니다.")