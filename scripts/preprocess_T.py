import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import torchvision.transforms.functional as F

# ===============================
# 1. 경로 및 설정
# ===============================
INPUT_DIR = "./data/cropped_img/T"
OUTPUT_DIR = "./data/final_processed/T"
PATCH_SIZE = 224

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===============================
# 2. 개선된 태블릿 전처리 함수 (Unsharp Mask 방식)
# ===============================
def scanner_style_preprocess_tablet_v2(img_bgr):
    """
    라플라시안 대신 언샤프 마스크를 사용하여 
    노이즈 증폭 없이 선명도만 개선합니다.
    """
    # [Step 1] 노이즈 제거 (Bilateral Filter)
    # Median Blur보다 정교하게 엣지는 살리면서 입자 노이즈만 제거합니다.
    # d=5, sigma 값을 낮게 잡아 뭉개짐을 방지합니다.
    denoised = cv2.bilateralFilter(img_bgr, d=5, sigmaColor=25, sigmaSpace=25)

    # [Step 2] CLAHE (대비 조정 강도 약화)
    # clipLimit을 3.5 -> 1.8로 대폭 낮추어 화질이 깨지는 것을 막습니다.
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img_clahe = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    # [Step 3] 언샤프 마스크 (선명화 방식 변경)
    # 라플라시안처럼 노이즈를 직접 건드리지 않고, 
    # 가우시안 블러와 원본의 차이를 이용하여 깨끗하게 선명하게 만듭니다.
    gaussian = cv2.GaussianBlur(img_clahe, (0, 0), 2.0)
    img_unsharp = cv2.addWeighted(img_clahe, 1.5, gaussian, -0.5, 0)

    # [Step 4] 스캐너 색감 시뮬레이션 (PIL)
    img_rgb = cv2.cvtColor(img_unsharp, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    
    # 조명 표준화: 밝기 1.05배(과노출 방지), 채도 0.85
    pil_img = F.adjust_brightness(pil_img, brightness_factor=1.05)
    pil_img = F.adjust_saturation(pil_img, saturation_factor=0.85)
    
    final_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return final_bgr

# ===============================
# 3. 메인 루프 (보간법 수정)
# ===============================
for root, dirs, files in os.walk(INPUT_DIR):
    for file in tqdm(files):
        if not file.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        input_path = os.path.join(root, file)
        relative_path = os.path.relpath(root, INPUT_DIR)
        save_dir = os.path.join(OUTPUT_DIR, relative_path)
        os.makedirs(save_dir, exist_ok=True)

        img = cv2.imread(input_path)
        if img is None: continue

        # 태블릿 전용 전처리 v2 적용
        img_processed = scanner_style_preprocess_tablet_v2(img)

        # 리사이즈 보간법 변경 (INTER_AREA -> INTER_CUBIC)
        # INTER_CUBIC이 선을 더 매끄럽게 표현하여 화질이 덜 깨져 보입니다.
        img_final = cv2.resize(img_processed, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_CUBIC)

        save_path = os.path.join(save_dir, file)
        cv2.imwrite(save_path, img_final)