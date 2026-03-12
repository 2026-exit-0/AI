import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import torchvision.transforms.functional as F

# ===============================
# 1. 경로 및 설정 (핸드폰 폴더 'P')
# ===============================
INPUT_DIR = "./data/cropped_img/P"
OUTPUT_DIR = "./data/final_processed/P"
PATCH_SIZE = 224

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===============================
# 2. 핸드폰 이미지용 복합 선명화 통합 함수 (완전 수정)
# ===============================
def scanner_style_preprocess_phone(img_bgr):
    """
    뭉개짐 현상을 해결하기 위해 노이즈 감소는 약화시키고, 
    다각도의 선명화 기법을 복합적으로 적용하여 미세 특징을 복원합니다.
    """
    # [Step 1] 미세한 노이즈만 제거 (Bilateral Filter 강도 대폭 약화)
    # 뭉개짐의 원인이었던 강도를 d=5, sigma 값을 15 정도로 낮추어 노이즈만 살짝 잡습니다.
    denoised_bgr = cv2.bilateralFilter(img_bgr, d=5, sigmaColor=15, sigmaSpace=15)

    # [Step 2] 마일드한 CLAHE (L 채널만)
    # 특징 복원을 위해 clipLimit을 이전보다 살짝 높인 3.0으로 설정하되, 타일 크기를 늘려 부드럽게 강조합니다.
    lab = cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(12, 12)) # 타일 크기 증가로 부드러움 확보
    l = clahe.apply(l)
    merged = cv2.merge((l, a, b))
    clahe_bgr = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    # [Step 3] 복합 선명화 (뭉개짐 디테일 복원 핵심)
    
    # 3-1. Unsharp Masking (모공/흉터 경계 뚜렷하게)
    # 가우시안 블러와 원본의 차이를 이용하여 경계를 강조합니다.
    gaussian = cv2.GaussianBlur(clahe_bgr, (0, 0), 1.5)
    unsharp = cv2.addWeighted(clahe_bgr, 1.5, gaussian, -0.5, 0)

    # 3-2. 고주파 강조 필터 (가장 미세한 질감 복원)
    # 모공의 형태가 완전히 뭉개진 것을 복원하기 위해 사용합니다.
    kernel_hp = np.array([[ 0, -0.2,  0], 
                            [-0.2, 1.8, -0.2], 
                            [ 0, -0.2,  0]])
    img_sharp_final = cv2.filter2D(unsharp, -1, kernel_hp)

    # [Step 4] 색상 및 밝기 최종 조정 (PIL)
    pil_img = Image.fromarray(cv2.cvtColor(img_sharp_final, cv2.COLOR_BGR2RGB))
    # 밝기는 1.1배로 살짝 올리고, 채도는 0.85로 낮추어 스캐너 특유의 차가운 톤을 강화합니다.
    pil_img = F.adjust_brightness(pil_img, brightness_factor=1.1)
    pil_img = F.adjust_saturation(pil_img, saturation_factor=0.85)

    # PIL -> RGB -> BGR
    final_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    return final_bgr

# ===============================
# 3. 메인 루프 (핸드폰 이미지 폴더 순회 및 저장)
# ===============================
print(f"작업 시작: {INPUT_DIR} -> {OUTPUT_DIR}")

for root, dirs, files in os.walk(INPUT_DIR):
    for file in tqdm(files):
        if not file.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        input_path = os.path.join(root, file)
        relative_path = os.path.relpath(root, INPUT_DIR)
        save_dir = os.path.join(OUTPUT_DIR, relative_path)
        os.makedirs(save_dir, exist_ok=True)

        # 1. 이미지 읽기 (BGR)
        img = cv2.imread(input_path)
        if img is None:
            continue

        # 2. 핸드폰 이미지용 스캐너 전처리 적용 (BGR)
        img_processed = scanner_style_preprocess_phone(img)

        # 3. 밝기 극단값 필터링 (너무 어둡거나 날아간 사진 제외)
        gray = cv2.cvtColor(img_processed, cv2.COLOR_BGR2GRAY)
        mean_val = gray.mean() / 255.0

        if mean_val < 0.15 or mean_val > 0.85:
            # 너무 어둡거나(0.15미만) 너무 밝은(0.85초과) 이미지는 제외
            continue

        # 4. ResNet 규격(224x224) 리사이즈 후 저장 (Normalize 적용하지 않음)
        img_final = cv2.resize(img_processed, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_CUBIC)

        save_path = os.path.join(save_dir, file)
        cv2.imwrite(save_path, img_final)

print("핸드폰 이미지 전처리 및 저장이 완료되었습니다.")