# issue 기록 남기기 주석(dacb7239103f0bffa4edd4fd7d4c4685bd795e55)
import os
import cv2
import shutil
import numpy as np
import torch
from torchvision import transforms as T
from torchvision.utils import save_image
from PIL import Image

# 1. 설정 및 경로
INPUT_DIR = './data_sbP'
OUTPUT_DIR = './processed_data'
CROP_SIZE = 224

if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR)

# 2. ResNet-50 표준 전처리
def get_resnet_preprocess(size=224):
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        # ImageNet 정규화
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# 3. 각도별 부위 좌표 비율
ANGLE_CONFIGS = {
    "F":   {"Forehead": (0.25, 0.5), "Nose": (0.55, 0.5), "L_Cheek": (0.60, 0.35), "R_Cheek": (0.60, 0.65)},
    "L30": {"Forehead": (0.25, 0.4), "Nose": (0.55, 0.45), "R_Cheek": (0.60, 0.7), "L_Cheek": (0.60, 0.2)},
    "R30": {"Forehead": (0.25, 0.6), "Nose": (0.55, 0.55), "L_Cheek": (0.60, 0.3), "R_Cheek": (0.60, 0.8)},
    "Fb":  {"Forehead": (0.4, 0.5), "Nose": (0.7, 0.5), "L_Cheek": (0.75, 0.35), "R_Cheek": (0.75, 0.65)},
    "Ft":  {"Forehead": (0.15, 0.5), "Nose": (0.4, 0.5), "L_Cheek": (0.5, 0.35), "R_Cheek": (0.5, 0.65)}
}

def process_stable():
    preprocess = get_resnet_preprocess(size=CROP_SIZE)
    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"🚀 총 {len(files)}개의 파일 처리를 시작합니다.")

    for file_name in files:
        # 파일명에서 라벨 및 각도 추출
        parts = file_name.split('_')
        label = parts[0]
        angle_key = parts[-1].split('.')[0]
        config = ANGLE_CONFIGS.get(angle_key, ANGLE_CONFIGS["F"])

        # 이미지 로드 (한글 경로 대응)
        img_path = os.path.join(INPUT_DIR, file_name)
        img_array = np.fromfile(img_path, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None: continue

        # 4. 필수 변환: BGR -> RGB (색상 뒤바뀜 방지)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img_rgb.shape

        for part_name, (ry, rx) in config.items():
            # 좌표 변환 및 크롭
            cx, cy = int(rx * w), int(ry * h)
            x1, y1 = max(0, cx - CROP_SIZE // 2), max(0, cy - CROP_SIZE // 2)
            crop_img = img_rgb[y1:y1+CROP_SIZE, x1:x1+CROP_SIZE]
            
            # 타입 안정성 확보: 0~255 정수형(uint8) 유지 (검은 화면 방지)
            crop_img_uint8 = crop_img.astype(np.uint8)
            
            # PIL 변환 후 ResNet 전처리 적용
            pil_img = Image.fromarray(crop_img_uint8)
            tensor_img = preprocess(pil_img)

            # 텐서의 평균값을 계산 (0~1 사이 값) - 임시방편조치이니 수정 권장
            mean_val = tensor_img.mean().item() # type: ignore

            # 필터링 조건 설정 (예: 0.15 미만은 너무 어둡고, 0.85 이상은 너무 밝음)
            if 0.15 < mean_val < 0.85: 
                save_dir = os.path.join(OUTPUT_DIR, label)
                os.makedirs(save_dir, exist_ok=True)
                
                save_name = f"{angle_key}_{part_name}_{file_name}"
                save_image(tensor_img, os.path.join(save_dir, save_name)) # type: ignore
            else:
                # 정보가 없는 사진은 저장하지 않고 로그만 남김
                reason = "과노출(날아감)" if mean_val >= 0.85 else "저노출(어두움)"
                print(f"⚠️ 제외됨: {file_name} [{part_name}] - {reason} (Mean: {mean_val:.2f})")

    print(f"✅ 전처리 완료! 생성 위치: {OUTPUT_DIR}")

if __name__ == "__main__":
    process_stable()