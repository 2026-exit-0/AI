import os
import cv2
import numpy as np
import random
from tqdm import tqdm
from PIL import Image
import torch
from torchvision import transforms as T

# ===============================
# 1. 경로 및 설정
# ===============================
INPUT_DIR = "./data/cropped_img"      # 원본 부위별 크롭 이미지 경로
OUTPUT_DIR = "./data/final_processed_v2"  # 전처리 후 저장 경로
os.makedirs(OUTPUT_DIR, exist_ok=True)

PATCH_SIZE = 512  # 가이드라인에 따른 ResNet-50 입력 사이즈
BLUR_THRESHOLD = 50.0  # 과도한 블러 필터링 임계값 (값이 낮을수록 흐림)

# ===============================
# 2. 전처리 핵심 로직 (피부 질감 유지)
# ===============================

def apply_mild_clahe(img_bgr):
    """L 채널에 CLAHE를 적용하여 명암비를 개선하되, 노이즈 증폭을 억제함"""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # clipLimit 1.5로 설정하여 과한 보정 방지 (가이드라인 준수)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    l = clahe.apply(l)

    merged = cv2.merge((l, a, b))
    img_bgr = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    # 아주 약한 샤프닝 (주름/질감 강조)
    blurred = cv2.GaussianBlur(img_bgr, (0, 0), 1.0)
    img_bgr = cv2.addWeighted(img_bgr, 1.1, blurred, -0.1, 0)

    return img_bgr

def scanner_style_preprocess(img_bgr):
    """스캐너 스타일의 깔끔한 피부 이미지를 위해 양방향 필터 적용"""
    img_processed = apply_mild_clahe(img_bgr)
    # Bilateral Filter: 엣지(주름)는 보존하고 노이즈만 제거
    img_processed = cv2.bilateralFilter(img_processed, d=5, sigmaColor=30, sigmaSpace=30)
    return img_processed

# ===============================
# 3. 데이터셋 검증 및 분할 (Metadata 생성)
# ===============================

def generate_metadata(output_dir):
    """학습이 가능한 데이터인지 판단하기 위한 리스트 파일 생성"""
    all_files = []
    
    # 폴더 구조를 순회하며 파일 경로와 라벨(폴더명) 수집
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file.lower().endswith((".jpg", ".png")):
                rel_path = os.path.relpath(os.path.join(root, file), output_dir)
                label = os.path.basename(root) 
                all_files.append(f"{rel_path} {label}")

    if not all_files:
        print("⚠️ 생성된 이미지가 없습니다. 경로를 확인하세요.")
        return

    random.shuffle(all_files)
    
    # 8:2 비율로 분할 (Baseline 체크용)
    split_idx = int(len(all_files) * 0.8)
    train_list = all_files[:split_idx]
    test_list = all_files[split_idx:]

    # Classification 결과 파일
    with open("class_test.txt", "w") as f:
        f.write("\n".join(test_list))
        
    # Regression 결과 파일 (동일 리스트 활용 혹은 점수 매핑 로직 추가 가능)
    with open("regression_test.txt", "w") as f:
        f.write("\n".join(test_list)) 

    print(f"\n✅ 메타데이터 생성 완료!")
    print(f"   - 총 데이터: {len(all_files)}건")
    print(f"   - 테스트용(class_test.txt): {len(test_list)}건 저장 완료.")

# ===============================
# 4. 메인 실행 루프
# ===============================

def main():
    print("🚀 전처리 시작...")
    
    for root, dirs, files in os.walk(INPUT_DIR):
        for file in tqdm(files):
            if not file.lower().endswith((".jpg", ".png")):
                continue

            input_path = os.path.join(root, file)
            img = cv2.imread(input_path)
            if img is None: continue

            # [검증 1] 과도한 블러 체크 (Laplacian Variance)
            lap_var = cv2.Laplacian(img, cv2.CV_64F).var()
            if lap_var < BLUR_THRESHOLD:
                continue

            # 전처리 적용
            img_processed = scanner_style_preprocess(img)

            # [검증 2] 밝기 극단값 필터링 (너무 어둡거나 밝은 사진 제외)
            mean_val = np.mean(img_processed) / 255.0
            if mean_val < 0.1 or mean_val > 0.9:
                continue

            # [사이즈 조정] 512x512 고정
            img_resized = cv2.resize(img_processed, (PATCH_SIZE, PATCH_SIZE))

            # 저장 경로 설정
            relative_path = os.path.relpath(root, INPUT_DIR)
            save_dir = os.path.join(OUTPUT_DIR, relative_path)
            os.makedirs(save_dir, exist_ok=True)
            
            save_path = os.path.join(save_dir, file)
            cv2.imwrite(save_path, img_resized)

    print("✨ 모든 이미지 전처리 완료.")
    
    # 마지막 단계: 메타데이터 생성
    generate_metadata(OUTPUT_DIR)

if __name__ == "__main__":
    main()