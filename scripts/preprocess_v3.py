import os
import cv2
import numpy as np
import random

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        def __init__(self, iterable, **kwargs):
            self._iter = list(iterable)
            self._total = len(self._iter)
        def __iter__(self):
            for i, item in enumerate(self._iter, 1):
                print(f"\r  처리 중 {i}/{self._total}", end="", flush=True)
                yield item
            print()

# ===============================
# 1. 경로 및 설정
# ===============================
INPUT_DIR = "./data/cropped_img"
OUTPUT_DIR = "./data/final_processed_v3"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PATCH_SIZE = 512
# [수정] 212px 소이미지는 Laplacian variance가 낮게 나오므로 임계값 하향 조정 (50 → 30)
BLUR_THRESHOLD = 30.0

# ===============================
# 2. 단계적 업스케일링 (핵심 수정)
# ===============================

def two_stage_upscale(img_bgr, target_size=512):
    """
    212 → 512 단순 확대 시 블러 발생 문제 해결.
    
    [변경 이유]
    - 2.4배 이상 한 번에 확대하면 INTER_LINEAR/CUBIC 모두 블러 발생
    - 중간 스텝(300px)을 거쳐 2단계로 나눠 확대하면 엣지 손실 최소화
    - INTER_LANCZOS4: 8x8 커널 기반 고품질 보간, 소->대 업스케일에 최적
    """
    h, w = img_bgr.shape[:2]
    
    if max(h, w) < target_size * 0.6:
        # 2단계 업스케일: 중간 스텝 경유
        mid_size = int(target_size * 0.65)  # ~333px
        img_mid = cv2.resize(img_bgr, (mid_size, mid_size), interpolation=cv2.INTER_CUBIC)
        img_up = cv2.resize(img_mid, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
    else:
        img_up = cv2.resize(img_bgr, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
    
    return img_up


def unsharp_mask(img_bgr, sigma=1.2, strength=1.3):
    """
    업스케일 후 손실된 엣지(주름, 모공 등 피부 텍스처) 복구.
    
    [변경 이유]
    - 기존 addWeighted 방식보다 자연스러운 선명도 복구
    - sigma 1.2: 피부 텍스처 주파수 대역에 맞게 조정
    - strength 1.3: 과도한 샤프닝 방지 (노이즈 증폭 억제)
    """
    blurred = cv2.GaussianBlur(img_bgr, (0, 0), sigma)
    sharpened = cv2.addWeighted(img_bgr, strength, blurred, -(strength - 1.0), 0)
    return sharpened

# ===============================
# 3. 전처리 핵심 로직
# ===============================

def apply_mild_clahe(img_bgr):
    """
    [변경 이유] clipLimit 1.5 → 2.0
    212px 소이미지는 명암 정보 밀도가 낮아 더 강한 보정이 필요.
    tileGridSize (8,8) 유지: 512px 기준 타일 크기로 적절.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    merged = cv2.merge((l, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def scanner_style_preprocess(img_bgr):
    """
    전처리 파이프라인:
    1. CLAHE (명암비 보정)
    2. Bilateral Filter (엣지 보존 노이즈 제거)
    3. 단계적 업스케일 (512x512, 화질 손실 최소화)
    4. Unsharp Mask (업스케일 후 텍스처 복구)
    
    [Bilateral Filter 파라미터 변경]
    - d: 5 → 7: 소이미지(212px)는 노이즈 비율이 높아 더 넓은 커널 필요
    - sigmaColor: 30 → 50: 피부 톤 경계를 더 잘 보존
    - sigmaSpace: 30 → 30: 공간 가중치는 유지
    """
    img_processed = apply_mild_clahe(img_bgr)
    img_processed = cv2.bilateralFilter(img_processed, d=7, sigmaColor=50, sigmaSpace=30)
    
    # [순서 변경] 업스케일을 전처리 후 마지막에 수행
    # → 작은 이미지에서 필터 적용 후 확대해야 아티팩트 최소화
    img_processed = two_stage_upscale(img_processed, target_size=512)
    img_processed = unsharp_mask(img_processed, sigma=1.2, strength=1.3)
    
    return img_processed

# ===============================
# 4. 데이터셋 검증 및 분할
# ===============================

def generate_metadata(output_dir):
    all_files = []
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
    split_idx = int(len(all_files) * 0.8)
    train_list = all_files[:split_idx]
    test_list = all_files[split_idx:]

    with open("class_test.txt", "w") as f:
        f.write("\n".join(test_list))

    with open("regression_test.txt", "w") as f:
        f.write("\n".join(test_list))

    print(f"\n✅ 메타데이터 생성 완료!")
    print(f"   - 총 데이터: {len(all_files)}건")
    print(f"   - 테스트용(class_test.txt): {len(test_list)}건 저장 완료.")

# ===============================
# 5. 메인 실행 루프
# ===============================

def main():
    print("🚀 전처리 시작...")

    for root, dirs, files in os.walk(INPUT_DIR):
        for file in tqdm(files):
            if not file.lower().endswith((".jpg", ".png")):
                continue

            input_path = os.path.join(root, file)
            img = cv2.imread(input_path)
            if img is None:
                continue

            # [검증 1] 블러 체크 (임계값 완화: 50 → 30)
            # 212px 소이미지는 고주파 성분이 적어 variance가 낮게 측정됨
            lap_var = cv2.Laplacian(img, cv2.CV_64F).var()
            if lap_var < BLUR_THRESHOLD:
                continue

            # 전처리 적용 (CLAHE → Bilateral → 업스케일 → Unsharp)
            img_processed = scanner_style_preprocess(img)

            # [검증 2] 밝기 극단값 필터링 (범위 완화: 0.1~0.9 → 0.08~0.92)
            # 소이미지는 부위 특성상(다크서클, 이마 등) 극단값이 나오기 쉬움
            mean_val = np.mean(img_processed) / 255.0
            if mean_val < 0.08 or mean_val > 0.92:
                continue

            # 저장
            relative_path = os.path.relpath(root, INPUT_DIR)
            save_dir = os.path.join(OUTPUT_DIR, relative_path)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, file)
            cv2.imwrite(save_path, img_processed, [cv2.IMWRITE_JPEG_QUALITY, 95])

    print("✨ 모든 이미지 전처리 완료.")
    generate_metadata(OUTPUT_DIR)


if __name__ == "__main__":
    main()