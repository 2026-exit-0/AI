import os
import json
import cv2
import numpy as np
from tqdm import tqdm

def process_damda_dataset(img_dir, label_root, output_root):
    if not os.path.exists(img_dir):
        print(f"오류: 이미지 폴더를 찾을 수 없습니다 -> {img_dir}")
        return

    image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    os.makedirs(output_root, exist_ok=True)
    
    device_map = {'D': '1. 디지털카메라', 'T': '2. 스마트패드', 'P': '3. 스마트폰'}
    success_count = 0

    # --- 전처리 설정 값 ---
    TARGET_SIZE = 512  # 해상도를 512로 상향하여 특징 보존
    JPEG_QUALITY = 95  # 저장 품질 설정

    for filename in tqdm(image_files):
        pure_name = os.path.splitext(filename)[0]
        parts = [p for p in pure_name.split('_') if p]
        
        if len(parts) < 3: continue

        device_code = parts[0][0].upper()
        subject_id = parts[1]
        session_id = parts[2]
        
        device_folder = device_map.get(device_code)
        subject_folder_path = os.path.join(label_root, device_folder, subject_id)
        
        # JSON 찾기 로직
        match_pattern = f"{subject_id}_{session_id}"
        target_json_path = None
        if os.path.exists(subject_folder_path):
            for f in os.listdir(subject_folder_path):
                if f.lower().endswith('.json') and f.startswith(match_pattern):
                    target_json_path = os.path.join(subject_folder_path, f)
                    break
        
        if not target_json_path: continue

        try:
            with open(target_json_path, "r", encoding='utf-8') as f:
                anno = json.load(f)

            # 데이터 구조 선택
            if "images" in anno:
                info = anno["images"]
            else:
                info = anno["annotations"][0]

            bbox = list(map(int, info["bbox"]))
            facepart = str(info.get("facepart", "00")).zfill(2)

            # 1. 이미지 로드 (한글 경로 대응)
            img_path = os.path.join(img_dir, filename)
            img_array = np.fromfile(img_path, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None: continue

            # 2. 크롭 영역 계산 (Bbox보다 약간 여유 있게)
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            crop_len = int(max(w, h) * 0.55) # 맥락 보존을 위해 10% 추가 마진
            
            y_min, y_max = max(cy - crop_len, 0), min(cy + crop_len, img.shape[0])
            x_min, x_max = max(cx - crop_len, 0), min(cx + crop_len, img.shape[1])
            cropped = img[y_min:y_max, x_min:x_max]

            # 3. 고화질 리사이즈 (Lanczos 보간법 사용)
            resized = cv2.resize(cropped, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_LANCZOS4)

            # 4. 피부 특징 강조 (Sharpening)
            # 미세 요철 및 모공 경계선을 뚜렷하게 만듦
            gaussian = cv2.GaussianBlur(resized, (0, 0), 2.0)
            sharpened = cv2.addWeighted(resized, 1.5, gaussian, -0.5, 0)

            # 5. 대비 강조 (CLAHE)
            # 조명 불균형을 해소하고 피부 질감을 극대화
            lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            l_final = clahe.apply(l)
            enhanced = cv2.cvtColor(cv2.merge((l_final, a, b)), cv2.COLOR_LAB2BGR)

            # 6. 저장 (한글 경로 대응 및 고품질 압축)
            save_dir = os.path.join(output_root, facepart)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{pure_name}_v2.jpg")
            
            is_success, buffer = cv2.imencode(".jpg", enhanced, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if is_success:
                with open(save_path, "wb") as f:
                    f.write(buffer)
                success_count += 1

        except Exception as e:
            print(f"\n[데이터 처리 에러] 파일명: {filename} | 에러내용: {e}")

    print(f"\n최종 성공: {success_count}개")

if __name__ == "__main__":
    # 사용자 환경 경로 (OneDrive 경로 직접 지정)
    img_p = r"C:\Users\YSB\OneDrive\Desktop\pre_images"
    lbl_p = r"C:\Users\YSB\OneDrive\Desktop\TL"
    out_p = r"C:\Users\YSB\OneDrive\Desktop\damda_processed_data_highres"
    
    process_damda_dataset(img_p, lbl_p, out_p)