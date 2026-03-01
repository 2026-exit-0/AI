# 모델 학습을 위한 변경
# 1. 사진 속 글씨 삭제
# 2. json 파일과 사진 연결
# 3. 1:1파일 매칭을 위한 파일명 변경(전처리 사진에 대한)

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
    
    # --- [데이터 설명서 기반 부위 매핑 복구] ---
    FACEPART_MAP = {
        "01": "Forehead", "02": "Glabella", "03": "R_Eye", "04": "L_Eye",
        "05": "R_Cheek", "06": "L_Cheek", "07": "Lip", "08": "Chin",
        1: "Forehead", 2: "Glabella", 3: "R_Eye", 4: "L_Eye",
        5: "R_Cheek", 6: "L_Cheek", 7: "Lip", 8: "Chin"
    }

    device_map = {'D': '1. 디지털카메라', 'T': '2. 스마트패드', 'P': '3. 스마트폰'}
    success_count = 0
    TARGET_SIZE = 512  
    JPEG_QUALITY = 95  

    for filename in tqdm(image_files):
        pure_name = os.path.splitext(filename)[0]
        parts = [p for p in pure_name.split('_') if p]
        
        if len(parts) < 3: continue

        # --- [경로 탐색 로직 유지] ---
        device_code = parts[0][0].upper()
        subject_id = parts[1]
        session_id = parts[2]
        
        device_folder = device_map.get(device_code)
        subject_folder_path = os.path.join(label_root, device_folder, subject_id)
        
        match_pattern = f"{subject_id}_{session_id}"
        
        if not os.path.exists(subject_folder_path):
            continue

        # 이미지 로드
        img_path = os.path.join(img_dir, filename)
        img_array = np.fromfile(img_path, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None: continue

        # 해당 이미지에 매칭되는 모든 부위(JSON) 처리
        for f_json in os.listdir(subject_folder_path):
            if f_json.lower().endswith('.json') and f_json.startswith(match_pattern):
                target_json_path = os.path.join(subject_folder_path, f_json)
                
                try:
                    with open(target_json_path, "r", encoding='utf-8') as f:
                        anno = json.load(f)

                    # 엉뚱한 부위 크롭 방지 로직 (각도 확인)
                    img_angle = parts[-1].upper()
                    json_angle = str(anno.get("angle", "")).upper()
                    if not json_angle:
                        json_fname = anno.get("info", {}).get("filename") or anno.get("filename", "")
                        if json_fname:
                            json_angle = os.path.splitext(json_fname)[0].split('_')[-1].upper()
                    
                    if json_angle and img_angle != json_angle:
                        continue

                    # 데이터 및 BBOX 추출
                    images_data = anno.get("images") or {}
                    
                    bbox = images_data.get("bbox")
                    if not bbox and "annotations" in anno:
                        if isinstance(anno["annotations"], list) and len(anno["annotations"]) > 0:
                            bbox = anno["annotations"][0].get("bbox")
                    
                    if not bbox: continue

                    # 크롭 수행
                    x1, y1, x2, y2 = map(int, bbox)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
                    
                    cropped = img[y1:y2, x1:x2].copy()
                    if cropped.size == 0: continue

                    # 비율 유지 리사이즈
                    h_crop, w_crop = cropped.shape[:2]
                    scale = TARGET_SIZE / max(h_crop, w_crop)
                    new_w, new_h = int(w_crop * scale), int(h_crop * scale)
                    
                    # 화질 개선 전처리 (CLAHE 등) 유지
                    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                    gaussian = cv2.GaussianBlur(resized, (0, 0), 2.0)
                    sharpened = cv2.addWeighted(resized, 1.5, gaussian, -0.5, 0)
                    lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
                    l, a, b = cv2.split(lab)
                    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
                    l_final = clahe.apply(l)
                    enhanced = cv2.cvtColor(cv2.merge((l_final, a, b)), cv2.COLOR_LAB2BGR)

                    # 🚨 텍스트 그리는 부분(cv2.putText 등)을 모두 삭제했습니다! 🚨

                    # 부위 이름 찾기 (폴더 생성용)
                    fp_code = images_data.get("facepart", "00")
                    part_name = FACEPART_MAP.get(fp_code, FACEPART_MAP.get(str(fp_code), f"Part_{fp_code}"))

                    # 저장 (부위 이름별 폴더 분류)
                    save_dir = os.path.join(output_root, part_name.upper())
                    os.makedirs(save_dir, exist_ok=True)
                    
                    # 🚨 [수정됨] 파일명을 JSON 파일 이름과 완전히 동일하게 맞춤 (확장자만 jpg)
                    json_basename = os.path.splitext(f_json)[0]
                    save_path = os.path.join(save_dir, f"{json_basename}.jpg")
                    
                    # 이미지 저장
                    _, buffer = cv2.imencode(".jpg", enhanced, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
                    with open(save_path, "wb") as f_out:
                        f_out.write(buffer)
                    success_count += 1

                except Exception as e:
                    print(f"\n[에러] {filename} ({f_json}) 처리 중 오류: {e}")

    print(f"\n✨ 최종 성공: {success_count}개 부위 텍스트 없이 깔끔하게 처리 완료")

if __name__ == "__main__":
    img_p = r'./TS' 
    lbl_p = r'./TL'
    out_p = r'./damda_cropped_dataset'  
    
    process_damda_dataset(img_p, lbl_p, out_p)