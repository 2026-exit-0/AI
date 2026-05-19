import os
import json
import cv2
import numpy as np
from tqdm import tqdm

def process_damda_dataset(img_dir, label_root, output_root):
    image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    os.makedirs(output_root, exist_ok=True)
    
    device_map = {'D': '1. 디지털카메라', 'T': '2. 스마트패드', 'P': '3. 스마트폰'}
    success_count = 0

    for filename in tqdm(image_files):
        # 디버깅을 위해 try-except를 제거하거나 에러를 명시적으로 출력합니다.
        pure_name = os.path.splitext(filename)[0]
        parts = [p for p in pure_name.split('_') if p]
        
        if len(parts) < 3: continue

        device_code = parts[0][0].upper()
        subject_id = parts[1]
        session_id = parts[2]
        
        device_folder = device_map.get(device_code)
        subject_folder_path = os.path.join(label_root, device_folder, subject_id)
        
        # JSON 찾기
        match_pattern = f"{subject_id}_{session_id}"
        target_json_path = None
        if os.path.exists(subject_folder_path):
            for f in os.listdir(subject_folder_path):
                if f.lower().endswith('.json') and f.startswith(match_pattern):
                    target_json_path = os.path.join(subject_folder_path, f)
                    break
        
        if not target_json_path: continue

        # --- 이 부분에서 에러가 날 확률이 높습니다 ---
        with open(target_json_path, "r", encoding='utf-8') as f:
            anno = json.load(f)

        # JSON 구조 확인을 위한 출력 (첫 번째 파일만)
        if success_count == 0:
            print(f"\n[JSON 구조 확인]: {anno.keys()}")
            # 만약 여기서 에러가 난다면 info를 찾는 방식이 틀린 것입니다.

        # AI Hub 데이터셋 버전에 따라 구조가 다를 수 있음
        try:
            # 보통 'images' 혹은 'annotations' 안에 데이터가 있습니다.
            if "images" in anno:
                info = anno["images"]
            else:
                info = anno["annotations"][0] # 리스트 형태일 경우

            bbox = list(map(int, info["bbox"])) # 여기서 에러 발생 가능성
            facepart = str(info.get("facepart", "00")).zfill(2)

            # 이미지 로드 및 전처리
            img_path = os.path.join(img_dir, filename)
            img_array = np.fromfile(img_path, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            # 크롭 및 저장
            x1, y1, x2, y2 = bbox
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            crop_len = max(x2 - x1, y2 - y1) // 2
            y_min, y_max = max(cy - crop_len, 0), min(cy + crop_len, img.shape[0])
            x_min, x_max = max(cx - crop_len, 0), min(cx + crop_len, img.shape[1])
            cropped = img[y_min:y_max, x_min:x_max]

            # (중략: CLAHE 및 리사이즈 로직 동일)
            resized = cv2.resize(cropped, (224, 224))
            
            save_dir = os.path.join(output_root, facepart)
            os.makedirs(save_dir, exist_ok=True)
            cv2.imwrite(os.path.join(save_dir, f"{pure_name}.jpg"), resized)
            
            success_count += 1
        except Exception as e:
            print(f"\n[데이터 처리 에러] 파일명: {filename} | 에러내용: {e}")
            # 에러 원인을 알기 위해 한 번만 출력하고 멈추려면 아래 break를 쓰세요
            # break 

    print(f"\n최종 성공: {success_count}개")

if __name__ == "__main__":
    img_p = r"C:\Users\YSB\OneDrive\Desktop\pre_images"
    lbl_p = r"C:\Users\YSB\OneDrive\Desktop\TL"
    out_p = r"C:\Users\YSB\OneDrive\Desktop\damda_processed_data"
    process_damda_dataset(img_p, lbl_p, out_p)