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

    # --- [질문자님 원본 설정 유지] ---
    device_map = {'D': '1. 디지털카메라', 'T': '2. 스마트패드', 'P': '3. 스마트폰'}
    success_count = 0
    TARGET_SIZE = 512  
    JPEG_QUALITY = 95  

    for filename in tqdm(image_files):
        pure_name = os.path.splitext(filename)[0]
        parts = [p for p in pure_name.split('_') if p]
        
        if len(parts) < 3: continue

        # --- [질문자님 원본 경로 로직: 절대 수정 안 함] ---
        device_code = parts[0][0].upper()
        subject_id = parts[1]
        session_id = parts[2]
        
        device_folder = device_map.get(device_code)
        subject_folder_path = os.path.join(label_root, device_folder, subject_id)
        
        match_pattern = f"{subject_id}_{session_id}"
        
        if not os.path.exists(subject_folder_path):
            continue

        # 이미지 로드 (여러 JSON이 있어도 원본 로드는 한 번만)
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

                    # 1. 데이터 추출 (None 에러 방지 처리)
                    images_data = anno.get("images") or {}
                    # equipment가 null이거나 없을 경우 빈 딕셔너리로 대체하여 .items() 에러 방지
                    equip_data = anno.get("equipment")
                    if not isinstance(equip_data, dict):
                        equip_data = {}
                    
                    # 2. BBOX 추출 (images 내부 혹은 annotations 확인) 
                    bbox = images_data.get("bbox")
                    if not bbox and "annotations" in anno:
                        if isinstance(anno["annotations"], list) and len(anno["annotations"]) > 0:
                            bbox = anno["annotations"][0].get("bbox")
                    
                    if not bbox: continue

                    # 3. 크롭 수행 ([x1, y1, x2, y2] 좌표 방식)
                    x1, y1, x2, y2 = map(int, bbox)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
                    
                    cropped = img[y1:y2, x1:x2].copy()
                    if cropped.size == 0: continue

                    # 4. 리사이즈 및 전처리 (Sharpening + CLAHE)
                    resized = cv2.resize(cropped, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_LANCZOS4)
                    gaussian = cv2.GaussianBlur(resized, (0, 0), 2.0)
                    sharpened = cv2.addWeighted(resized, 1.5, gaussian, -0.5, 0)
                    lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
                    l, a, b = cv2.split(lab)
                    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
                    l_final = clahe.apply(l)
                    enhanced = cv2.cvtColor(cv2.merge((l_final, a, b)), cv2.COLOR_LAB2BGR)

                    # 5. 수치 데이터 추출 (설명서 기반 moisture, elasticity 우선 확인) 
                    m_val = anno.get("moisture") or next((v for k, v in equip_data.items() if 'moisture' in k.lower()), "N/A")
                    e_val = anno.get("elasticity") or next((v for k, v in equip_data.items() if 'elasticity_r2' in k.lower()), "N/A")

                    # 6. 텍스트 표시
                    cv2.rectangle(enhanced, (0, 0), (TARGET_SIZE, 85), (0, 0, 0), -1)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    def fmt(v):
                        try: return f"{float(v):.2f}"
                        except: return str(v)

                    # 부위 번호 매핑 사용
                    fp_code = images_data.get("facepart", "00")
                    part_name = FACEPART_MAP.get(fp_code, FACEPART_MAP.get(str(fp_code), f"Part_{fp_code}"))

                    cv2.putText(enhanced, f"Part: {part_name}", (15, 25), font, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(enhanced, f"Moist: {fmt(m_val)}", (15, 52), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(enhanced, f"Elastic: {fmt(e_val)}", (15, 79), font, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

                    # 7. 저장 (부위 이름별 폴더 분류)
                    save_dir = os.path.join(output_root, part_name.upper())
                    os.makedirs(save_dir, exist_ok=True)
                    
                    save_path = os.path.join(save_dir, f"{pure_name}_{part_name}.jpg")
                    _, buffer = cv2.imencode(".jpg", enhanced, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
                    with open(save_path, "wb") as f_out:
                        f_out.write(buffer)
                    success_count += 1

                except Exception as e:
                    print(f"\n[에러] {filename} ({f_json}) 처리 중 오류: {e}")

    print(f"\n최종 성공: {success_count}개 부위 처리 완료")

if __name__ == "__main__":
    img_p = r"C:\Users\YSB\OneDrive\Desktop\pre_images"
    lbl_p = r"C:\Users\YSB\OneDrive\Desktop\TL"
    out_p = r"C:\Users\YSB\OneDrive\Desktop\damda_final_processed"
    
    process_damda_dataset(img_p, lbl_p, out_p)