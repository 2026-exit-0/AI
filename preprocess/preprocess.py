import cv2
import numpy as np
import os

def save_skin_crops(input_dir, output_root):
    # 1. 저장할 폴더 구조 생성
    parts = ["Forehead", "Nose", "L_Cheek", "R_Cheek", "L_Eye", "R_Eye", "Chin"]
    for part in parts:
        os.makedirs(os.path.join(output_root, part), exist_ok=True)

    # 2. 이미지 처리 루프
    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue
            
        file_path = os.path.join(input_dir, filename)
        img_array = np.fromfile(file_path, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None: continue
        
        h, w, _ = img.shape
        # CLAHE로 텍스처 강조
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        img_enhanced = cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2RGB)

        # 3. 각도별 좌표 설정 (명세서의 13가지 각도 대응)
        # 기본값: 정면(Front)
        ratios = {
            "Forehead": (0.25, 0.5), "Nose": (0.5, 0.5), 
            "L_Cheek": (0.6, 0.35), "R_Cheek": (0.6, 0.65),
            "L_Eye": (0.45, 0.35), "R_Eye": (0.45, 0.65), "Chin": (0.8, 0.5)
        }

        # 파일명 규칙에 따른 각도 보정 (AI Hub 파일명 예시 기준)
        fn = filename.upper()
        if "L15" in fn or "L30" in fn or "LEFT" in fn: # 좌측면
            ratios["L_Cheek"] = (0.6, 0.45) # 볼 중심이 우측(중앙쪽)으로 이동
            ratios["Forehead"] = (0.25, 0.4)
        elif "R15" in fn or "R30" in fn or "RIGHT" in fn: # 우측면
            ratios["R_Cheek"] = (0.6, 0.55)
            ratios["Forehead"] = (0.25, 0.6)
        elif "UP" in fn or "상" in fn: # 상향
            ratios["Forehead"] = (0.15, 0.5)
            ratios["Chin"] = (0.7, 0.5)

        # 4. 부위별 크롭 및 저장
        crop_size = 224
        for part_name, (ry, rx) in ratios.items():
            cx, cy = int(rx * w), int(ry * h)
            x1, y1 = max(0, cx - crop_size//2), max(0, cy - crop_size//2)
            crop = img_enhanced[y1:y1+crop_size, x1:x1+crop_size]
            
            # 크기 미달 시 리사이즈
            if crop.shape[0] < crop_size or crop.shape[1] < crop_size:
                crop = cv2.resize(crop, (crop_size, crop_size))

            # 결과 저장 (파일명에 원본 이름 포함)
            save_name = f"{os.path.splitext(filename)[0]}_{part_name}.jpg"
            save_path = os.path.join(output_root, part_name, save_name)
            
            # 한글 경로 대응 저장
            is_success, buffer = cv2.imencode(".jpg", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            if is_success:
                with open(save_path, "wb") as f:
                    f.write(buffer)

    print(f"모든 이미지의 부위별 분류 저장이 완료되었습니다. 위치: {output_root}")

# --- 실행 ---
if __name__ == "__main__":
    input_folder = r"C:\Users\YSB\OneDrive\Desktop\pre_images"
    output_folder = r"C:\Users\YSB\OneDrive\Desktop\damda_processed_data"
    save_skin_crops(input_folder, output_folder)