import json, cv2, os, errno
import numpy as np
import pandas as pd

def mkdir(path):
    if path == "": return
    try: os.makedirs(path, exist_ok=True)
    except OSError as e:
        if e.errno != errno.EEXIST: raise

# --- [사용자 경로 설정] ---
MY_IMAGES_DIR = r"E:\2026_exit_0_r\AI\data\train"          
TOTAL_JSON_DIR = r"E:\2026_exit_0_r\AI\data\total_labels"   
OUTPUT_ROOT = r"E:\2026_exit_0_r\AI\data\categorized_dataset" 

def get_core_id(filename):
    name_only = os.path.splitext(filename)[0]
    return name_only.split('__', 1)[1] if '__' in name_only else name_only

def calc_redness_score_and_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red1 = cv2.inRange(hsv, (0, 40, 40), (10, 255, 255))
    lower_red2 = cv2.inRange(hsv, (170, 40, 40), (180, 255, 255))
    red_mask = cv2.add(lower_red1, lower_red2)
    score = round((cv2.countNonZero(red_mask) / (img.shape[0] * img.shape[1])) * 100, 4)
    
    # 붉은색 마스크를 원본 이미지에 빨갛게 오버레이(시각화 강화를 위해)
    redness_img = img.copy()
    redness_img[red_mask > 0] = [0, 0, 255] # Red overlay
    
    return score, redness_img

def calc_roughness_score_and_img(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    score = round(laplacian.var(), 4)
    
    # 라플라시안 필터를 통과한 거친 질감을 눈에 잘 띄게 스케일링하여 저장
    abs_laplacian = cv2.convertScaleAbs(laplacian)
    roughness_img = cv2.applyColorMap(abs_laplacian, cv2.COLORMAP_JET)
    
    return score, roughness_img

def process_diagnostic_crop():
    all_imgs = [f for f in os.listdir(MY_IMAGES_DIR) if f.lower().endswith(('.jpg', '.png'))]
    my_image_map = {get_core_id(f): f for f in all_imgs}
    
    analysis_results = []
    print("🚀 JSON 메타데이터 추출 및 통계적 정규화 파이프라인 가동...")

    for root, dirs, files in os.walk(TOTAL_JSON_DIR):
        for json_file in files:
            if not json_file.endswith('.json'): continue
            json_id = os.path.splitext(json_file)[0]
            
            matched_img = None
            for core_id, origin_name in my_image_map.items():
                if core_id in json_id:
                    prefix = origin_name.split('__')[0]
                    if any(x in json_id for x in ['P_', 'D_', 'T_']) and prefix + '_' not in json_id: continue
                    matched_img = origin_name; break
            
            if matched_img:
                img = cv2.imread(os.path.join(MY_IMAGES_DIR, matched_img))
                if img is None: continue
                
                with open(os.path.join(root, json_file), "r", encoding='utf-8') as f:
                    data = json.load(f)

                # --- 1. JSON 메타데이터 추출 ---
                info = data.get("info", {})
                age = info.get("age", None)
                gender = info.get("gender", None)
                skin_type = info.get("skin_type", None)
                sensitive = info.get("sensitive", None)
                
                # 측정 장비 데이터 추출 (부위별로 키값이 다름)
                equipment = data.get("equipment")
                if not isinstance(equipment, dict):
                    equipment = {}
                
                # --- 2. 이미지 처리 및 크롭 ---
                bbox = data["images"].get("bbox")
                facepart = data["images"].get("facepart", 0)
                
                if bbox is None or len(bbox) < 4:
                    cropped = img
                    status_note = "FULL_IMAGE_RECOVERY"
                else:
                    bbox = list(map(int, bbox))
                    center_x, center_y = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
                    scale = 0.8 if facepart == 8 else (0.7 if facepart == 9 else 1.0)
                    crop_len = int(max(bbox[2]-bbox[0], bbox[3]-bbox[1]) / 2 * scale)
                    
                    y_s, y_e = max(int(center_y-crop_len), 0), min(int(center_y+crop_len), img.shape[0])
                    x_s, x_e = max(int(center_x-crop_len), 0), min(int(center_x+crop_len), img.shape[1])
                    cropped = img[y_s:y_e, x_s:x_e]
                    
                    if cropped.size == 0: 
                        cropped = img
                        status_note = "SIZE_ZERO_RECOVERY"
                    else:
                        status_note = "NORMAL_CROP"

                resized = cv2.resize(cropped, (256, 256))
                
                # 시각화된 이미지와 스코어를 분리해서 확보
                red_val, red_visual = calc_redness_score_and_mask(resized)
                rough_val, rough_visual = calc_roughness_score_and_img(resized)
                
                # 기본 정보 세팅
                row_data = {
                    "img_raw": resized, "img_red": red_visual, "img_rough": rough_visual,
                    "id": json_id, "part": facepart,
                    "age": age, "gender": gender, "skin_type": skin_type, "sensitive": sensitive,
                    "redness_raw": red_val, "roughness_raw": rough_val, "note": status_note
                }
                # Equipment 딕셔너리를 통째로 업데이트 (flatten)
                for k, v in equipment.items():
                    row_data[f"eq_{k}"] = v
                    
                analysis_results.append(row_data)
                print(f"📊 이미지 병합 중: {json_id} ({status_note})", end='\r')

    if not analysis_results: return
    
    # --- 3. 통계적 정규화 (Z-Score) 계산 ---
    print("\n\n🧮 Pandas 통계적 정규화(Z-Score) 프로세스 실행 중...")
    df = pd.DataFrame(analysis_results)
    
    # 이미지 계산 값 정규화
    red_mean = df['redness_raw'].mean()
    red_std = df['redness_raw'].std()
    df['redness_zscore'] = round((df['redness_raw'] - red_mean) / red_std, 4) if red_std != 0 else 0
    
    rough_mean = df['roughness_raw'].mean()
    rough_std = df['roughness_raw'].std()
    df['roughness_zscore'] = round((df['roughness_raw'] - rough_mean) / rough_std, 4) if rough_std != 0 else 0

    # 장비(Sensor) 센서 수치들에 대해서도 일괄 Z-Score 계산 시도
    eq_cols = [col for col in df.columns if str(col).startswith('eq_') and pd.api.types.is_numeric_dtype(df[col])]
    for col in eq_cols:
        col_mean = df[col].mean()
        col_std = df[col].std()
        df[f'{col}_zscore'] = round((df[col] - col_mean) / col_std, 4) if col_std != 0 else 0
    
    print("📂 Z-Score 기반 카테고리 폴더로 이미지 복사를 시작합니다...")
    
    # iterrows()는 numpy 배열(이미지) 보존을 간혹 깨뜨리므로 원래 리스트와 매칭하여 순회
    for idx, row in df.iterrows():
        part = int(row['part'])
        is_red = row['redness_zscore'] > 0
        is_rough = row['roughness_zscore'] > 0
        is_oily = row['skin_type'] in [2, 4]
        
        assigned_categories = []
        
        if part in [5, 6] and is_red:
            assigned_categories.append("Redness(홍조)")
        
        if part in [1, 2, 3, 4, 7] and is_rough:
            assigned_categories.append("Wrinkle(주름)")
            
        if part in [1, 5, 6, 9]:
            if (is_oily and row['roughness_zscore'] > -0.5) or is_rough:
                assigned_categories.append("Pore(모공)")
            
        if part in [0, 8] and is_rough:
            assigned_categories.append("Trouble(트러블)")
            
        if not assigned_categories:
            assigned_categories.append("Normal(양호)")
            
        assigned_categories = list(set(assigned_categories))
        original_data = analysis_results[idx] # 원본 이미지 배열 데이터
        
        for category in assigned_categories:
            save_path = os.path.join(OUTPUT_ROOT, category)
            mkdir(save_path)
            
            part_str = str(part).zfill(2)
            base_name = f"{row['id']}_part{part_str}"
            # 카테고리 특성에 맞는 보정(시각화) 이미지를 저장
            if category == "Redness(홍조)":
                target_img = original_data["img_red"]
            elif category in ["Wrinkle(주름)", "Pore(모공)", "Trouble(트러블)"]:
                target_img = original_data["img_rough"]
            else: # Normal(양호)
                target_img = original_data["img_raw"]
                
            # cv2.imwrite는 경로에 한글이 포함되면 작동하지 않으므로 numpy/imencode 방식으로 우회 저장
            is_success, im_buf_arr = cv2.imencode(".jpg", target_img)
            if is_success:
                im_buf_arr.tofile(os.path.join(save_path, f"{base_name}.jpg"))

    # CSV 저장 (이미지 객체는 뺌)
    csv_df = df.drop(columns=['img_raw', 'img_red', 'img_rough'])
    csv_df.to_csv(os.path.join(OUTPUT_ROOT, "categorized_report_with_stats.csv"), index=False)
    
    print(f"✅ 성공적으로 완료되었습니다! 시각화된(보정된) 이미지가 카테고리 폴더에 저장되었습니다.")

if __name__ == "__main__":
    process_diagnostic_crop()