import os
import json
import cv2
import re
from tqdm import tqdm

# ================= [경로 설정] =================
MY_IMAGES_DIR = './data_sbP'    
TOTAL_LABEL_ROOT = './TL'    
SAVE_ROOT = './cropped_dataset'        
# ===============================================

def get_match_key(filename):
    match = re.search(r'(\d{4}_\d{2})', filename)
    return match.group(1) if match else None

def main():
    if not os.path.exists(SAVE_ROOT):
        os.makedirs(SAVE_ROOT)

    # 1. 내 이미지 목록 (키: 0003_01)
    img_map = {get_match_key(f): f for f in os.listdir(MY_IMAGES_DIR) if get_match_key(f)}
    
    # 2. JSON 파일 경로 싹 다 모으기
    json_paths = []
    for root, dirs, files in os.walk(TOTAL_LABEL_ROOT):
        for f in files:
            if f.lower().endswith('.json'):
                json_paths.append(os.path.join(root, f))
    
    print(f"🚀 {len(img_map)}개의 이미지에서 부위별 조각을 찾습니다...")

    success_count = 0

    # 3. JSON을 하나씩 까보면서 내 이미지랑 맞으면 바로 자르기!
    for j_path in tqdm(json_paths, desc="부위별 크롭 중"):
        j_name = os.path.basename(j_path)
        j_key = get_match_key(j_name)
        
        # 내 이미지 목록에 있는 번호라면?
        if j_key in img_map:
            try:
                with open(j_path, "r", encoding='utf-8') as f:
                    data = json.load(f)
                
                img_info = data.get("images", {})
                bbox = img_info.get("bbox")
                facepart = img_info.get("facepart")

                # 💡 핵심: 좌표가 없거나, 부위가 0번(얼굴 전체)이면 무조건 패스!
                if not bbox or str(facepart) == '0' or facepart == 0:
                    continue

                # 이미지 불러오기 (매번 불러오지만 이게 제일 안전하고 확실함)
                img_name = img_map[j_key]
                img_path = os.path.join(MY_IMAGES_DIR, img_name)
                img_obj = cv2.imread(img_path)
                if img_obj is None: continue

                # 좌표 계산
                x1, y1, x2, y2 = map(int, bbox)
                if x2 - x1 <= 0 or y2 - y1 <= 0: continue

                center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                side_length = max(x2 - x1, y2 - y1) / 2
                
                h, w, _ = img_obj.shape
                cy1, cy2 = max(int(center_y - side_length), 0), min(int(center_y + side_length), h)
                cx1, cx2 = max(int(center_x - side_length), 0), min(int(center_x + side_length), w)

                cropped = img_obj[cy1:cy2, cx1:cx2]
                if cropped.size == 0: continue

                # 256x256 리사이즈
                final_img = cv2.resize(cropped, (256, 256))

                # 폴더 생성 및 저장
                part_dir = os.path.join(SAVE_ROOT, f"part_{str(facepart).zfill(2)}")
                os.makedirs(part_dir, exist_ok=True)
                
                save_name = f"{os.path.splitext(img_name)[0]}_p{facepart}.jpg"
                cv2.imwrite(os.path.join(part_dir, save_name), final_img)
                
                success_count += 1

            except Exception as e:
                # 에러나도 멈추지 않고 다음 파일로!
                continue

    print(f"\n✨ 진짜 성공! 총 {success_count}개의 부위별(이마, 볼 등) 조각이 저장되었습니다!")

if __name__ == "__main__":
    main()