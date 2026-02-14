import cv2
import os
import cv2

# 입력 경로를 실제 폴더명인 data_sbP로 변경
input_dir = './data_sbP' 
output_dir = './preprocessed_images'

# 폴더 존재 확인 (오타 방지)
if not os.path.exists(input_dir):
    print(f"경고: '{input_dir}' 폴더를 찾을 수 없습니다. 폴더명을 다시 확인해주세요.")
else:
    file_list = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for file_name in file_list:
        # 2. 파일명 분석 (D__1019_01_Fb.jpg -> ['D', '', '1019', '01', 'Fb.jpg'])
        # 언더바가 두 개인 경우를 대비해 split('_') 결과에서 빈 문자열은 제외
        parts = [p for p in file_name.split('_') if p]
        
        if len(parts) < 4:
            continue
            
        data_type = parts[0]    # D, P, T 등
        sample_id = parts[1]    # 1019, 0025 등
        angle = parts[3].split('.')[0]  # Fb, L, F 등 (확장자 제거)

        # 3. 데이터 타입별 저장 폴더 생성 (예: ./processed_data/D/)
        save_path = os.path.join(output_dir, data_type)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # 4. 이미지 처리
        img = cv2.imread(os.path.join(input_dir, file_name))
        if img is None: continue

        # [전처리 예시: 정방형 리사이징 및 가우시안 블러로 노이즈 제거]
        img_resized = cv2.resize(img, (512, 512))
        # img_blurred = cv2.GaussianBlur(img_resized, (5, 5), 0)

        # 5. 새로운 이름으로 저장 (혹은 원본 이름 유지)
        # 예: D_1019_Fb_resized.jpg
        new_name = f"{data_type}_{sample_id}_{angle}.jpg"
        cv2.imwrite(os.path.join(save_path, new_name), img_resized)

    print(f"총 {len(file_list)}개의 이미지 전처리가 완료되었습니다.")