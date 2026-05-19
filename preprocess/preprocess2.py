import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def get_landmark_crops(image_path, crop_size=224):
    """
    MediaPipe 대신 이미지 해상도 비율 기반으로 
    부위별(이마, 코, 볼, 눈가) 고화질 크롭 수행
    """
    # 1. 한글 경로 대응 이미지 로드
    img_array = np.fromfile(image_path, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        return None, None
    
    h, w, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 2. CLAHE 적용 (텍스처 강조)
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)) 
    cl = clahe.apply(l)
    img_clahe = cv2.merge((cl, a, b))
    img_enhanced = cv2.cvtColor(img_clahe, cv2.COLOR_LAB2RGB)

    # 3. 부위별 좌표 비율 설정 (한국인 피부 데이터 표준 가이드 기반 정규화 좌표)
    # 이미지의 [세로 비율, 가로 비율] 입니다. 데이터 샘플을 보고 숫자를 미세 조정하세요.
    parts_ratio = {
        "Forehead": (0.25, 0.5),     # 이마 (상단 중앙)
        "Nose": (0.55, 0.5),         # 코 (중앙)
        "L_Cheek": (0.60, 0.35),     # 왼쪽 볼
        "R_Cheek": (0.60, 0.65),     # 오른쪽 볼
        "L_Eye_Corner": (0.45, 0.3),  # 왼쪽 눈가
        "R_Eye_Corner": (0.45, 0.7)   # 오른쪽 눈가
    }

    crops = {}
    for name, (ry, rx) in parts_ratio.items():
        # 비율을 픽셀 좌표로 변환
        cx, cy = int(rx * w), int(ry * h)
        
        # 4. 원본 화질 유지를 위해 Resize 없이 Crop만 수행
        x1 = max(0, cx - crop_size // 2)
        y1 = max(0, cy - crop_size // 2)
        x2 = min(w, x1 + crop_size)
        y2 = min(h, y1 + crop_size)
        
        crop = img_enhanced[y1:y2, x1:x2]
        
        # 224x224 규격 확인 (이미지 가장자리 처리용)
        if crop.shape[0] != crop_size or crop.shape[1] != crop_size:
            crop = cv2.resize(crop, (crop_size, crop_size))
            
        crops[name] = crop

    return img_rgb, crops

def visualize_test(origin, crops):
    """원본과 6개 부위 크롭 결과를 나란히 출력"""
    if crops is None: return

    plt.figure(figsize=(18, 8))
    
    # 1. 원본 표시
    plt.subplot(2, 4, 1)
    plt.imshow(origin)
    plt.title("Original Image")
    plt.axis('off')

    # 2. 각 부위별 크롭 이미지 표시
    for i, (name, crop_img) in enumerate(crops.items()):
        plt.subplot(2, 4, i + 2)
        plt.imshow(crop_img)
        plt.title(f"Part: {name}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# --- 테스트 실행부 ---
if __name__ == "__main__":
    # 이미지 폴더 경로 (r을 붙여 경로 에러 방지)
    input_dir = r"C:\Users\YSB\OneDrive\Desktop\pre_images"
    
    test_limit = 2  
    count = 0

    if not os.path.exists(input_dir):
        print(f"경로를 찾을 수 없습니다: {input_dir}")
    else:
        for filename in os.listdir(input_dir):
            if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                file_path = os.path.join(input_dir, filename)
                
                # 함수 호출 (구조 유지)
                origin, crops = get_landmark_crops(file_path, crop_size=224)
                
                if origin is not None and crops is not None:
                    print(f"[{count+1}] {filename} 부위별 크롭 완료 (비율 기반)")
                    visualize_test(origin, crops)
                    count += 1
                
                if count >= test_limit:
                    break

        print("모든 부위별 테스트가 완료되었습니다.")