import os
import cv2
import numpy as np
import torch
from torchvision import transforms as T
from torchvision.utils import save_image
from PIL import Image

INPUT_DIR = './data_sbP'
OUTPUT_DIR = './processed_data_cv2'
CROP_SIZE = 224

def get_cascade_path() -> str:
    cv2_data_path = os.path.join(os.path.dirname(cv2.__file__), 'data', 'haarcascade_frontalface_default.xml')
    if os.path.exists(cv2_data_path):
        return cv2_data_path

    cv2_data = getattr(cv2, 'data', None)
    if cv2_data is not None:
        haarcascades = getattr(cv2_data, 'haarcascades', None)
        if haarcascades:
            candidate = haarcascades + 'haarcascade_frontalface_default.xml'
            if os.path.exists(candidate):
                return candidate

    raise FileNotFoundError("❌ haarcascade_frontalface_default.xml 파일을 찾을 수 없습니다.")

cascade_path = get_cascade_path()
face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    raise FileNotFoundError("❌ 에러: 얼굴 인식 모델 로드 실패")

def get_resnet_preprocess() -> T.Compose:
    return T.Compose([
        T.Resize((CROP_SIZE, CROP_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def process_face_roi() -> None:
    preprocess = get_resnet_preprocess()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.png'))]
    print(f"[INFO] 총 이미지 수: {len(files)}")

    for file_name in files:
        img_array = np.fromfile(os.path.join(INPUT_DIR, file_name), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[SKIP] 이미지 로드 실패: {file_name}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) == 0:
            print(f"[SKIP] 얼굴 미검출: {file_name}")
            continue

        x, y, w, h = int(faces[0][0]), int(faces[0][1]), int(faces[0][2]), int(faces[0][3])
        face_roi = img[y:y+h, x:x+w]
        img_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)

        # 얼굴 크기에 비례한 크롭 반경 (최소 30px 보장)
        radius_y = max(30, int(h * 0.20))
        radius_x = max(30, int(w * 0.20))

        parts: dict[str, tuple[int, int]] = {
            "Forehead": (int(h * 0.2), int(w * 0.5)),
            "L_Cheek":  (int(h * 0.6), int(w * 0.3)),
            "R_Cheek":  (int(h * 0.6), int(w * 0.7)),
            "Nose":     (int(h * 0.5), int(w * 0.5))
        }

        label = file_name.split('_')[0]

        for part_name, (py, px) in parts.items():
            y1 = max(0, py - radius_y)
            y2 = min(h, py + radius_y)
            x1 = max(0, px - radius_x)
            x2 = min(w, px + radius_x)
            crop = cv2.resize(img_rgb[y1:y2, x1:x2], (CROP_SIZE, CROP_SIZE))

            pil_crop = Image.fromarray(crop.astype(np.uint8))
            tensor_img: torch.Tensor = preprocess(pil_crop) # type: ignore
            mean_val: float = tensor_img.mean().item()

            if 0.15 < mean_val < 0.85:
                save_path = os.path.join(OUTPUT_DIR, label)
                os.makedirs(save_path, exist_ok=True)
                save_image(tensor_img, os.path.join(save_path, f"{part_name}_{file_name}"))

    print("[INFO] 처리 완료!")

if __name__ == "__main__":
    process_face_roi()