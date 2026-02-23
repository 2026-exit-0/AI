import os
import shutil

IMG_BASE = "data/img"
LABEL_BASE = "data/label"
TARGET_FOLDERS = ["D", "P", "T"]

for folder in TARGET_FOLDERS:

    print(f"\n===== Processing {folder} =====")

    img_root = os.path.join(IMG_BASE, folder)
    label_root = os.path.join(LABEL_BASE, folder)

    # ==========================
    # 1️⃣ 이미지 정보 수집
    # ==========================
    image_set = set()
    valid_prefix = set()

    for root, dirs, files in os.walk(img_root):
        for file in files:
            if file.lower().endswith((".jpg", ".png")):
                image_set.add(file)
                valid_prefix.add(file[:4])  # 앞 4자리

    # ==========================
    # 2️⃣ 1차 정리: 폴더 단위 삭제
    # ==========================
    for subfolder in os.listdir(label_root):

        sub_path = os.path.join(label_root, subfolder)

        if not os.path.isdir(sub_path):
            continue

        if subfolder not in valid_prefix:
            print(f"[1차 삭제] 폴더 삭제: {sub_path}")
            shutil.rmtree(sub_path)

    # ==========================
    # 3️⃣ 2차 정리: 파일 단위 삭제
    # ==========================
    for root, dirs, files in os.walk(label_root):
        for file in files:

            if not file.endswith(".json"):
                continue

            json_path = os.path.join(root, file)

            name_without_ext = file[:-5]  # .json 제거

            if "_" not in name_without_ext:
                continue

            # 마지막 _00 제거
            image_name = name_without_ext.rsplit("_", 1)[0] + ".jpg"

            if image_name not in image_set:
                print(f"[2차 삭제] JSON 삭제: {json_path}")
                os.remove(json_path)

print("\n===== 정리 완료 =====")
