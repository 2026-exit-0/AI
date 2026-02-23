# 이미지 개수와 라벨 개수가 일치하는지 확인하는 스크립트
import os

IMG_BASE = "data/img"
LABEL_BASE = "data/label"
TARGET_FOLDERS = ["D", "P", "T"]

for folder in TARGET_FOLDERS:

    print(f"\n===== {folder} =====")

    img_root = os.path.join(IMG_BASE, folder)
    label_root = os.path.join(LABEL_BASE, folder)

    # --------------------------
    # 1️⃣ 이미지 prefix 수집 (sub 포함)
    # --------------------------
    image_prefix_set = set()
    total_image_count = 0

    for root, dirs, files in os.walk(img_root):
        for file in files:
            if file.lower().endswith((".jpg", ".png")):
                total_image_count += 1
                prefix = file[:4]
                image_prefix_set.add(prefix)

    # --------------------------
    # 2️⃣ 라벨 prefix 폴더 수집
    # --------------------------
    label_prefix_set = set()

    if os.path.exists(label_root):
        for folder_name in os.listdir(label_root):
            sub_path = os.path.join(label_root, folder_name)
            if os.path.isdir(sub_path):
                label_prefix_set.add(folder_name)

    # --------------------------
    # 3️⃣ 결과 출력
    # --------------------------
    print(f"총 이미지 파일 개수: {total_image_count}")
    print(f"이미지 prefix 개수: {len(image_prefix_set)}")
    print(f"라벨 prefix 폴더 개수: {len(label_prefix_set)}")

    # --------------------------
    # 4️⃣ 불일치 분석
    # --------------------------
    only_in_image = image_prefix_set - label_prefix_set
    only_in_label = label_prefix_set - image_prefix_set

    print(f"이미지에만 있는 prefix 개수: {len(only_in_image)}")
    print(f"라벨에만 있는 prefix 개수: {len(only_in_label)}")

    if only_in_image:
        print("이미지에만 있는 prefix (최대 10개):", sorted(list(only_in_image))[:10])

    if only_in_label:
        print("라벨에만 있는 prefix (최대 10개):", sorted(list(only_in_label))[:10])
