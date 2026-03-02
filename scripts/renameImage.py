# import os

# BASE_DIR = "data/img"
# TARGET_FOLDERS = ["D", "P", "T"]

# for folder in TARGET_FOLDERS:
#     root_path = os.path.join(BASE_DIR, folder)

#     for root, dirs, files in os.walk(root_path):
#         for file in files:

#             # D_, P_, T_ 로 시작하는 경우
#             if file.startswith(("D_", "P_", "T_")):
#                 old_path = os.path.join(root, file)

#                 # 앞의 두 글자 제거
#                 new_name = file[2:]
#                 new_path = os.path.join(root, new_name)

#                 os.rename(old_path, new_path)
#                 print(f"Renamed: {old_path} -> {new_path}")

#             # 이미 "_"로 시작하는 경우도 정리
#             elif file.startswith("_"):
#                 old_path = os.path.join(root, file)
#                 new_name = file[1:]
#                 new_path = os.path.join(root, new_name)

#                 os.rename(old_path, new_path)
#                 print(f"Fixed leading _: {old_path} -> {new_path}")

import os
import shutil
from tqdm import tqdm

BASE_DIR = "data/cropped_img"

PART_MAP = {
    "00": "full_face",
    "01": "forehead",
    "02": "between_the_eyebrows",
    "03": "right_eye",
    "04": "left_eye",
    "05": "right_cheek",
    "06": "left_cheek",
    "07": "mouth",
    "08": "chin"
}

EQUIPMENT_LIST = ["D", "P", "T"]

for equipment in EQUIPMENT_LIST:

    print(f"\n===== {equipment} 처리 시작 =====")

    equipment_path = os.path.join(BASE_DIR, equipment)

    # 현재 장비 폴더 안 모든 파일 재귀 탐색
    for root, dirs, files in os.walk(equipment_path):
        for file in tqdm(files):

            if not file.lower().endswith((".jpg", ".png")):
                continue

            file_path = os.path.join(root, file)

            # 파일명에서 부위 코드 추출
            name_without_ext = os.path.splitext(file)[0]
            parts = name_without_ext.split("_")

            if len(parts) < 4:
                print(f"[SKIP] 형식 이상: {file}")
                continue

            part_code = parts[-1]

            if part_code not in PART_MAP:
                print(f"[SKIP] 정의되지 않은 코드: {file}")
                continue

            part_name = PART_MAP[part_code]

            # 목적지 폴더 생성
            target_dir = os.path.join(equipment_path, part_name)
            os.makedirs(target_dir, exist_ok=True)

            target_path = os.path.join(target_dir, file)

            # 이미 같은 위치면 건너뜀
            if file_path == target_path:
                continue

            shutil.move(file_path, target_path)

print("\n구조 재정렬 완료")