import os

BASE_DIR = "data/img"
TARGET_FOLDERS = ["D", "P", "T"]

for folder in TARGET_FOLDERS:
    root_path = os.path.join(BASE_DIR, folder)

    for root, dirs, files in os.walk(root_path):
        for file in files:

            # D_, P_, T_ 로 시작하는 경우
            if file.startswith(("D_", "P_", "T_")):
                old_path = os.path.join(root, file)

                # 앞의 두 글자 제거
                new_name = file[2:]
                new_path = os.path.join(root, new_name)

                os.rename(old_path, new_path)
                print(f"Renamed: {old_path} -> {new_path}")

            # 이미 "_"로 시작하는 경우도 정리
            elif file.startswith("_"):
                old_path = os.path.join(root, file)
                new_name = file[1:]
                new_path = os.path.join(root, new_name)

                os.rename(old_path, new_path)
                print(f"Fixed leading _: {old_path} -> {new_path}")
