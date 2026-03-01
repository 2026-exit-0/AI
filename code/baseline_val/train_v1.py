#전처리가 완료된 모든 데이터 셋을 활용한 학습(300개는 코드 추가예정)

import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from PIL import Image
import matplotlib.pyplot as plt

# GPU 모니터링 라이브러리 (선택)
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

# ==========================================
# 1. 환경 설정 및 하이퍼파라미터
# ==========================================
os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"  # RTX 5060 호환
os.environ["CUDA_MODULE_LOADING"] = "LAZY"

BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 30  # 검증을 위해 에폭 수를 조금 늘렸습니다.
INPUT_SIZE = 512

# ⭐ 수빈님의 실제 경로에 맞게 수정해 주세요!
JSON_DIR = r'C:\2026_graduPrj\AI\TL'  # 원본 JSON 라벨이 있는 최상위 폴더
IMG_DIR = r'C:\2026_graduPrj\AI\damda_cropped_dataset' # 방금 전처리 완료한 크롭 이미지 폴더

# ==========================================
# 2. 진짜 정답지를 읽는 Dataset 클래스 (Step 1)
# ==========================================
class DamdaSkinDataset(Dataset):
    def __init__(self, json_dir, img_dir, transform=None):
        self.transform = transform
        self.samples = []

        # 1. 전처리된 이미지 파일들을 미리 딕셔너리로 쫙 찾아둡니다.
        img_dict = {}
        for root, _, files in os.walk(img_dir):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    basename = os.path.splitext(f)[0]
                    img_dict[basename] = os.path.join(root, f)

        # 2. JSON 파일을 돌면서 이름이 똑같은 이미지가 있으면 짝지어 줍니다.
        for root, _, files in os.walk(json_dir):
            for f in files:
                if f.endswith('.json'):
                    basename = os.path.splitext(f)[0]
                    if basename in img_dict:
                        self.samples.append((os.path.join(root, f), img_dict[basename]))

        print(f"✅ 데이터 준비 완료: 총 {len(self.samples)}쌍의 (이미지+정답)이 매칭되었습니다.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        json_path, img_path = self.samples[idx]
        
        # 이미지 로드
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        # JSON 정답 로드
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # [분류 정답] 피부 타입 (0~4 등급이라 가정)
        skin_type = data.get('info', {}).get('skin_type', 0)
        class_label = torch.tensor(skin_type, dtype=torch.long)
        
        # [회귀 정답] 수분량 (Moisture)
        # JSON 구조에 따라 조금 다를 수 있으나, 보통 equipment 안에 있습니다.
        equip_data = data.get('equipment', {})
        if not isinstance(equip_data, dict): equip_data = {}
        moisture = data.get('moisture') or next((v for k, v in equip_data.items() if 'moisture' in k.lower()), 0.0)
        
        try:
            moisture = float(moisture)
        except:
            moisture = 0.0
            
        reg_label = torch.tensor([moisture], dtype=torch.float32)

        return image, class_label, reg_label

# ==========================================
# 3. 모델 구조 (기존 유지)
# ==========================================
class BaselineResNet(nn.Module):
    def __init__(self, num_classes=6):
        super(BaselineResNet, self).__init__()
        self.backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Linear(num_ftrs, num_classes)
        self.regressor = nn.Linear(num_ftrs, 1)

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features), self.regressor(features)

# ==========================================
# 4. Train / Validation 분리 및 메인 학습 (Step 2)
# ==========================================
def run_train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 학습 시작 기기: {device}")

    transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 1. 전체 데이터셋 불러오기
    full_dataset = DamdaSkinDataset(JSON_DIR, IMG_DIR, transform)
    
    # 2. ⭐ Train(80%)과 Validation(20%) 분리 ⭐
    total_size = len(full_dataset)
    val_size = int(total_size * 0.2)
    train_size = total_size - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"📊 학습용(Train): {train_size}개 | 검증용(Val): {val_size}개")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = BaselineResNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()

    history = {'train_loss': [], 'val_loss': [], 'train_mae': [], 'val_mae': []}

    print("-" * 75)
    print(f"{'Epoch':^7} | {'T-Loss':^8} | {'V-Loss':^8} | {'T-MAE':^8} | {'V-MAE':^8}")
    print("-" * 75)

    for epoch in range(EPOCHS):
        # ------------------ [Train Phase] ------------------
        model.train()
        t_loss, t_mae = 0.0, 0.0
        
        for images, c_labels, r_labels in train_loader:
            images, c_labels, r_labels = images.to(device), c_labels.to(device), r_labels.to(device)
            
            optimizer.zero_grad()
            c_out, r_out = model(images)
            loss = criterion_cls(c_out, c_labels) + criterion_reg(r_out, r_labels)
            loss.backward()
            optimizer.step()

            t_loss += loss.item()
            t_mae += torch.abs(r_out - r_labels).mean().item()
            
        avg_t_loss = t_loss / len(train_loader)
        avg_t_mae = t_mae / len(train_loader)

        # ------------------ [Validation Phase] ------------------
        model.eval() # 모델을 평가 모드로 전환 (Dropout, BatchNorm 동작 변경)
        v_loss, v_mae = 0.0, 0.0
        
        with torch.no_grad(): # 미분 계산을 멈춰서 메모리를 아낌
            for images, c_labels, r_labels in val_loader:
                images, c_labels, r_labels = images.to(device), c_labels.to(device), r_labels.to(device)
                c_out, r_out = model(images)
                loss = criterion_cls(c_out, c_labels) + criterion_reg(r_out, r_labels)
                
                v_loss += loss.item()
                v_mae += torch.abs(r_out - r_labels).mean().item()
                
        avg_v_loss = v_loss / len(val_loader)
        avg_v_mae = v_mae / len(val_loader)

        # 기록 및 출력
        history['train_loss'].append(avg_t_loss)
        history['val_loss'].append(avg_v_loss)
        history['train_mae'].append(avg_t_mae)
        history['val_mae'].append(avg_v_mae)

        print(f"{epoch+1:^7} | {avg_t_loss:^8.4f} | {avg_v_loss:^8.4f} | {avg_t_mae:^8.4f} | {avg_v_mae:^8.4f}")

    # ==========================================
    # 5. 결과 시각화 (Train vs Validation 비교)
    # ==========================================
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', color='red')
    plt.plot(history['val_loss'], label='Val Loss', color='blue', linestyle='--')
    plt.title('Loss (Train vs Val)')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_mae'], label='Train MAE', color='red')
    plt.plot(history['val_mae'], label='Val MAE', color='blue', linestyle='--')
    plt.title('MAE (Train vs Val)')
    plt.xlabel('Epoch'); plt.ylabel('MAE')
    plt.legend()

    plt.tight_layout()
    plt.savefig('train_val_result.png')
    print("\n📊 'train_val_result.png'에 결과가 저장되었습니다. 과적합 여부를 확인해 보세요!")

if __name__ == "__main__":
    run_train()