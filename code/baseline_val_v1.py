import os
os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0" # RTX 4090급으로 인식하게 함
os.environ["CUDA_MODULE_LOADING"] = "LAZY" # 메모리 로딩 최적화
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from PIL import Image
import matplotlib.pyplot as plt

# ==========================================
# 1. 하이퍼파라미터 및 경로 설정
# ==========================================
BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 20
INPUT_SIZE = 512
SAMPLE_SIZE = 300  # 테스트용 샘플 개수

# ⭐ 수빈님의 실제 데이터셋 경로
ROOT_DIR = r'C:\2026_graduPrj\AI\damda_cropped_dataset'

# ==========================================
# 2. 커스텀 데이터셋
# ==========================================
class SkinValidationDataset(Dataset):
    def __init__(self, root_dir, sample_size=300, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_paths = []

        all_images = []
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"❌ 폴더를 찾을 수 없습니다: {root_dir}")

        for subdir in os.listdir(root_dir):
            subdir_path = os.path.join(root_dir, subdir)
            if os.path.isdir(subdir_path):
                for f in os.listdir(subdir_path):
                    if f.lower().endswith(('.jpg', '.png')):
                        all_images.append(os.path.join(subdir_path, f))

        if len(all_images) > sample_size:
            self.img_paths = random.sample(all_images, sample_size)
        else:
            self.img_paths = all_images

        print(f"✅ 데이터 탐색 완료! 총 {len(all_images)}개 중 {len(self.img_paths)}개를 샘플링했습니다.")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # 현재는 베이스라인 검증을 위한 더미(가짜) 라벨 사용
        dummy_class_label = torch.tensor(1)
        dummy_reg_label = torch.tensor([45.5])

        if self.transform:
            image = self.transform(image)
            
        return image, dummy_class_label, dummy_reg_label

# ==========================================
# 3. 모델 구축 (ResNet-50)
# ==========================================
class BaselineResNet(nn.Module):
    def __init__(self, num_classes=5):
        super(BaselineResNet, self).__init__()
        # 최신 문법 적용 (pretrained=True 대신 weights 사용)
        self.backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.classifier = nn.Linear(num_ftrs, num_classes)
        self.regressor = nn.Linear(num_ftrs, 1)

    def forward(self, x):
        features = self.backbone(x)
        class_out = self.classifier(features)
        reg_out = self.regressor(features)
        return class_out, reg_out

# ==========================================
# 4. 시각화 함수
# ==========================================
def plot_metrics(loss_hist, mae_hist):
    epochs = range(1, len(loss_hist) + 1)
    
    plt.figure(figsize=(12, 5))

    # Loss 그래프 (에포크 1의 오차가 크므로 로그 스케일 권장)
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss_hist, 'r-o', label='Train Loss')
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.yscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()

    # MAE 그래프
    plt.subplot(1, 2, 2)
    plt.plot(epochs, mae_hist, 'b-s', label='Train MAE')
    plt.title('Training MAE per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('MAE Value')
    plt.grid(True, alpha=0.5)
    plt.legend()

    plt.tight_layout()
    plt.savefig('baseline_training_result.png')
    print("\n📊 그래프가 'baseline_training_result.png'로 저장되었습니다.")
    plt.show()

# ==========================================
# 5. 메인 학습 루프
# ==========================================
def train_baseline():
# ⭐ CPU에서 CUDA(GPU)로 변경! 
    # 사용 가능할 때만 CUDA를 쓰고, 아니면 CPU로 자동 전환되는 안전한 코드입니다.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu") # 에러 안나는지 한번씩 확인해보기_sbP
    
    print(f"🚀 학습 시작! 사용 기기: {device}")
    if device.type == 'cuda':
        print(f"✨ 현재 사용 중인 GPU: {torch.cuda.get_device_name(0)}")

    # 모델을 GPU 메모리로 보냅니다.
    model = BaselineResNet(num_classes=5).to(device)

    transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = SkinValidationDataset(root_dir=ROOT_DIR, sample_size=SAMPLE_SIZE, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = BaselineResNet(num_classes=5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()
    mae_metric = nn.L1Loss()

    # 결과 저장을 위한 리스트
    loss_history = []
    mae_history = []

    print("📉 학습 진행 중...")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss, total_mae = 0.0, 0.0
        
        for images, cls_labels, reg_labels in dataloader:
            images, cls_labels, reg_labels = images.to(device), cls_labels.to(device), reg_labels.to(device)
            
            optimizer.zero_grad()
            cls_preds, reg_preds = model(images)
            
            loss_cls = criterion_cls(cls_preds, cls_labels)
            loss_reg = criterion_reg(reg_preds, reg_labels)
            loss = loss_cls + loss_reg
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_mae += mae_metric(reg_preds, reg_labels).item()
            
        avg_loss = total_loss / len(dataloader)
        avg_mae = total_mae / len(dataloader)
        
        loss_history.append(avg_loss)
        mae_history.append(avg_mae)
        
        print(f"  [Epoch {epoch+1:02d}/{EPOCHS}] Loss: {avg_loss:.4f} | MAE: {avg_mae:.4f}")

    # 학습 완료 후 시각화
    plot_metrics(loss_history, mae_history)

if __name__ == "__main__":
    train_baseline()