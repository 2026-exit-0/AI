import os
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from PIL import Image
import matplotlib.pyplot as plt

# GPU 점유율 확인용 라이브러리
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

# ==========================================
# 1. 환경 설정 및 하이퍼파라미터
# ==========================================
# RTX 5060(Blackwell) 최적화 설정
os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0" 
os.environ["CUDA_MODULE_LOADING"] = "LAZY"

BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 20
INPUT_SIZE = 512
SAMPLE_SIZE = 300 

# 데이터셋 경로 (수빈님의 실제 경로)
ROOT_DIR = r'C:\2026_graduPrj\AI\damda_cropped_dataset'

# ==========================================
# 2. 유틸리티 함수 (GPU 모니터링)
# ==========================================
def get_gpu_status():
    """현재 GPU의 점유율(%)과 메모리(MB)를 반환합니다."""
    if not PYNVML_AVAILABLE or not torch.cuda.is_available():
        return 0, 0
    
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return util.gpu, info.used / 1024**2
    except:
        return 0, 0

# ==========================================
# 3. 데이터셋 및 모델 정의 (기존 구조 유지)
# ==========================================
class SkinValidationDataset(Dataset):
    def __init__(self, root_dir, sample_size=300, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_paths = []

        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"❌ 경로를 찾을 수 없습니다: {root_dir}")

        all_images = []
        for subdir in os.listdir(root_dir):
            subdir_path = os.path.join(root_dir, subdir)
            if os.path.isdir(subdir_path):
                for f in os.listdir(subdir_path):
                    if f.lower().endswith(('.jpg', '.png')):
                        all_images.append(os.path.join(subdir_path, f))

        self.img_paths = random.sample(all_images, min(len(all_images), sample_size))
        print(f"✅ 데이터 로드 완료: {len(self.img_paths)}개 샘플링됨.")

    def __len__(self): return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform: image = self.transform(image)
        return image, torch.tensor(1), torch.tensor([45.5]) # 더미 라벨

class BaselineResNet(nn.Module):
    def __init__(self, num_classes=5):
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
# 4. 메인 학습 루틴
# ==========================================
def run_train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 학습 시작 기기: {device} ({torch.cuda.get_device_name(0)})")

    # 모델 및 데이터 준비
    model = BaselineResNet().to(device)
    transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = SkinValidationDataset(ROOT_DIR, SAMPLE_SIZE, transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()

    loss_history, mae_history = [], []
    start_time = time.time()

    print(f"{'Epoch':^7} | {'Time':^8} | {'GPU %':^7} | {'VRAM':^10} | {'Loss':^8} | {'MAE':^8}")
    print("-" * 60)

    for epoch in range(EPOCHS):
        epoch_start = time.time()
        model.train()
        total_loss, total_mae = 0.0, 0.0

        for images, cls_labels, reg_labels in dataloader:
            images, cls_labels, reg_labels = images.to(device), cls_labels.to(device), reg_labels.to(device)
            
            optimizer.zero_grad()
            c_out, r_out = model(images)
            loss = criterion_cls(c_out, cls_labels) + criterion_reg(r_out, reg_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_mae += torch.abs(r_out - reg_labels).mean().item()

        # 에폭 통계 계산
        epoch_time = time.time() - epoch_start
        gpu_util, vram_used = get_gpu_status()
        avg_loss = total_loss / len(dataloader)
        avg_mae = total_mae / len(dataloader)

        loss_history.append(avg_loss)
        mae_history.append(avg_mae)

        print(f"{epoch+1:^7} | {epoch_time:^7.2f}s | {gpu_util:^6}% | {vram_used:^7.0f} MB | {avg_loss:^8.4f} | {avg_mae:^8.4f}")

    print(f"\n✨ 전체 학습 시간: {(time.time()-start_time)/60:.2f}분")
    
    # 결과 그래프 저장
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1); plt.plot(loss_history); plt.title('Loss')
    plt.subplot(1,2,2); plt.plot(mae_history); plt.title('MAE')
    plt.savefig('training_log.png')
    print("📊 'training_log.png'에 결과가 저장되었습니다.")

if __name__ == "__main__":
    run_train()