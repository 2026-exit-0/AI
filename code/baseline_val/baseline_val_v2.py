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
import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image

class DamdaSkinDataset(Dataset):
    def __init__(self, json_root_dir, img_root_dir=None, transform=None):
        """
        json_root_dir: JSON 파일들이 들어있는 최상위 폴더 (예: 'TL')
        img_root_dir: 이미지 파일들이 들어있는 최상위 폴더 (JSON과 같은 폴더면 None으로 둠)
        """
        self.json_root_dir = "C:\2026_graduPrj\AI\TL"
        self.img_root_dir = img_root_dir or json_root_dir
        self.transform = transform
        self.data_infos = []

        # 1. os.walk를 이용해 하위 폴더를 다 뒤져서 모든 .json 파일 경로를 찾습니다.
        for root, dirs, files in os.walk(json_root_dir):
            for file in files:
                if file.endswith('.json'):
                    json_path = os.path.join(root, file)
                    self.data_infos.append(json_path)
                    
        print(f"✅ 정답지 로드 완료! 총 {len(self.data_infos)}개의 JSON 라벨을 찾았습니다.")

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        json_path = self.data_infos[idx]
        
        # 2. JSON 파일 읽기
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # 3. 이미지 파일 이름 가져오기 ("0003_02_F.jpg")
        img_name = data['info']['filename']
        
        # 4. 이미지 실제 경로 완성하기
        # JSON 파일이 있는 폴더와 동일한 위치에 이미지가 있다고 가정합니다.
        json_dir = os.path.dirname(json_path)
        img_path = os.path.join(json_dir, img_name)
        
        # 만약 이미지가 다른 상위 폴더(예: 'TS')에 있다면 경로를 바꿔치기 합니다.
        if self.json_root_dir != self.img_root_dir:
            img_path = img_path.replace(self.json_root_dir, self.img_root_dir)

        # 5. 이미지 로드 및 전처리
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            # 혹시 이미지가 누락된 경우를 대비한 안전장치
            print(f"⚠️ 이미지를 찾을 수 없습니다: {img_path}")
            image = Image.new('RGB', (512, 512), (0, 0, 0)) # 검은색 빈 이미지 대체
            
        if self.transform:
            image = self.transform(image)
            
        # 6. ⭐ 진짜 정답(Label) 가져오기 ⭐
        # 분류 정답: 피부 타입 (skin_type)
        skin_type = data['info'].get('skin_type', 0)
        class_label = torch.tensor(skin_type, dtype=torch.long)
        
        # 회귀 정답: 색소 침착 개수 (pigmentation_count)
        # 만약 데이터가 null 이거나 키값이 없을 경우 0으로 처리하는 안전장치
        equipment_data = data.get('equipment', {})
        pigmentation = equipment_data.get('pigmentation_count')
        if pigmentation is None:
            pigmentation = 0.0
            
        reg_label = torch.tensor([pigmentation], dtype=torch.float32)
        
        return image, class_label, reg_label

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
    dataset = DamdaSkinDataset(ROOT_DIR, SAMPLE_SIZE, transform)
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