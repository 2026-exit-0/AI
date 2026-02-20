import torchvision.transforms as T

def get_resnet_preprocess(size=224, is_training=True):
    """
    ResNet-50 전처리 파이프라인
    - size: 224 (표준) 또는 448 (피부 질환 등 미세 특징이 중요할 때 권장)
    - is_training: 학습 데이터셋인 경우 데이터 증강(Augmentation) 적용
    """
    transforms = [
        T.Resize((size, size)), # 이미지 사이즈 통일
    ]
    
    if is_training:
        # 학습 시에는 모델의 일반화 성능을 높이기 위해 랜덤 변환 추가
        transforms.extend([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(15),
        ])
    
    transforms.extend([
        T.ToTensor(),           # 텐서 변환 및 0~1 스케일링
        T.Normalize(            # ImageNet 표준 정규화
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return T.Compose(transforms)

if __name__ == "__main__":
    # 작성한 전처리 함수 불러오기
  preprocess = get_resnet_preprocess(size=224, is_training=True)
  print("전처리 파이프라인이 성공적으로 생성되었습니다!")
    
    # 실제 이미지 하나 테스트해보기 (경로는 본인 데이터 하나로 수정)
    # from PIL import Image
    # img = Image.open("../data/train/sample_image.jpg")
    # result = preprocess(img)
    # print(f"전처리 후 텐서 크기: {result.shape}") 
    # 결과가 torch.Size([3, 224, 224])로 나오면 완벽!