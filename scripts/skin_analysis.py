"""
피부 분석 항목별 이미지 전처리 모듈
======================================

[구현 항목]
  1.  유분       - 상위 2% 고밝기 픽셀 감지
  2.  주름       - HSL S채널 기반 깊이 측정 + 길이 필터링
  3.  피지       - UVL 반응 색상(주홍/노란/녹색) 감지
  4.  모공       - Morphology Opening + VSL 깊이 측정
  5.  기미·잡티  - 절대 밝기 + 주변 대비 이중 가중치
  6.  색소침착   - UV 특성 모사, 갈색 멜라닌 검출
  7.  다크서클   - 눈 하단 영역 어두움/색상 분석
  8.  광채       - VSL 기법 모사, 고반사 영역 감지
  9.  홍조       - R/G 채널 대비 붉기 지수 계산
  10. 번들거림   - 고밝기 specular highlight 감지
  11. 칙칙함     - 피부 톤 대비 광채 부족 영역 감지
  12. 피부색     - 부위별 RGB 측정 및 피부톤 분류

[제외 항목] 수분(센서), 각질(테이프), 탄력도(주름+칙칙함 종합 → 별도 계산)

[사용법]
  analyzer = SkinAnalyzer(img_bgr)
  results  = analyzer.run_all()
  vis      = analyzer.visualize(results)
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")


# ============================================================
# 공통 유틸
# ============================================================

def to_gray(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

def to_lab(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

def to_hls(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HLS)

def to_hsv(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

def normalize_score(value: float, min_v: float = 0.0, max_v: float = 1.0) -> float:
    """0~100 점수로 정규화"""
    return float(np.clip((value - min_v) / (max_v - min_v + 1e-8) * 100, 0, 100))


@dataclass
class AnalysisResult:
    """각 항목의 분석 결과"""
    score: float          # 0~100 지수
    mask: np.ndarray      # 감지 영역 마스크 (0/255)
    detail: Dict = field(default_factory=dict)   # 세부 수치


# ============================================================
# 1. 유분 (Sebum / Oil)
# ============================================================

def analyze_oil(img_bgr: np.ndarray) -> AnalysisResult:
    """
    상위 2%의 밝은 픽셀이 유분이 많은 영역으로 간주
    
    처리 흐름:
      1) 그레이스케일 변환
      2) 상위 2% 밝기 임계값 산출 (np.percentile 98)
      3) 임계값 이상 픽셀 → 유분 마스크
      4) 전체 픽셀 대비 유분 픽셀 비율 → 유분 지수
    """
    gray = to_gray(img_bgr)
    
    # 상위 2% 임계값
    threshold = np.percentile(gray, 98)
    oil_mask = (gray >= threshold).astype(np.uint8) * 255
    
    # 노이즈 제거 (작은 점 제거)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    oil_mask = cv2.morphologyEx(oil_mask, cv2.MORPH_OPEN, kernel)
    
    ratio = oil_mask.sum() / 255 / gray.size
    score = normalize_score(ratio, 0.005, 0.05)
    
    return AnalysisResult(
        score=score,
        mask=oil_mask,
        detail={"threshold": float(threshold), "oil_pixel_ratio": float(ratio)}
    )


# ============================================================
# 2. 주름 (Wrinkle)
# ============================================================

def analyze_wrinkle(img_bgr: np.ndarray,
                    min_length: int = 20,
                    depth_channel: str = "S") -> AnalysisResult:
    """
      - 주름은 주변보다 어둡게 나타남
      - HSL 색공간의 S 채널 값으로 깊이 측정
      - 일정 길이 이상인 경우에만 주름으로 판정
      - 주름 길이 × 평균 깊이 / 전체 이미지 크기 = 주름 지수
    
    처리 흐름:
      1) HLS의 S채널(채도) 반전 → 주름 영역은 채도 낮음(어두운 선)
      2) Canny 엣지로 주름 후보 선 검출
      3) HoughLinesP로 선 길이 필터링 (min_length 이상만)
      4) 감지된 주름 영역의 S채널 평균 → 깊이 대리값
      5) 주름 지수 = Σ(길이 × 평균깊이) / 이미지크기
    """
    h, w = img_bgr.shape[:2]
    hls = to_hls(img_bgr)
    s_channel = hls[:, :, 2]   # S: 채도 (주름 깊이 대리값)
    l_channel = hls[:, :, 1]   # L: 명도
    
    # 주름 = 주변보다 어두운 선 → L채널 반전 후 엣지 검출
    l_inv = cv2.bitwise_not(l_channel)
    blurred = cv2.GaussianBlur(l_inv, (3, 3), 0)
    edges = cv2.Canny(blurred, threshold1=30, threshold2=80)
    
    # 체모 제거: 너무 얇고 긴 직선 제거 (체모는 거의 완벽한 직선)
    lines_hair = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50,
                                  minLineLength=w//4, maxLineGap=5)
    if lines_hair is not None:
        hair_mask = np.zeros_like(edges)
        for line in lines_hair:
            x1, y1, x2, y2 = line[0]
            cv2.line(hair_mask, (x1, y1), (x2, y2), 255, 2)
        edges = cv2.bitwise_and(edges, cv2.bitwise_not(hair_mask))
    
    # 주름 후보 선분 검출 (min_length 이상)
    wrinkle_mask = np.zeros((h, w), dtype=np.uint8)
    total_wrinkle_score = 0.0
    
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=15,
                             minLineLength=min_length, maxLineGap=8)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.hypot(x2 - x1, y2 - y1)
            
            # 해당 선 위의 S채널 값 = 깊이 대리값
            line_img = np.zeros((h, w), dtype=np.uint8)
            cv2.line(line_img, (x1, y1), (x2, y2), 255, 2)
            depth_vals = s_channel[line_img > 0]
            avg_depth = float(depth_vals.mean()) if len(depth_vals) > 0 else 0.0
            
            total_wrinkle_score += length * avg_depth
            cv2.line(wrinkle_mask, (x1, y1), (x2, y2), 255, 2)
    
    # 주름 지수 = Σ(길이 × 깊이) / 이미지 크기
    wrinkle_index = total_wrinkle_score / (h * w)
    score = normalize_score(wrinkle_index, 0, 15)
    
    return AnalysisResult(
        score=score,
        mask=wrinkle_mask,
        detail={"wrinkle_index": wrinkle_index, "line_count": 0 if lines is None else len(lines)}
    )


# ============================================================
# 3. 피지 (Sebaceous / Porphyrin)
# ============================================================

def analyze_sebaceous(img_bgr: np.ndarray) -> AnalysisResult:
    """
      - 포피린은 UVL에 반응하여 주홍색 빛
      - 노란색 및 녹색 빛을 감지하여 모공 막힘 여부 진단
      - 크기에 따른 보정식을 적용해 피지 지수 산출
    
    처리 흐름:
      일반광 이미지에서 UVL 형광 반응 색상 근사:
      1) 주홍색(orange-red) 범위: HSV 기준 H=0~25
      2) 노란색 범위: H=25~35
      3) 연두/녹색 범위: H=35~85
      4) 각 색상 마스크 합산 → 피지 후보 영역
      5) 크기별 보정: 작은 점(모공 막힘)에 높은 가중치
    """
    hsv = to_hsv(img_bgr)
    h_ch, s_ch, v_ch = cv2.split(hsv)
    
    # 충분한 채도 & 밝기를 가진 픽셀만 (형광 반응 모사)
    valid = (s_ch > 60) & (v_ch > 80)
    
    # 주홍색 (포피린 형광: orange~red)
    orange_red = valid & (h_ch <= 25) & (s_ch > 80)
    # 노란색
    yellow = valid & (h_ch > 25) & (h_ch <= 35)
    # 녹색 (여드름 전구 물질)
    green = valid & (h_ch > 35) & (h_ch <= 85) & (s_ch > 70)
    
    combined = (orange_red | yellow | green).astype(np.uint8) * 255
    
    # 크기별 보정: contour 면적에 따라 가중합
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sebaceous_index = 0.0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 5:
            continue
        # 작을수록(모공 막힘) 더 높은 가중치 → 역수 보정
        weight = 1.0 / (1.0 + np.log1p(area))
        sebaceous_index += area * weight
    
    h, w = img_bgr.shape[:2]
    sebaceous_index /= (h * w)
    score = normalize_score(sebaceous_index, 0, 0.01)
    
    return AnalysisResult(
        score=score,
        mask=combined,
        detail={"sebaceous_index": sebaceous_index, "pore_count": len(contours)}
    )


# ============================================================
# 4. 모공 (Pore)
# ============================================================

def analyze_pore(img_bgr: np.ndarray) -> AnalysisResult:
    """
      - Morphology Opening Method를 통해 모공을 분리
      - VSL 알고리즘으로 깊이를 측정
      - 깊이와 넓이를 기반으로 모공 지수 산출
    
    처리 흐름:
      1) 그레이스케일 → Gaussian Blur로 배경 추정
      2) 원본 - 배경 = 국소적으로 어두운 점(모공)
      3) Morphology Opening으로 작고 둥근 모공 분리
      4) VSL 깊이 근사: L채널 반전값 = 상대적 어두움
      5) 모공 지수 = Σ(넓이 × 평균깊이) / 이미지크기
    """
    gray = to_gray(img_bgr)
    lab = to_lab(img_bgr)
    l_ch = lab[:, :, 0]
    
    # 배경(저주파) 추정 후 모공(고주파 어두운 점) 분리
    background = cv2.GaussianBlur(gray, (21, 21), 0)
    diff = background.astype(np.int16) - gray.astype(np.int16)
    diff = np.clip(diff, 0, 255).astype(np.uint8)
    
    # Morphology Opening: 작고 둥근 구조만 남김 (모공 크기 = 3~15px)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(diff, cv2.MORPH_OPEN, kernel_open)
    
    # 이진화
    _, pore_mask = cv2.threshold(opened, 15, 255, cv2.THRESH_BINARY)
    
    # 너무 큰 영역 제거 (모공이 아닌 그림자)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    not_shadow = cv2.morphologyEx(pore_mask, cv2.MORPH_OPEN, kernel_close)
    pore_mask = cv2.subtract(pore_mask, not_shadow)
    
    # 모공 지수: 넓이 × 깊이(L채널 어두움)
    contours, _ = cv2.findContours(pore_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pore_index = 0.0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 3 or area > 200:   # 모공 크기 범위 필터
            continue
        mask_cnt = np.zeros_like(l_ch)
        cv2.drawContours(mask_cnt, [cnt], -1, 255, -1)
        depth = float(255 - l_ch[mask_cnt > 0].mean())   # 어두울수록 깊음
        pore_index += area * depth
    
    h, w = img_bgr.shape[:2]
    pore_index /= (h * w)
    score = normalize_score(pore_index, 0, 500)
    
    return AnalysisResult(
        score=score,
        mask=pore_mask,
        detail={"pore_index": pore_index, "pore_count": len(contours)}
    )


# ============================================================
# 5. 기미·잡티 (Pigmentation Spot)
# ============================================================

def analyze_spot(img_bgr: np.ndarray,
                 local_radius: int = 30) -> AnalysisResult:
    """
      - 밝기 값을 기준으로 탐지
      - 어두운 정도에 따라 1차 가중치
      - 주변 영역 대비 어두움 정도에 2차 가중치
      - 두 값을 결합 후 이미지 크기로 정규화
    
    처리 흐름:
      1) L채널(밝기) 추출
      2) 1차 가중치: 전역 어두움 (전체 평균 대비)
      3) 2차 가중치: 국소 어두움 (주변 local_radius 반경 대비)
      4) 두 가중치 결합 → 잡티 후보 마스크
      5) 잡티 지수 = 면적합 / 이미지크기
    """
    lab = to_lab(img_bgr)
    l_ch = lab[:, :, 0].astype(np.float32)
    
    # 1차 가중치: 전역 평균 대비 어두운 정도
    global_mean = l_ch.mean()
    w1 = np.clip(global_mean - l_ch, 0, None)   # 평균보다 얼마나 어두운가
    
    # 2차 가중치: 국소 평균 대비 어두운 정도
    blur_size = local_radius * 2 + 1
    local_mean = cv2.GaussianBlur(l_ch, (blur_size, blur_size), local_radius / 3)
    w2 = np.clip(local_mean - l_ch, 0, None)
    
    # 결합 점수 (동일 가중치)
    combined_score = (w1 + w2) / 2.0
    
    # 임계값: 결합 점수 상위 5%를 잡티로 판정
    threshold = np.percentile(combined_score, 95)
    spot_mask = (combined_score >= threshold).astype(np.uint8) * 255
    
    # 너무 작은 노이즈 제거
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    spot_mask = cv2.morphologyEx(spot_mask, cv2.MORPH_OPEN, kernel)
    
    # 잡티 지수: 면적 × 평균 결합점수 / 이미지크기
    h, w = img_bgr.shape[:2]
    spot_pixels = combined_score[spot_mask > 0]
    spot_index = float(spot_pixels.sum()) / (h * w) if len(spot_pixels) > 0 else 0.0
    score = normalize_score(spot_index, 0, 5)
    
    return AnalysisResult(
        score=score,
        mask=spot_mask,
        detail={"spot_index": spot_index, "global_mean_L": float(global_mean)}
    )


# ============================================================
# 6. 색소침착 (Pigmentation / UV Damage)
# ============================================================

def analyze_pigmentation(img_bgr: np.ndarray) -> AnalysisResult:
    """
      - 멜라닌이 자외선을 흡수해 반응하는 특수 조명 활용
      - 색소침착 부위는 더 어둡게 나타남
      - 갈색 멜라닌 부위만 검출
      - 절대 밝기 + 주변 대비에 각각 가중치 → 색소침착 지수
    
    처리 흐름:
      UV 조명 효과 근사: 갈색 멜라닌은 RGB에서 R>G>B 패턴 + 낮은 밝기
      1) LAB에서 L(밝기) + b*(황색도) 추출
      2) 갈색 범위: Lab에서 낮은 L, 양의 b* (황갈색)
      3) HSV에서 피부 갈색 H범위 마스킹
      4) 절대 밝기 가중치 + 주변 대비 가중치 결합
    """
    lab = to_lab(img_bgr)
    l_ch = lab[:, :, 0].astype(np.float32)
    b_ch = lab[:, :, 2].astype(np.float32)  # b*: 양수=황색, 음수=청색
    
    hsv = to_hsv(img_bgr)
    h_hsv = hsv[:, :, 0]
    s_hsv = hsv[:, :, 1]
    
    # 갈색 멜라닌 색상 범위 (HSV: H=10~30, 피부 갈색~적갈색)
    brown_hue = (h_hsv >= 8) & (h_hsv <= 30) & (s_hsv > 40)
    
    # 낮은 밝기 (일반 피부보다 어두운)
    dark_region = l_ch < (l_ch.mean() - l_ch.std() * 0.3)
    
    # 황색도 높은 영역 (멜라닌 색소)
    warm_region = b_ch > (b_ch.mean() + 5)
    
    # 색소침착 후보: 갈색 + 어두움 + 황색도
    candidate = brown_hue & dark_region & warm_region
    
    # 절대 밝기 가중치 (w1): 어두울수록 높음
    w1 = np.clip(100 - l_ch, 0, 100) / 100.0
    
    # 주변 대비 가중치 (w2)
    local_bg = cv2.GaussianBlur(l_ch, (61, 61), 20)
    w2 = np.clip(local_bg - l_ch, 0, 50) / 50.0
    
    combined = (w1 + w2) / 2.0 * candidate.astype(np.float32)
    
    # 마스크
    pig_mask = (combined > 0.1).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    pig_mask = cv2.morphologyEx(pig_mask, cv2.MORPH_CLOSE, kernel)
    pig_mask = cv2.morphologyEx(pig_mask, cv2.MORPH_OPEN, 
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)))
    
    h, w = img_bgr.shape[:2]
    pig_index = float(combined.sum()) / (h * w)
    score = normalize_score(pig_index, 0, 0.05)
    
    return AnalysisResult(
        score=score,
        mask=pig_mask,
        detail={"pigmentation_index": pig_index}
    )


# ============================================================
# 7. 다크서클 (Dark Circle)
# ============================================================

def analyze_dark_circle(img_bgr: np.ndarray,
                         eye_region_ratio: Tuple = (0.15, 0.45, 0.1, 0.4)) -> AnalysisResult:
    """
      - 눈 밑이 어둡고 그늘져 보이는 증상
      - 멜라닌 색소 침착, 피하정맥 노출, 눈가 피부 얇음, 혈액순환 저하
      - AI 모델을 통해 다크서클 영역 자동 예측 (→ 여기서는 규칙 기반 근사)
    
    처리 흐름:
      1) 눈 하단 관심 영역(ROI) 추출 (상단 15~45%, 좌우 10~90%)
      2) LAB의 L채널로 어두움 정도 계산
      3) 피하정맥: b* 음수(청색조) 영역 감지
      4) 색소 침착: b* 양수(황갈색) + 어두운 영역
      5) 두 원인 결합 → 다크서클 마스크 및 지수
      
    Note: eye_region_ratio = (top, bottom, left, right) 비율
          얼굴 크롭 이미지 기준. 부위별 크롭이면 (0.3, 0.8, 0.05, 0.95) 권장
    """
    h, w = img_bgr.shape[:2]
    t, b, l, r = eye_region_ratio
    roi = img_bgr[int(h*t):int(h*b), int(w*l):int(w*r)]
    
    if roi.size == 0:
        empty = np.zeros((h, w), dtype=np.uint8)
        return AnalysisResult(score=0.0, mask=empty, detail={"error": "ROI empty"})
    
    lab_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    l_roi = lab_roi[:, :, 0].astype(np.float32)
    b_roi = lab_roi[:, :, 2].astype(np.float32)  # b*
    
    # 주변 평균 대비 어두운 정도
    mean_l = l_roi.mean()
    dark = l_roi < (mean_l - 8)
    
    # 피하정맥: 청색조 (b* < 128, 즉 LAB b채널 < 중립값)
    b_neutral = 128.0
    vein = (b_roi < b_neutral - 3) & dark
    
    # 멜라닌: 황갈색 (b* > 128) + 어두움
    melanin = (b_roi > b_neutral + 3) & dark
    
    dc_roi_mask = ((vein | melanin)).astype(np.uint8) * 255
    
    # 형태학적 정리
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dc_roi_mask = cv2.morphologyEx(dc_roi_mask, cv2.MORPH_CLOSE, kernel)
    
    # 전체 이미지 크기 마스크로 복원
    full_mask = np.zeros((h, w), dtype=np.uint8)
    full_mask[int(h*t):int(h*b), int(w*l):int(w*r)] = dc_roi_mask
    
    dc_ratio = dc_roi_mask.sum() / 255 / roi.shape[0] / roi.shape[1]
    
    # 어두움 강도도 반영
    dark_intensity = float(mean_l - l_roi[dc_roi_mask > 0].mean()) if dc_roi_mask.any() else 0.0
    dc_index = dc_ratio * (1 + dark_intensity / 50)
    score = normalize_score(dc_index, 0, 0.5)
    
    return AnalysisResult(
        score=score,
        mask=full_mask,
        detail={"dark_circle_index": dc_index, "mean_L_roi": float(mean_l),
                "dark_intensity": dark_intensity}
    )


# ============================================================
# 8. 광채 (Radiance / Glow)
# ============================================================

def analyze_radiance(img_bgr: np.ndarray) -> AnalysisResult:
    """
      - VSL 기법을 활용해 얼굴의 광채를 감지
      - 피부 세포의 반사율, 표면의 매끄러움, 색상의 균일성에 영향
      
    VSL(Vascular/Surface Light) 근사:
      1) Specular highlight 영역: V채널 상위 + S채널 하위 (빛 반사)
      2) 표면 균일성: 국소 표준편차가 낮은 영역
      3) 색상 균일성: LAB에서 a*, b* 변화가 작은 영역
      4) 세 조건 결합 → 광채 영역 (높을수록 피부가 빛남)
    """
    hsv = to_hsv(img_bgr)
    v_ch = hsv[:, :, 2].astype(np.float32)
    s_ch = hsv[:, :, 1].astype(np.float32)
    
    lab = to_lab(img_bgr)
    a_lab = lab[:, :, 1].astype(np.float32)
    b_lab = lab[:, :, 2].astype(np.float32)
    
    # 1) Specular highlight: 밝고 무채색에 가까운 영역
    bright = v_ch > np.percentile(v_ch, 75)
    low_sat = s_ch < np.percentile(s_ch, 40)
    specular = bright & low_sat
    
    # 2) 표면 균일성: 국소 표준편차 (낮을수록 매끄러움)
    gray = to_gray(img_bgr).astype(np.float32)
    local_mean = cv2.GaussianBlur(gray, (15, 15), 5)
    local_sq_mean = cv2.GaussianBlur(gray**2, (15, 15), 5)
    local_std = np.sqrt(np.clip(local_sq_mean - local_mean**2, 0, None))
    smooth = local_std < np.percentile(local_std, 40)
    
    # 3) 색상 균일성: a*, b* 채널의 국소 표준편차
    a_std = cv2.GaussianBlur((a_lab - cv2.GaussianBlur(a_lab, (15,15), 5))**2, (15,15), 5)
    b_std = cv2.GaussianBlur((b_lab - cv2.GaussianBlur(b_lab, (15,15), 5))**2, (15,15), 5)
    color_uniform = (a_std < np.percentile(a_std, 50)) & (b_std < np.percentile(b_std, 50))
    
    # 결합: 두 가지 이상 조건 만족
    vote = specular.astype(np.uint8) + smooth.astype(np.uint8) + color_uniform.astype(np.uint8)
    radiance_mask = (vote >= 2).astype(np.uint8) * 255
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    radiance_mask = cv2.morphologyEx(radiance_mask, cv2.MORPH_CLOSE, kernel)
    
    h, w = img_bgr.shape[:2]
    radiance_ratio = radiance_mask.sum() / 255 / (h * w)
    # 광채는 높을수록 좋음 → 역방향 스코어
    score = normalize_score(radiance_ratio, 0, 0.6)
    
    return AnalysisResult(
        score=score,
        mask=radiance_mask,
        detail={"radiance_ratio": float(radiance_ratio)}
    )


# ============================================================
# 9. 홍조 (Redness / Flushing)
# ============================================================

def analyze_redness(img_bgr: np.ndarray) -> AnalysisResult:
    """
      - 색차 분석을 통해 감지
      - RGB 영상의 R, G 채널 대비 주변과의 색상 차이로 붉기 지수 계산
    
    처리 흐름:
      1) R채널 / (G채널 + ε) → R/G 비율 맵 (붉기 강도)
      2) 주변 평균과의 차이 계산 (국소 색상 대비)
      3) LAB a* 채널: 양수일수록 붉음 → 보조 지표
      4) 두 지표 결합 → 붉기 지수 및 마스크
    """
    img_float = img_bgr.astype(np.float32)
    b_ch = img_float[:, :, 0]
    g_ch = img_float[:, :, 1]
    r_ch = img_float[:, :, 2]
    
    # R/G 비율 맵
    rg_ratio = r_ch / (g_ch + 1.0)
    
    # 국소 평균 대비 R/G 차이
    local_rg = cv2.GaussianBlur(rg_ratio, (31, 31), 10)
    rg_diff = rg_ratio - local_rg   # 주변보다 얼마나 붉은가
    
    # LAB a* 채널 (붉기 강도)
    lab = to_lab(img_bgr)
    a_ch = lab[:, :, 1].astype(np.float32)
    a_neutral = 128.0
    redness_a = np.clip(a_ch - a_neutral, 0, None)  # 양수=붉음
    
    # 두 지표 정규화 후 결합
    rg_norm = cv2.normalize(np.clip(rg_diff, 0, None), None, 0, 1, cv2.NORM_MINMAX)
    a_norm = cv2.normalize(redness_a, None, 0, 1, cv2.NORM_MINMAX)
    combined = (rg_norm * 0.6 + a_norm * 0.4)
    
    # 임계값 기반 마스크
    threshold = np.percentile(combined, 85)
    redness_mask = (combined >= threshold).astype(np.uint8) * 255
    
    # 너무 밝은 영역(유분 반사) 제거
    v_ch = to_hsv(img_bgr)[:, :, 2]
    redness_mask[v_ch > 240] = 0
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    redness_mask = cv2.morphologyEx(redness_mask, cv2.MORPH_OPEN, kernel)
    
    h, w = img_bgr.shape[:2]
    redness_pixels = combined[redness_mask > 0]
    redness_index = float(redness_pixels.mean() * redness_mask.sum() / 255) / (h * w) \
                    if len(redness_pixels) > 0 else 0.0
    score = normalize_score(redness_index, 0, 0.05)
    
    return AnalysisResult(
        score=score,
        mask=redness_mask,
        detail={"redness_index": redness_index}
    )


# ============================================================
# 10. 번들거림 (Shine / Gloss)
# ============================================================

def analyze_shine(img_bgr: np.ndarray) -> AnalysisResult:
    """
      - 피지 분비 과도하면 번들거림 발생
      - AI 알고리즘이 번들거리는 부위 식별
      
    번들거림 vs 광채 차이:
      - 광채: 균일하고 부드러운 반사 (건강한 피부)
      - 번들거림: 국소적으로 강한 specular highlight (유분 과다)
    
    처리 흐름:
      1) V채널 상위 3% (매우 밝은 spot)
      2) S채널 낮음 (무채색에 가까운 → 유분 반사)
      3) 국소 대비가 높은 영역 (주변보다 급격히 밝음 → specular)
      4) 세 조건 교집합 → 번들거림 마스크
    """
    hsv = to_hsv(img_bgr)
    v_ch = hsv[:, :, 2].astype(np.float32)
    s_ch = hsv[:, :, 1].astype(np.float32)
    
    # 매우 밝은 spot (상위 3%)
    very_bright = v_ch >= np.percentile(v_ch, 97)
    
    # 낮은 채도 (유분 반사는 무채색)
    low_sat = s_ch < 60
    
    # 국소 대비: 주변보다 급격히 밝은 영역
    local_v = cv2.GaussianBlur(v_ch, (21, 21), 7)
    local_contrast = v_ch - local_v
    high_contrast = local_contrast > np.percentile(local_contrast, 90)
    
    shine_mask = (very_bright & low_sat & high_contrast).astype(np.uint8) * 255
    
    # 모폴로지 정리
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    shine_mask = cv2.morphologyEx(shine_mask, cv2.MORPH_DILATE, kernel)
    shine_mask = cv2.morphologyEx(shine_mask, cv2.MORPH_OPEN, kernel)
    
    h, w = img_bgr.shape[:2]
    shine_ratio = shine_mask.sum() / 255 / (h * w)
    score = normalize_score(shine_ratio, 0, 0.03)
    
    return AnalysisResult(
        score=score,
        mask=shine_mask,
        detail={"shine_ratio": float(shine_ratio)}
    )


# ============================================================
# 11. 칙칙함 (Dullness)
# ============================================================

def analyze_dullness(img_bgr: np.ndarray) -> AnalysisResult:
    """
      - 각질 세포 과도 축적으로 광채를 잃은 상태
      - 피부 본연의 톤과 비교해 광채가 부족하고 어두워 보이는 영역 감지
    
    처리 흐름:
      1) ITA(Individual Typology Angle) 계산: arctan((L-50)/b*)
         → 높을수록 밝은 피부, 낮을수록 어두운 피부
      2) 색상 균일성 낮은 영역 감지 (불균일한 피부 톤)
      3) L채널 낮고 채도도 낮은 회색빛 영역
      4) 세 조건 결합 → 칙칙함 마스크
    """
    lab = to_lab(img_bgr)
    l_ch = lab[:, :, 0].astype(np.float32)
    b_lab = lab[:, :, 2].astype(np.float32)
    
    hsv = to_hsv(img_bgr)
    s_hsv = hsv[:, :, 1].astype(np.float32)
    
    # ITA 계산 (낮을수록 칙칙)
    ita = np.degrees(np.arctan2(l_ch - 50.0, b_lab - 128.0 + 1e-6))
    low_ita = ita < np.percentile(ita, 35)   # 하위 35%: 칙칙한 영역
    
    # 낮은 밝기
    dark = l_ch < l_ch.mean() - 5
    
    # 낮은 채도 (회색빛: 활기 없는 피부)
    low_chroma = s_hsv < np.percentile(s_hsv, 40)
    
    dullness_map = (low_ita & dark & low_chroma).astype(np.uint8) * 255
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dullness_map = cv2.morphologyEx(dullness_map, cv2.MORPH_CLOSE, kernel)
    dullness_map = cv2.morphologyEx(dullness_map, cv2.MORPH_OPEN, 
                                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    
    h, w = img_bgr.shape[:2]
    dull_ratio = dullness_map.sum() / 255 / (h * w)
    
    # ITA 평균도 반영
    mean_ita = float(ita.mean())
    dull_index = dull_ratio * (1 + max(0, 30 - mean_ita) / 30)
    score = normalize_score(dull_index, 0, 0.5)
    
    return AnalysisResult(
        score=score,
        mask=dullness_map,
        detail={"dullness_index": dull_index, "mean_ITA": mean_ita}
    )


# ============================================================
# 12. 피부색 (Skin Tone)
# ============================================================

# ITA 기준 피부 타입 분류 (Chardon et al.)
ITA_SKIN_TYPES = [
    (55,  "Very Light"),
    (41,  "Light"),
    (28,  "Intermediate"),
    (10,  "Tan"),
    (-30, "Brown"),
    (-90, "Dark"),
]

def analyze_skin_color(img_bgr: np.ndarray) -> AnalysisResult:
    """
      - 얼굴 전체 촬영 후 부위별 영역 분류 → RGB 값 측정
      - 부위별 RGB + 기준 피부톤 RGB 비교 → 피부색 계산
    
    처리 흐름:
      1) 피부 마스크 생성 (너무 어둡거나 밝은 픽셀 제외)
      2) 피부 영역 평균 RGB 측정
      3) ITA 지수 계산 → 피부 타입 분류
      4) 기준 피부톤 대비 색차(ΔE) 계산
    """
    lab = to_lab(img_bgr)
    l_ch = lab[:, :, 0].astype(np.float32)
    a_ch = lab[:, :, 1].astype(np.float32)
    b_ch = lab[:, :, 2].astype(np.float32)
    
    # 피부 영역 마스크 (극단값 제외)
    skin_mask = (l_ch > 30) & (l_ch < 230) & \
                (a_ch > 125) & (a_ch < 155) & \
                (b_ch > 125) & (b_ch < 175)
    skin_mask = skin_mask.astype(np.uint8) * 255
    
    if skin_mask.sum() == 0:
        skin_mask = np.ones(img_bgr.shape[:2], dtype=np.uint8) * 255
    
    # 피부 영역 평균 RGB
    r_mean = float(img_bgr[:,:,2][skin_mask>0].mean())
    g_mean = float(img_bgr[:,:,1][skin_mask>0].mean())
    b_mean = float(img_bgr[:,:,0][skin_mask>0].mean())
    
    # 평균 L, b로 ITA 계산
    l_mean = float(l_ch[skin_mask>0].mean())
    b_mean_lab = float(b_ch[skin_mask>0].mean())
    ita_val = float(np.degrees(np.arctan2(l_mean - 50, b_mean_lab - 128 + 1e-6)))
    
    # ITA로 피부 타입 분류
    skin_type = "Dark"
    for ita_threshold, type_name in ITA_SKIN_TYPES:
        if ita_val > ita_threshold:
            skin_type = type_name
            break
    
    # 점수: ITA 정규화 (높을수록 밝은 피부)
    score = normalize_score(ita_val, -90, 90)
    
    return AnalysisResult(
        score=score,
        mask=skin_mask,
        detail={
            "ITA": ita_val,
            "skin_type": skin_type,
            "mean_RGB": (r_mean, g_mean, b_mean),
            "mean_L": l_mean
        }
    )


# ============================================================
# 통합 탄력도 계산 (주름 + 칙칙함 종합)
# ============================================================

def calc_elasticity(wrinkle_score: float, dullness_score: float) -> float:
    """
    주름과 칙칙함의 측정 결과를 종합하여 피부 탄력도를 산출
    탄력도 = 100 - (주름 × 0.6 + 칙칙함 × 0.4)
    """
    return max(0.0, 100.0 - (wrinkle_score * 0.6 + dullness_score * 0.4))


# ============================================================
# 메인 분석기 클래스
# ============================================================

class SkinAnalyzer:
    """
    피부 이미지 종합 분석기
    
    사용 예시:
        analyzer = SkinAnalyzer(img_bgr)
        results  = analyzer.run_all()
        print(results['redness'].score)    # 0~100
        print(results['redness'].mask)     # numpy mask
        vis = analyzer.visualize(results)
        cv2.imwrite("output.jpg", vis)
    """
    
    def __init__(self, img_bgr: np.ndarray):
        self.img = img_bgr
    
    def run_all(self) -> Dict[str, AnalysisResult]:
        results = {}
        
        print("  [1/12] 유분 분석...")
        results['oil']           = analyze_oil(self.img)
        
        print("  [2/12] 주름 분석...")
        results['wrinkle']       = analyze_wrinkle(self.img)
        
        print("  [3/12] 피지 분석...")
        results['sebaceous']     = analyze_sebaceous(self.img)
        
        print("  [4/12] 모공 분석...")
        results['pore']          = analyze_pore(self.img)
        
        print("  [5/12] 기미·잡티 분석...")
        results['spot']          = analyze_spot(self.img)
        
        print("  [6/12] 색소침착 분석...")
        results['pigmentation']  = analyze_pigmentation(self.img)
        
        print("  [7/12] 다크서클 분석...")
        results['dark_circle']   = analyze_dark_circle(self.img)
        
        print("  [8/12] 광채 분석...")
        results['radiance']      = analyze_radiance(self.img)
        
        print("  [9/12] 홍조 분석...")
        results['redness']       = analyze_redness(self.img)
        
        print("  [10/12] 번들거림 분석...")
        results['shine']         = analyze_shine(self.img)
        
        print("  [11/12] 칙칙함 분석...")
        results['dullness']      = analyze_dullness(self.img)
        
        print("  [12/12] 피부색 분석...")
        results['skin_color']    = analyze_skin_color(self.img)
        
        # 탄력도 (주름 + 칙칙함 종합)
        elasticity = calc_elasticity(results['wrinkle'].score, results['dullness'].score)
        results['elasticity'] = AnalysisResult(
            score=elasticity,
            mask=np.zeros(self.img.shape[:2], dtype=np.uint8),
            detail={"formula": "100 - (wrinkle×0.6 + dullness×0.4)"}
        )
        
        return results
    
    def visualize(self, results: Dict[str, AnalysisResult]) -> np.ndarray:
        """분석 결과를 컬러 오버레이로 시각화"""
        
        # 항목별 오버레이 색상 (BGR)
        overlay_colors = {
            'oil':          (0,   255, 255),  # 노란색
            'wrinkle':      (0,   0,   200),  # 빨간색
            'sebaceous':    (0,   165, 255),  # 주황색
            'pore':         (255, 0,   150),  # 보라색
            'spot':         (0,   100, 180),  # 갈색
            'pigmentation': (0,   80,  150),  # 짙은 갈색
            'dark_circle':  (150, 0,   0),    # 짙은 파랑
            'radiance':     (0,   255, 200),  # 청록색
            'redness':      (0,   0,   255),  # 순빨강
            'shine':        (255, 255, 0),    # 하늘색
            'dullness':     (100, 100, 100),  # 회색
        }
        
        # 항목명 레이블
        labels = {
            'oil': 'Oil', 'wrinkle': 'Wrinkle', 'sebaceous': 'Sebaceous',
            'pore': 'Pore', 'spot': 'Spot', 'pigmentation': 'Pigment',
            'dark_circle': 'DarkCircle', 'radiance': 'Radiance',
            'redness': 'Redness', 'shine': 'Shine', 'dullness': 'Dullness'
        }
        
        img_vis = self.img.copy()
        for key, color in overlay_colors.items():
            if key not in results:
                continue
            mask = results[key].mask
            if mask.shape != img_vis.shape[:2]:
                continue
            overlay = img_vis.copy()
            overlay[mask > 0] = color
            img_vis = cv2.addWeighted(img_vis, 0.7, overlay, 0.3, 0)
        
        # 스코어 텍스트 출력
        h, w = img_vis.shape[:2]
        panel_w = 260
        panel = np.ones((h, panel_w, 3), dtype=np.uint8) * 30  # 다크 배경
        
        # 항목별 점수 바 출력
        items_for_panel = [
            ('Oil',        results.get('oil')),
            ('Wrinkle',    results.get('wrinkle')),
            ('Sebaceous',  results.get('sebaceous')),
            ('Pore',       results.get('pore')),
            ('Spot',       results.get('spot')),
            ('Pigment',    results.get('pigmentation')),
            ('DarkCircle', results.get('dark_circle')),
            ('Radiance',   results.get('radiance')),
            ('Redness',    results.get('redness')),
            ('Shine',      results.get('shine')),
            ('Dullness',   results.get('dullness')),
            ('Elasticity', results.get('elasticity')),
            ('SkinColor',  results.get('skin_color')),
        ]
        
        y_start = 25
        row_h = max(20, (h - 40) // len(items_for_panel))
        
        for i, (name, res) in enumerate(items_for_panel):
            if res is None:
                continue
            y = y_start + i * row_h
            score = res.score
            
            # 라벨
            cv2.putText(panel, f"{name}", (8, y + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
            
            # 점수 바 배경
            bar_x, bar_w_total, bar_h = 100, 130, 12
            cv2.rectangle(panel, (bar_x, y + 2), (bar_x + bar_w_total, y + 2 + bar_h),
                          (70, 70, 70), -1)
            
            # 점수 바 색상 (낮을수록 초록, 높을수록 빨강 — 광채/탄력/피부색은 반전)
            positive_items = {'radiance', 'elasticity', 'skin_color'}
            key_lower = name.lower().replace('circle','_circle').replace('color','_color')
            if name.lower() in ['radiance', 'elasticity', 'skincolor']:
                bar_color = (0, int(200 * score / 100), 0)
            else:
                ratio = score / 100
                bar_color = (0, int(200 * (1 - ratio)), int(200 * ratio))
            
            bar_fill = int(bar_w_total * score / 100)
            if bar_fill > 0:
                cv2.rectangle(panel, (bar_x, y + 2), (bar_x + bar_fill, y + 2 + bar_h),
                              bar_color, -1)
            
            # 점수 숫자
            cv2.putText(panel, f"{score:.0f}", (bar_x + bar_w_total + 5, y + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        
        # 피부 타입 표기
        if 'skin_color' in results and 'skin_type' in results['skin_color'].detail:
            skin_type = results['skin_color'].detail['skin_type']
            cv2.putText(panel, f"Type: {skin_type}",
                        (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 220, 255), 1)
        
        return np.hstack([img_vis, panel])


# ============================================================
# 실행 예시
# ============================================================

if __name__ == "__main__":
    import sys, os
    
    img_path = sys.argv[1] if len(sys.argv) > 1 else "./sample.jpg"
    
    if not os.path.exists(img_path):
        print(f"이미지 없음: {img_path}")
        print("사용법: python skin_analysis.py <이미지경로>")
        sys.exit(1)
    
    img = cv2.imread(img_path)
    if img is None:
        print("이미지 로드 실패")
        sys.exit(1)
    
    print(f"\n🔍 피부 분석 시작: {img_path} ({img.shape[1]}x{img.shape[0]})\n")
    
    analyzer = SkinAnalyzer(img)
    results  = analyzer.run_all()
    
    print("\n📊 분석 결과 (0~100)")
    print("=" * 45)
    
    result_items = [
        ("유분",       "oil"),
        ("주름",       "wrinkle"),
        ("피지",       "sebaceous"),
        ("모공",       "pore"),
        ("기미·잡티",  "spot"),
        ("색소침착",   "pigmentation"),
        ("다크서클",   "dark_circle"),
        ("광채",       "radiance"),
        ("홍조",       "redness"),
        ("번들거림",   "shine"),
        ("칙칙함",     "dullness"),
        ("탄력도",     "elasticity"),
        ("피부색",     "skin_color"),
    ]
    
    for name_kr, key in result_items:
        if key not in results:
            continue
        r = results[key]
        bar = "█" * int(r.score / 5) + "░" * (20 - int(r.score / 5))
        extra = ""
        if key == "skin_color" and "skin_type" in r.detail:
            extra = f"  [{r.detail['skin_type']}]"
        print(f"  {name_kr:<10} {bar} {r.score:5.1f}{extra}")
    
    print("=" * 45)
    
    vis = analyzer.visualize(results)
    out_path = img_path.replace(".", "_analyzed.")
    cv2.imwrite(out_path, vis)
    print(f"\n✅ 시각화 저장: {out_path}\n")