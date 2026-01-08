# SAM-HQ Auto-Labeler

YOLOv8 Pose와 SAM-HQ를 결합한 고정밀 보호장비(PPE) 자동 라벨링 파이프라인입니다.
작업자의 영상을 입력받아, 각 신체 부위 및 보호장비(안전모, 보안경, 장갑, 팔토시, 상/하의, 덧신 등)에 대한 정밀한 Segmentation Polygon 라벨을 자동으로 생성합니다.

## ✨ 주요 기능

- **High-Quality Segmentation**: SAM-HQ(Segment Anything Model in High Quality)를 사용하여 객체의 미세한 경계까지 포착합니다.
- **Smart Mapping Logic**: YOLO-Pose 키포인트를 활용한 지능형 매핑 로직으로 마스크에 정확한 라벨을 부여합니다.
    - **Proximity Voting**: 손목, 발목 등 중요 키포인트 근처의 작은 객체를 정밀하게 분류(장갑, 팔토시 등).
    - **Refinement**: 1차 매핑 후 오분류된 객체에 대한 재검사를 통해 수정.
- **Auto-Cleanup**: 사람 영역 외 배경 노이즈 자동 제거 (Person Segmentation 기반).
- **Format Support**: YOLO Segmentation 포맷의 Dataset 자동 생성 (`images/`, `labels/`).

## 📦 모델 준비

이 프로젝트는 다음 3개의 사전 학습된 모델을 사용합니다:

### 필수 모델

1. **SAM-HQ (ViT-B)** - 고품질 세그멘테이션
   - 파일명: `sam_hq_vit_b.pth`
   - 다운로드: [SAM-HQ GitHub Releases](https://github.com/SysCV/sam-hq/releases)
   - 경로: `models/sam_hq_vit_b.pth`

2. **YOLOv8s-seg** - 사람 영역 세그멘테이션
   - 파일명: `yolov8s-seg.pt`
   - 다운로드: [Ultralytics YOLOv8](https://github.com/ultralytics/assets/releases)
   - 경로: `models/yolov8s-seg.pt`

3. **YOLOv8m-pose** - 포즈 추정
   - 파일명: `yolov8m-pose.pt`
   - 다운로드: [Ultralytics YOLOv8](https://github.com/ultralytics/assets/releases)
   - 경로: `models/yolov8m-pose.pt`

### 설치 방법

```bash
# 1. models 디렉토리 생성 (이미 존재하는 경우 생략)
mkdir -p models

# 2. SAM-HQ 모델 다운로드
wget -O models/sam_hq_vit_b.pth https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_b.pth

# 3. YOLOv8 모델 다운로드 (Python 환경에서)
pip install ultralytics
python -c "from ultralytics import YOLO; YOLO('yolov8s-seg.pt'); YOLO('yolov8m-pose.pt')"
mv yolov8s-seg.pt models/
mv yolov8m-pose.pt models/
```

> **⚠️ 중요**: 모델 파일은 용량이 크기 때문에 GitHub 저장소에 포함되어 있지 않습니다. 위 명령어를 사용하여 **반드시 모델을 다운로드**한 후 프로젝트를 실행하세요.

## 🚀 시작하기

### 1. 환경 설정 (Docker)

Docker 및 Docker Compose가 설치되어 있어야 합니다.

```bash
# 1. 이미지 빌드 및 컨테이너 실행
docker-compose up -d

# 2. 컨테이너 접속 (선택 사항)
# main.py 실행을 위해 컨테이너 내부 쉘을 사용할 수 있습니다.
docker exec -it sam_autolabel_dev /bin/bash
```

### 2. 데이터셋 생성 (Auto-Labeling)

`main.py` 스크립트를 사용하여 비디오 파일을 데이터셋으로 변환합니다. Docker 외부(Host)에서도 `docker exec`를 통해 바로 실행 가능합니다.

```bash
# 기본 실행 (30프레임 간격)
docker exec sam_autolabel_dev python3 /workspace/main.py \
    --video_path /workspace/storage/video.mp4 \
    --output_dir /workspace/dataset_result

# 옵션 적용 (300프레임 간격, 시각화 활성화, 기존 결과 덮어쓰기)
docker exec sam_autolabel_dev python3 /workspace/main.py \
    --video_path /workspace/test_codes/source/test_video.mp4 \
    --output_dir /workspace/test_codes/result_main \
    --interval 300 \
    --visualize \
    --overwrite
```

**주요 옵션:**
- `--video_path`: 입력 비디오 파일 경로 (필수)
- `--output_dir`: 결과 데이터셋 저장 경로 (필수)
- `--interval`: 프레임 샘플링 간격 (기본값: 30)
- `--visualize`: 종합 시각화 이미지 생성 여부 (활성화 시 `/vis` 폴더에 저장)
- `--conf_threshold`: 사람 검출 신뢰도 임계값 (기본값: 0.5)
- `--overwrite`: 출력 디렉토리가 존재하면 삭제 후 재생성

## 📁 프로젝트 구조

```
sam_autolabel/
├── main.py                  # 메인 실행 스크립트 (Entry Point)
├── docker-compose.yml       # Docker 환경 설정
├── src/                     # 핵심 모듈
│   ├── auto_labeler.py      # 전체 파이프라인 관리
│   ├── mask_mapper.py       # 키포인트 기반 매핑 & Refinement 로직
│   ├── samhq_segmentation.py# SAM-HQ 추론 Wrapper
│   ├── person_segmenter.py  # YOLO-Seg 기반 사람 영역 추출
│   ├── pose_estimation.py   # YOLO-Pose 추론 Wrapper
│   ├── label_converter.py   # YOLO 포맷 변환기
│   └── visualizer.py        # 결과 시각화 모듈
├── config/                  # 설정 파일 (config.yaml)
├── models/                  # 모델 가중치 저장소
└── test_codes/              # 테스트 및 검증 스크립트
```

## 🏷️ 라벨링 클래스

다음과 같은 클래스를 자동으로 분류합니다:
- `head_cover` (두건/안전모)
- `goggles` (보안경)
- `mask` (마스크)
- `upper_body` (상의 - 방진복)
- `pants` (하의 - 방진복)
- `gloves` (장갑)
- `arm_covers` (팔토시)
- `shoe_covers` (덧신)

## ✅ 결과물 형식

**YOLO Segmentation Format** (`.txt`):
```
<class-id> <x1> <y1> <x2> <y2> ... <xn> <yn>
```
* 모든 좌표는 0~1 사이로 정규화된 Polygon 좌표입니다.
* `images/`: 원본 이미지 (Sampling된 프레임)
* `labels/`: 라벨링 텍스트 파일
* `vis/`: (옵션) 시각화 이미지
