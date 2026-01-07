# SAM Auto-Labeler

SAM 기반 보호장비 자동 라벨링 도구 - PoC

## 프로젝트 개요

작업자가 착용한 보호장비(헬멧, 장갑, 상의, 바지, 부츠)를 자동으로 검출하고 라벨링하는 도구입니다.

### 핵심 기술
- **SAM2**: 클래스 무관 세그멘테이션으로 익명 마스크 생성
- **YOLO-Pose**: 신체 키포인트 검출
- **매핑 로직**: 키포인트 근접성 기반으로 마스크에 의미 부여

## 빠른 시작

### 1. Docker 환경 구축

```bash
# Docker 이미지 빌드
docker-compose build

# 컨테이너 실행
docker-compose up -d

# 컨테이너 접속
docker exec -it sam_autolabel_dev /bin/bash
```

### 2. 모델 테스트

```bash
# SAM2 테스트
python scripts/test_sam.py

# YOLO-Pose 테스트
python scripts/test_pose.py
```

### 3. 추론 실행

```bash
# 단일 이미지 처리
python scripts/run_inference.py --image data/input/sample.jpg

# 폴더 일괄 처리
python scripts/batch_process.py --input data/input --output data/output
```

## 프로젝트 구조

```
sam-auto-labeler/
├── config/
│   ├── config.yaml              # 설정 파일
│   └── classes.yaml             # 클래스 정의
├── src/
│   ├── sam_segmentation.py      # SAM2 세그멘테이션
│   ├── pose_estimation.py       # YOLO-Pose 키포인트 추출
│   ├── mask_mapper.py           # 마스크-라벨 매핑
│   ├── bbox_converter.py        # 마스크→바운딩박스 변환
│   ├── label_exporter.py        # 라벨 출력
│   ├── visualizer.py            # 시각화
│   ├── person_filter.py         # Person 영역 필터링
│   └── pipeline.py              # 전체 파이프라인
├── scripts/
│   ├── run_inference.py         # 단일 이미지 추론
│   └── batch_process.py         # 배치 처리
├── notebooks/
│   └── demo.ipynb               # 데모 노트북
├── data/
│   ├── input/                   # 입력 이미지
│   ├── output/                  # 생성된 라벨
│   └── visualizations/          # 시각화 결과
└── models/                      # 모델 체크포인트

```

## 출력 형식

### YOLO 포맷 (label.txt)
```
# class_id x_center y_center width height (normalized 0-1)
0 0.512 0.156 0.089 0.112    # helmet
1 0.234 0.534 0.045 0.067    # gloves (left)
2 0.501 0.445 0.234 0.289    # upper_body
3 0.498 0.734 0.198 0.356    # pants
4 0.445 0.923 0.067 0.089    # boots (left)
```

## 개발 계획

자세한 개발 계획은 `plan_n_process/plan.md`를 참조하세요.

## 하드웨어 요구사항

- GPU: RTX 3070 (8GB VRAM) 이상
- RAM: 16GB 이상 권장

## 라이선스

MIT License
