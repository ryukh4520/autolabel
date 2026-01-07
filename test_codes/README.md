# SAM2 Video Segmentation Test

동영상에서 SAM2 자동 세그멘테이션을 테스트하는 코드입니다.

## 사용 방법

### 1. 동영상 준비

`test_codes/source/` 폴더에 테스트할 동영상 파일을 넣어주세요.

지원 형식: `.mp4`, `.avi`, `.mov`, `.mkv`

### 2. 테스트 실행

Docker 컨테이너 내에서 실행:

```bash
# 컨테이너 접속
docker exec -it sam_autolabel_dev /bin/bash

# 테스트 실행
cd /workspace
python3 test_codes/test_sam_video.py
```

### 3. 결과 확인

결과는 `test_codes/result/` 폴더에 저장됩니다.

- 파일명 형식: `frame_XXXXXX_seg.png`
- 각 이미지는 원본 프레임과 세그멘테이션 결과를 나란히 표시

## 설정

### 프레임 간격 변경

기본값은 300프레임입니다. 변경하려면 `test_sam_video.py`의 `main()` 함수에서:

```python
process_video(
    video_path=video_path,
    output_dir=result_dir,
    frame_interval=300  # 이 값을 변경
)
```

### SAM2 파라미터 조정

`process_video()` 함수의 `SAM2AutomaticMaskGenerator` 설정:

```python
mask_generator = SAM2AutomaticMaskGenerator(
    model=sam2,
    points_per_side=32,           # 그리드 포인트 수 (높을수록 세밀)
    pred_iou_thresh=0.88,         # IoU 임계값
    stability_score_thresh=0.95,  # 안정성 점수 임계값
)
```

## 출력 정보

테스트 실행 시 다음 정보를 출력합니다:

- GPU 정보 및 VRAM 사용량
- 동영상 정보 (프레임 수, FPS, 해상도)
- 처리할 프레임 개수
- 각 프레임의 검출된 마스크 개수
- 추론 시 VRAM 사용량

## 예상 처리 시간

- RTX 3070 기준
- 1920x1080 해상도
- 프레임당 약 5-10초 (이미지 크기에 따라 다름)

## 문제 해결

### 동영상 파일을 찾을 수 없음

```
[ERROR] No video files found in /workspace/test_codes/source
```

→ `test_codes/source/` 폴더에 동영상 파일을 넣어주세요.

### VRAM 부족

→ `points_per_side` 값을 낮추거나 (예: 16), 이미지 크기를 줄이세요.

### 프레임 읽기 실패

→ 동영상 코덱이 지원되는지 확인하세요. 필요시 ffmpeg로 재인코딩하세요.
