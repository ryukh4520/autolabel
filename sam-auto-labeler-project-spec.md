# SAM-based Auto-Labeler for Protective Equipment Detection

## Project Overview

Build a PoC (Proof of Concept) auto-labeling tool that generates draft labels for protective equipment worn by workers. The goal is NOT to produce perfect final labels, but to create initial annotations that human reviewers can quickly verify and refine.

---

## Problem Statement

- Workers wear protective equipment with multiple parts (helmet, gloves, upper body gear, pants, boots)
- Currently, human annotators manually draw bounding boxes for each equipment part
- This is time-consuming and labor-intensive
- We need a tool to generate draft labels automatically to speed up the annotation process

---

## Technical Approach

### Core Concept

SAM (Segment Anything Model) is a **class-agnostic** segmentation model. It identifies "visually distinguishable regions" but does NOT know what those regions are (e.g., it doesn't know if a mask is a "helmet" or "gloves").

To add semantic meaning to SAM's anonymous masks, we combine it with **human pose estimation**. By mapping masks to body keypoint locations, we can infer equipment labels.

### Pipeline Architecture

```
┌─────────────────┐
│   Input Image   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│                    Parallel Processing               │
│  ┌─────────────────┐       ┌─────────────────────┐  │
│  │  SAM2 Automatic │       │  YOLO-Pose          │  │
│  │  Mask Generation│       │  Keypoint Detection │  │
│  └────────┬────────┘       └──────────┬──────────┘  │
│           │                           │              │
│           ▼                           ▼              │
│    [Anonymous Masks]           [Body Keypoints]      │
└─────────────────────────────────────────────────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │  Mask-Keypoint      │
         │  Mapping Engine     │
         └──────────┬──────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │  Labeled Bounding   │
         │  Boxes (Draft)      │
         └─────────────────────┘
```

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Segmentation Model | SAM2 Base | VRAM efficiency on RTX 3070 (8GB) |
| Pose Model | YOLO-Pose | Fast, accurate, single-stage detection |
| Labeling Strategy | Keypoint proximity mapping | No additional training required |
| Output Scope | Person region only | Filter out background objects (buildings, vehicles, etc.) |

---

## Mask-to-Label Mapping Logic

### Body Keypoint Reference (COCO format)

```
0: nose           1: left_eye       2: right_eye
3: left_ear       4: right_ear      5: left_shoulder
6: right_shoulder 7: left_elbow     8: right_elbow
9: left_wrist     10: right_wrist   11: left_hip
12: right_hip     13: left_knee     14: right_knee
15: left_ankle    16: right_ankle
```

### Equipment-Keypoint Association

| Equipment Label | Associated Keypoints | Mapping Rule |
|-----------------|---------------------|--------------|
| `helmet` | nose, eyes, ears | Mask above head region (nose_y - offset) |
| `gloves` | wrists (9, 10) | Masks overlapping with wrist/hand area |
| `upper_body` | shoulders (5, 6), hips (11, 12) | Largest mask covering torso region |
| `pants` | hips (11, 12), knees (13, 14), ankles (15, 16) | Mask covering hip-to-ankle region |
| `boots` | ankles (15, 16) | Masks below ankle keypoints |

### Mapping Algorithm (Pseudocode)

```python
def map_masks_to_equipment(masks, keypoints):
    labeled_masks = []
    
    for mask in masks:
        # Skip masks outside person bounding box
        if not overlaps_with_person(mask, keypoints):
            continue
        
        # Calculate mask centroid
        centroid = get_mask_centroid(mask)
        
        # Determine label based on keypoint proximity
        if is_above_head(centroid, keypoints):
            label = "helmet"
        elif overlaps_with_hands(mask, keypoints):
            label = "gloves"
        elif overlaps_with_torso(mask, keypoints):
            label = "upper_body"
        elif overlaps_with_legs(mask, keypoints):
            label = "pants"
        elif is_below_ankles(centroid, keypoints):
            label = "boots"
        else:
            label = "unknown"
        
        labeled_masks.append((mask, label))
    
    return labeled_masks
```

---

## Project Structure

```
sam-auto-labeler/
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── setup.py                     # Package installation
│
├── config/
│   └── config.yaml              # Model paths, thresholds, hyperparameters
│
├── src/
│   ├── __init__.py
│   ├── sam_segmentation.py      # SAM2 automatic mask generation
│   ├── pose_estimation.py       # YOLO-Pose keypoint extraction
│   ├── mask_mapper.py           # Map masks to equipment labels
│   ├── bbox_converter.py        # Convert masks to bounding boxes
│   ├── label_exporter.py        # Export to YOLO/COCO format
│   ├── visualizer.py            # Visualization utilities
│   ├── person_filter.py         # Filter masks to person region only
│   └── pipeline.py              # Main pipeline orchestration
│
├── notebooks/
│   └── demo.ipynb               # Interactive demo notebook
│
├── tests/
│   ├── test_sam_segmentation.py
│   ├── test_pose_estimation.py
│   ├── test_mask_mapper.py
│   └── test_pipeline.py
│
├── data/
│   ├── input/                   # Input images
│   ├── output/                  # Generated labels
│   └── visualizations/          # Result visualizations
│
└── scripts/
    ├── run_inference.py         # CLI inference script
    └── batch_process.py         # Batch processing script
```

---

## Technical Requirements

### Hardware
- GPU: RTX 3070 (8GB VRAM) or equivalent
- RAM: 16GB+ recommended

### Software Dependencies

```
# requirements.txt
torch>=2.0.0
torchvision>=0.15.0
segment-anything-2        # SAM2
ultralytics>=8.0.0        # YOLO-Pose
opencv-python>=4.8.0
numpy>=1.24.0
pillow>=10.0.0
pyyaml>=6.0
matplotlib>=3.7.0
tqdm>=4.65.0
```

### VRAM Usage Estimate

| Component | VRAM |
|-----------|------|
| SAM2 Base | ~4GB |
| YOLO-Pose | ~1GB |
| Image buffer | ~1GB |
| **Total** | **~6GB** ✅ |

---

## Configuration

### config/config.yaml

```yaml
# Model settings
sam:
  model_type: "sam2_base"
  checkpoint: "models/sam2_base.pt"
  device: "cuda"
  
pose:
  model: "yolov8m-pose.pt"
  device: "cuda"
  conf_threshold: 0.5

# Mask filtering
mask_filter:
  min_area: 500           # Minimum mask area in pixels
  max_area_ratio: 0.5     # Maximum mask area as ratio of image
  person_overlap_threshold: 0.3

# Keypoint mapping thresholds
mapping:
  head_offset_ratio: 0.1  # Ratio above nose for helmet detection
  hand_radius: 50         # Pixel radius around wrist for gloves
  torso_expand_ratio: 0.1 # Expansion ratio for torso region

# Output settings
output:
  format: "yolo"          # "yolo" or "coco"
  save_visualization: true
  visualization_alpha: 0.5
```

---

## Output Format

### YOLO Format

```
# label.txt (one line per object)
# class_id x_center y_center width height (normalized 0-1)

0 0.512 0.156 0.089 0.112    # helmet
1 0.234 0.534 0.045 0.067    # gloves (left)
1 0.789 0.523 0.048 0.071    # gloves (right)
2 0.501 0.445 0.234 0.289    # upper_body
3 0.498 0.734 0.198 0.356    # pants
4 0.445 0.923 0.067 0.089    # boots (left)
4 0.556 0.928 0.071 0.092    # boots (right)
```

### Class Mapping

```yaml
# classes.yaml
names:
  0: helmet
  1: gloves
  2: upper_body
  3: pants
  4: boots
```

---

## Implementation Notes

### SAM Behavior Considerations

1. **Over-segmentation**: SAM may split a single piece of equipment into multiple masks (e.g., due to logos, reflective strips)
   - Solution: Merge adjacent masks with same label

2. **Under-segmentation**: SAM may merge visually similar adjacent equipment (e.g., same-color shirt and pants)
   - Solution: Use pose keypoints to force split at hip level

3. **Background inclusion**: SAM's automatic mode segments everything in the image
   - Solution: Filter masks using person bounding box from pose detection

### Edge Cases to Handle

- Multiple people in one image
- Partially visible equipment (cropped at image edge)
- Occluded equipment (e.g., hands in pockets)
- Non-standard equipment colors/styles

---

## Success Criteria for PoC

| Criteria | Target |
|----------|--------|
| Equipment detection rate | >80% of visible equipment detected |
| Label accuracy | >70% correct label assignment |
| Processing speed | <5 seconds per image on RTX 3070 |
| VRAM usage | <7GB peak |
| Human review time reduction | >50% compared to manual labeling |

---

## Future Enhancements (Post-PoC)

1. **Fine-tuned SAM**: Train SAM on protective equipment dataset for better segmentation
2. **Multi-person support**: Handle images with multiple workers
3. **Video processing**: Extend to video with temporal consistency
4. **Active learning**: Use human corrections to improve mapping logic
5. **Web interface**: Build simple UI for easy human review

---

## References

- [Segment Anything Model 2 (SAM2)](https://github.com/facebookresearch/segment-anything-2)
- [Ultralytics YOLOv8 Pose](https://docs.ultralytics.com/tasks/pose/)
- [COCO Keypoint Format](https://cocodataset.org/#keypoints-2020)

---

## Notes

- This is a **PoC** - prioritize working prototype over production optimization
- SAM produces **"visually distinguishable regions"**, not semantic classes
- Pose keypoints provide the **semantic meaning** to anonymous masks
- Expect some errors - **human review is part of the intended workflow**
- Draft labels should **reduce labeling time**, not eliminate human involvement
