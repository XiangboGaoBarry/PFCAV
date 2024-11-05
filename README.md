# STAMP: Scalable Task And Model-agnostic Collaborative Perception

This repository provides an open heterogeneous multi-agent collaborative perception framework for autonomous driving.

Our framework supports:

- [x] **Heterogeneous Modalities:** Each agent can be equipped with sensors of different modalities.  
  - [x] LiDAR  
  - [x] Camera  
  - [x] LiDAR + Camera  

- [x] **Heterogeneous Model Architectures and Parameters:** Each agent can be equipped with different model architectures.  
  - [x] **Encoder**  
    - [x] PointPillars (LiDAR)  
    - [x] SECOND (LiDAR)  
    - [x] Pixor (LiDAR)  
    - [x] VoxelNet (LiDAR)  
    - [x] Lift-Splat-Shoot [ResNet] (Camera)  
    - [x] Lift-Splat-Shoot [EfficientNet] (Camera)  
  - [x] **Fusion model**  
    - [x] Window Attention first proposed by [V2X-ViT (ECCV 2022)](https://github.com/DerrickXuNu/v2x-vit)  
    - [x] Pyramid Fusion first proposed by [HEAL (ICLR 2024)](https://openreview.net/forum?id=KkrDUGIASk)  
    - [x] Fused Axial Attention first proposed by [CoBevt (PMLR 2023)](https://github.com/DerrickXuNu/CoBEVT)  
    - [x] Cross-Vehicle Aggregation first proposed by [V2VNet (ECCV 2022)](https://arxiv.org/abs/2008.07519)

- [x] **Heterogeneous Downstream Tasks:** Each agent can be trained towards various downstream tasks (training objectives).  
  - [x] 3D Object Detection  
  - [x] BEV Segmentation  

- [x] **Multiple Datasets:**
  - [x] [OPV2V](https://github.com/DerrickXuNu/OpenCOOD)
  - [x] [OPV2V-H](https://huggingface.co/datasets/yifanlu/OPV2V-H)
  - [x] [V2V4Real](https://github.com/ucla-mobility/V2V4Real)

## Future Work

We are committed to expanding our framework's capabilities. Future updates will include support for:
- Additional modalities
- New model architectures
- Diverse downstream tasks
- More datasets

## Getting Started

### Data Preparation

For data and environment preparation, please refer to the [HEAL repository](https://github.com/yifanlu0227/HEAL).

### Training

To reproduce our results, use the following commands:

#### 3D Object Detection on OPV2V Dataset
```bash
bash train_object_detection.sh
```

#### 3D Object Detection on V2V4Real Dataset
```bash
bash train_v2v4real.sh
```

#### Task- and Model-Agnostic Setting on OPV2V Dataset
```bash
bash task_agnostic.sh
```

### Checkpoints

We are in the process of preparing model checkpoints for release. Please stay tuned for updates.

## Acknowledgements

This project builds upon the excellent work of [HEAL](https://github.com/yifanlu0227/HEAL). We extend our sincere gratitude to their team for their outstanding contributions to the field.

## Contributing and Contact

For the purpose of double blind review, we will release the contact information later. 
