# NaviFormer: A Spatio-Temporal Context-Aware Transformer for Object Navigation.


## 1. Environment Dependencies
This work requires ​Python 3.7 and ​PyTorch 1.8.1, and relies on [habitat-sim_v0.2.2](https://github.com/facebookresearch/habitat-sim).  
You need to install the semantic segmentation model [RedNet](https://github.com/JindongJiang/RedNet) and [detectron2](https://github.com/facebookresearch/detectron2/), and download the [pre-trained weights](https://drive.google.com/file/d/1U0dS44DIPZ22nTjw0RfO431zV-lMPcvv/view?usp=share_link) in 'RedNet/model' path.

## 2. Data
The ​[HM3D](https://aihabitat.org/datasets/hm3d/) dataset must be downloaded following the provided [instructions](https://github.com/facebookresearch/habitat-sim/blob/089f6a41474f5470ca10222197c23693eef3a001/datasets/HM3D.md).

## 3. Execution
Run 
```
python main.py.
```
[Pre-trained weights](https://drive.google.com/file/d/1CFXZFVzJjJwz4HZZ2ANn_ESfHsAUGEA5/view?usp=drive_link) for this work are also available for reference.

## Acknowledgments
This work references implementations from [​SemExp](https://github.com/devendrachaplot/Object-Goal-Navigation) and [​L3MVN](https://github.com/ybgdgh/L3MVN). We extend our sincere gratitude to their contributions!


