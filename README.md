Always Clear Depth: Robust Monocular Depth Estimation under Adverse Weather
====================================
This repository holds the code and data for the paper:

[**Always Clear Depth: Robust Monocular Depth Estimation under Adverse Weather**, IJCAI 2025.]()\
Kui Jiang, Jing Cao, Zhaocheng Yu, Junjun Jiang, Jingchun Zhou



Environment Setup
------------------  
We implement our method within the md4all, and the environment is the same as md4all. Therefore, you can refer to:
- [**md4all**(ICCV2023)](https://github.com/md4all/md4all)


Datasets
------
You can prepare the nuScenes dataset and RobotCar dataset by referring to [**md4all**(ICCV2023)](https://github.com/md4all/md4all).
You can download the translation datasets for nuScenes and RobotCar from [here](https://drive.google.com/drive/u/0/folders/1jL8b7l1kTVSQjnF2hFiVUEGcgJYScuAZ)


Training
---------

Before training ACDepth, you need to prepare the teacher model, which is a self-supervised model trained on clear images, just like the baseline in md4all. In addition, you also need to prepare the small version of the [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2?tab=readme-ov-file). Note this is a relative depth estimation model.

**nuScenes Dataset** :

```
python train.py --config <PATH>/config/train_ACDepth_nuscenes.yaml
```
**RobotCar Dataset** :

```
python train.py --config <PATH>/config/train_ACDepth_robotcar.yaml
``` 


Evaluation
---------

You can download the model from [here](https://drive.google.com/drive/u/0/folders/1xOHo4-stl6P1gvatZyiZ1qoiCIwKc-Vt)
**nuScenes Dataset** :

```
python train.py --config <PATH>/config/eval_ACDepth_nuscenes.yaml
```
**RobotCar Dataset** :

```
python train.py --config <PATH>/config/eval_ACDepth_robotcar.yaml
``` 

Testing
---------
**nuScenes Dataset** :

```
python train.py --config <PATH>/config/test_ACDepth_nuscenes.yaml
```
**RobotCar Dataset** :

```
python train.py --config <PATH>/config/test_ACDepth_robotcar.yaml
``` 


Thanks
--------

Our method is based on [md4all](https://github.com/md4all/md4all), [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) and [CycleGAN-Turbo](https://github.com/GaParmar/img2img-turbo). You can refer to their README files and source code for more implementation details. We thank the authors for their contributions.

License
--------
This repository is released under the MIT Licence license as found in the LICENSE file.
