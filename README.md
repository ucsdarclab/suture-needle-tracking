# Suture Needle Tracking

This repository contains the implementation of the suture-needle-tracking method proposed in [this paper](https://arxiv.org/abs/2109.12722). 

## Installation

1. We provide a yaml file which includes all the dependencies and required packages. 
To create a conda environment with this file, run 

    ```bash
    conda env create --file needle_tracking_env.yaml
    ```

2. Clone this repository and run 

    ```bash
    cd suture-needle-tracking
    ```

## Run tracking on a test dataset

We provide a test dataset [here](https://drive.google.com/file/d/1H6Hzb_w4B19WqQQUz35PeNQ3osIdHcgQ/view?usp=sharing). 
After downloading the dataset, move the data folder into the project's base folder. 

To run tracking on this dataset: 

```bash
bash run.sh
```

## Citation

```
@article{chiu2021markerless,
  title={Markerless Suture Needle 6D Pose Tracking with Robust Uncertainty Estimation for Autonomous Minimally Invasive Robotic Surgery},
  author={Chiu, Zih-Yun and Liao, Albert Z and Richter, Florian and Johnson, Bjorn and Yip, Michael C},
  journal={arXiv preprint arXiv:2109.12722},
  year={2021}
}
```
