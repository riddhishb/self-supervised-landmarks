# self-supervised-landmarks
Repository for self-supervised landmark discovery


### Requirements

- pytorch 
- pynrrd (for 3d images)

### Usage

The use of this models is via config files, an example config file for Shepp-logan phantom dataset is given in ./configs/phantom_data.json

To train the model

```
python train.py --model="2d or 3d" --config_file="path to config file"
```

The network could be 2d or 3d and the second argument is the config file path. all the other parameters including the save and data director is inside teh config file

For inference 

```
python test.py --model="2d or 3d" --config_file="path to config file" --redu_remove --use_best --num_out=num ts to be retained
```

redu_remove is a boolean argument that determines if redundant points are removed or not
use best is  also a boolean argument that determines if the best checkpoint is used or the final checkpoint.
num_out is an integer that determines the number of particles to be retained after redundancy removal

### Reference

If you are utilizing this code please cite one of the following

1. [Leveraging unsupervised image registration for discovery of landmark shape descriptor](https://www.sciencedirect.com/science/article/abs/pii/S1361841521002036)

```
@article{bhalodia2021leveraging,
title = {Leveraging unsupervised image registration for discovery of landmark shape descriptor},
journal = {Medical Image Analysis},
volume = {73},
pages = {102157},
year = {2021},
issn = {1361-8415},
doi = {https://doi.org/10.1016/j.media.2021.102157},
url = {https://www.sciencedirect.com/science/article/pii/S1361841521002036},
author = {Riddhish Bhalodia and Shireen Elhabian and Ladislav Kavan and Ross Whitaker}
}
```

2. [Self-supervised discovery of anatomical shape landmarks](http://link.springer.com/chapter/10.1007/978-3-030-59719-1_61)

```
@inproceedings{bhalodia2020self,
  title={Self-supervised discovery of anatomical shape landmarks},
  author={Bhalodia, Riddhish and Kavan, Ladislav and Whitaker, Ross T},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={627--638},
  year={2020},
  organization={Springer}
}
```
