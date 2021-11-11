# self-supervised-landmarks
Repository for self-supervised landmark discovery


### Requirements

pytorch 
pynrrd (for 3d images)

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
