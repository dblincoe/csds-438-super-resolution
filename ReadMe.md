# CSDS 438: Super Resolution
This project houses code to train models to perform super resolution over images

## Installation
Given an installation of python >3.7 and updated pip, you should be able to run,
`pip install -r requirements.txt`

It is probably smart to do this inside of a virtual machine

## Training a model
To train a model the `train.py` script can be used. There are to current models that can be trained:
the EDSR and SRResNet model. You can train py running,
`python train.py`

If you want to see the available training options you can run,
`python train.py --help`

## Inferencing over test images
This is still rudementary, but you can add folders to the `test-images` folder. You must have trained an EDSR model with scale `4` and created a folder named `example-output` in the root of the project directory. You can inference patches of the test images by running, 
`python test.py`

## Who supported this project
David Blincoe
    - Created Models
    - Update Util Tools
    - Created inference script

Chris Toomey
    - Implimented training loop
    - Added image augmentation functions

Paul Rodriguez
    - Collected training data
