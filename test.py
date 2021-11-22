import numpy as np
import tensorflow as tf
from cv2 import cv2
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from tensorflow.python.keras.losses import MeanAbsoluteError

from data.data_loaders import load_test_images
from data.data_utils import augment_images, downsample_images
from models.edsr import EDSR
from train import Trainer

# =========================#
#  Testing Model Inference
# =========================#

# Load images
hr_imgs = load_test_images()
hr_imgs = augment_images(hr_imgs, patch_size=(100, 100), max_patches=1)
lr_imgs = downsample_images(hr_imgs, 4)

# Tensorize
hr_imgs = tf.convert_to_tensor(hr_imgs, np.float32)
hr_imgs = hr_imgs.numpy()

lr_imgs = tf.convert_to_tensor(lr_imgs, np.float32)
lr_imgs = lr_imgs.numpy()

train = Trainer(
    EDSR(scale=4),
    loss=MeanAbsoluteError(),
    learning_rate=PiecewiseConstantDecay(boundaries=[200000], values=[1e-4, 5e-5]),
)
model = train.model

i = 0
for sr_image, lr_image, hr_image in zip(model.predict(lr_imgs), lr_imgs, hr_imgs): 
    cv2.imwrite(f"./example-output/lr-{i}.png", lr_image.astype(int))
    cv2.imwrite(f"./example-output/hr-{i}.png", hr_image.astype(int))
    cv2.imwrite(f"./example-output/sr-{i}.png", sr_image.astype(int))

    i += 1
