import os
from datetime import datetime

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from cv2 import cv2
from tensorflow.keras.losses import MeanAbsoluteError, MeanSquaredError

from data.data_loaders import load_test_images
from data.data_utils import downsample_images
from models.edsr import EDSR
from models.srgan import SRDescriminator, SRResNet
from train import Trainer

# Define the Keras TensorBoard callback.
logdir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir, write_graph=True)

# =========================#
#  Testing Model Inference
# =========================#

# Load images
hr_imgs = load_test_images()
lr_imgs = downsample_images(hr_imgs, 4)

# Tensorize
hr_imgs = tf.convert_to_tensor(hr_imgs, np.float32)
hr_imgs = (hr_imgs.numpy() - 127.5) / 127.5

lr_imgs = tf.convert_to_tensor(lr_imgs, np.float32)
lr_imgs = (lr_imgs.numpy() - 127.5) / 127.5

one = tf.convert_to_tensor([1]*len(lr_imgs), np.float32)

# Testing SR Descriminator
# sr_descriminator_model = SRDescriminator()
# sr_descriminator_model.compile(
#     optimizer="adam", loss=MeanAbsoluteError(), metrics=["accuracy"]
# )
# sr_descriminator_model.run_eagerly = True

# sr_descriminator_model.fit(
#     lr_imgs, one, batch_size=1, epochs=2, callbacks=[tensorboard_callback]
# )

# Testing SR ResNet
sr_resnet_model = SRResNet()
sr_resnet_model.compile(
    optimizer="adam", loss=MeanAbsoluteError(), metrics=["accuracy"]
)
sr_resnet_model.run_eagerly = True

sr_resnet_model.fit(
    lr_imgs, hr_imgs, batch_size=1, epochs=20, callbacks=[tensorboard_callback]
)

# Testing EDSR
# edsr_model = SRResNet()
# edsr_model.compile(
#     optimizer="adam", loss=MeanAbsoluteError(), metrics=["accuracy"]
# )
# edsr_model.run_eagerly = True

# edsr_model.fit(
#     lr_imgs, hr_imgs, batch_size=5, epochs=10, callbacks=[tensorboard_callback]
# )

i = 0
for sr_image, lr_image, hr_image in zip(sr_resnet_model.predict(lr_imgs),lr_imgs, hr_imgs):
    cv2.imwrite(f'./example-output/lr-{i}.png', (lr_image * 127.5) + 127.5)
    cv2.imwrite(f'./example-output/hr-{i}.png', (hr_image * 127.5) + 127.5)
    cv2.imwrite(f'./example-output/sr-{i}.png', (sr_image * 127.5) + 127.5)

    i += 1