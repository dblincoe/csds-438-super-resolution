from datetime import datetime

import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
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

# Create Model
sr_resnet_model = SRDescriminator()
sr_resnet_model.compile(
    optimizer="adam", loss=MeanAbsoluteError(), metrics=["accuracy"]
)
sr_resnet_model.run_eagerly = True

Trainer(sr_resnet_model, MeanSquaredError())

# Load images
hr_imgs = load_test_images()
lr_imgs = downsample_images(hr_imgs, 4)

# Tensorize
hr_imgs = tf.convert_to_tensor(hr_imgs, np.float32)
hr_imgs = (hr_imgs.numpy() - 127.5) / 127.5

lr_imgs = tf.convert_to_tensor(lr_imgs, np.float32)
lr_imgs = (lr_imgs.numpy() - 127.5) / 127.5

one = tf.convert_to_tensor([1], np.float32)

# Fit model
sr_resnet_model.fit(
    lr_imgs, one, batch_size=1, epochs=2, callbacks=[tensorboard_callback]
)
