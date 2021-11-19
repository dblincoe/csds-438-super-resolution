import os

# from models.edsr import EDSR
from models.common import add_num_images, evaluate
from data.data_loaders import load_test_images, load_images_from_folder
from data.data_utils import downsample_images, resize_images

import tensorflow as tf
from tensorflow.keras.losses import (
    Loss,
    MeanAbsoluteError,
    MeanSquaredError,
)
from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from tensorflow.keras.optimizers import Adam

# from models.sr_model import SRModel
from models.srgan import SRResNet
from models.edsr import EDSR


class Trainer:
    def __init__(self,
                 model: EDSR(),
                 loss: Loss,
                 learning_rate: None,
                 checkpoint_dir_base: str = "./ckpts/") -> None:

        self.model = model
        self.loss = loss

        self.optimizer = Adam(learning_rate)

        self.checkpoint = tf.train.Checkpoint(model=model,
                                              optimizer=Adam(learning_rate))

        self.checkpoint_mngr = tf.train.CheckpointManager(
            checkpoint=self.checkpoint,
            directory=os.path.join(checkpoint_dir_base, model.name),
            max_to_keep=2,
        )

        self.rebuild()

    def train(self, train_data, valid_data, epochs):
        """Trains the model using a training and validation set"""

        for epoch in range(epochs):
            loss_mean = Mean()

            for lr_img, hr_img in train_data:
                lr_img, hr_img = add_num_images(lr_img), add_num_images(hr_img)
                
                loss = self.train_step(lr_img, hr_img)
                loss_mean(loss)

            self.checkpoint_mngr.save()

            print(f"Epoch {epoch+1} loss: {loss_mean.result().numpy()}")

        # Compute PSNR on validation dataset
        psnr_value, ssim_value = evaluate(self.model, valid_data)

        print("Done Training!")
        print(
            f'Final Metrics: Loss = {loss_mean.result().numpy()}, PSNR: {psnr_value.numpy()}, SSIM: {ssim_value.numpy()}'
        )

    @tf.function
    def train_step(self, lr, hr):
        with tf.GradientTape() as tape:

            lr, hr = tf.cast(lr, tf.float32), tf.cast(hr, tf.float32)

            # this should be lr going into model and hr in loss
            sr = self.model(lr, training=True)
            loss_value = self.loss(hr, sr)

        gradients = tape.gradient(loss_value, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables))

        return loss_value

    def rebuild(self):
        """Rebuilds the checkpoint"""
        if self.checkpoint_mngr.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_mngr.latest_checkpoint)


trainer = Trainer(model=EDSR(),
                  loss=MeanAbsoluteError(),
                  learning_rate=PiecewiseConstantDecay(boundaries=[200000],
                                                       values=[1e-4, 5e-5]))

hr_imgs = load_images_from_folder("train-data")
hr_imgs = resize_images(hr_imgs, 200, 200)
lr_imgs = downsample_images(hr_imgs, 4)
train_valid_split = int(0.8 * len(lr_imgs))

combined_data = list(zip(lr_imgs, hr_imgs))
train_data = combined_data[:train_valid_split]
valid_data = combined_data[train_valid_split:]

trainer.train(train_data, valid_data, epochs=10)
