import os

# from models.edsr import EDSR
from models.common import add_num_images, evaluate
from data.data_loaders import load_test_images, load_images_from_folder
from data.data_utils import downsample_images, resize_images, rotate_images

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
                 checkpoint_dir_base: str = "./ckpts/",
                 saved_model_dir_base: str = "./output/") -> None:

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

        self.saved_model_dir_base = saved_model_dir_base

        self.rebuild()

    def train(self, train_data, valid_data, epochs):
        """Trains the model using a training and validation set"""

        # if the total number of runs is greater than epochs, save the model and exit
        run = int(self.checkpoint_mngr.latest_checkpoint.split('-')[1])
        if run > epochs:
            self.model.save(
                os.path.join(self.saved_model_dir_base, self.model.name))

        # else:
        for epoch in range(epochs):

            loss_mean = Mean()

            # TODO: Iterate over batches not individual images
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

        self.model.save(
            os.path.join(self.saved_model_dir_base, self.model.name))

    @tf.function
    def train_step(self, lr, hr):
        with tf.GradientTape() as tape:
            lr, hr = tf.cast(lr, tf.float32), tf.cast(hr, tf.float32)

            sr = self.model(lr, training=True)
            step_loss = self.loss(hr, sr)

            gradients = tape.gradient(step_loss,
                                      self.model.trainable_variables)
            self.optimizer.apply_gradients(
                zip(gradients, self.model.trainable_variables))

        return step_loss

    def rebuild(self):
        """Rebuilds the checkpoint"""
        if self.checkpoint_mngr.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_mngr.latest_checkpoint)


trainer = Trainer(model=SRResNet(),
                  loss=MeanAbsoluteError(),
                  learning_rate=PiecewiseConstantDecay(boundaries=[200000],
                                                       values=[1e-4, 5e-5]))

# TODO: Augment training images by taking smaller swatches and rotating/flipping them
load_hr_imgs = load_images_from_folder("train-data")
orig_hr_imgs = resize_images(load_hr_imgs, 200, 200)
hr_imgs = orig_hr_imgs
for i in range(1, 4):
    rotated_imgs = rotate_images(orig_hr_imgs, i)
    hr_imgs = hr_imgs + rotated_imgs

lr_imgs = downsample_images(hr_imgs, 4)
train_valid_split = int(0.8 * len(lr_imgs))

combined_data = list(zip(lr_imgs, hr_imgs))
train_data = combined_data[:train_valid_split]
valid_data = combined_data[train_valid_split:]
batch_size = 10

for i in range(0, len(train_data), batch_size):
    batch = train_data[i:i + batch_size]
    trainer.train(batch, valid_data, epochs=9)

# trainer.train(train_data, valid_data, epochs=3, batch_size=10)

# TODO: Make a script that can load a model and evaluate/inference images