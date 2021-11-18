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

        # self.checkpoint = tf.train.Checkpoint(model=model,
        #                                       optimizer=Adam(learning_rate))

        # self.checkpoint_mngr = tf.train.CheckpointManager(
        #     checkpoint=self.checkpoint,
        #     directory=os.path.join(checkpoint_dir_base, str(type(model))),
        #     max_to_keep=2,
        # )

        # self.rebuild()

    def train(self, train_data, valid_data, num_steps):
        """Trains the model using a training and validation set"""
        loss_mean = Mean()

        # run training steps on dataset in order to get loss
        for lr_img, hr_img in train_data:
            lr_img, hr_img = add_num_images(lr_img), add_num_images(hr_img)
            loss = self.train_step(lr_img, hr_img)
            loss_mean(loss)
            print(loss_mean.result().numpy())

        # Compute PSNR on validation dataset
        psnr_value = evaluate(self.model, valid_data)

        print(
            f'loss = {loss_mean.result().numpy()}, PSNR = {psnr_value.numpy()}'
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
# hr_imgs = load_test_images()
# lr_imgs = downsample_images(hr_imgs, 4)
# dataset_size = len(hr_imgs)
# train_data = [(lr_imgs[x], hr_imgs[x]) for x in range(dataset_size - 1)]
# valid_data = [(lr_imgs[dataset_size - 1], hr_imgs[dataset_size - 1])]

# trainer.train(train_data, valid_data, 1000)

hr_imgs = load_images_from_folder("train-data")
hr_imgs = resize_images(hr_imgs, 200, 200)
# hr_imgs = downsample_images(hr_imgs, 1)
lr_imgs = downsample_images(hr_imgs, 4)
train_valid_split = int(0.8 * len(lr_imgs))
train_data = [(lr_imgs[x], hr_imgs[x]) for x in range(train_valid_split)]
valid_data = [(lr_imgs[x], hr_imgs[x])
              for x in range(train_valid_split, len(lr_imgs))]

trainer.train(train_data, valid_data, 1000)
