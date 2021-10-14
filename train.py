import os

import tensorflow as tf
from tensorflow.keras.losses import (
    Loss,
    MeanAbsoluteError,
    MeanSquaredError,
)

from models.sr_model import SRModel


class Trainer:
    def __init__(
        self, model: SRModel, loss: Loss, checkpoint_dir_base: str = "./ckpts/"
    ) -> None:
        self.loss = loss

        self.checkpoint = tf.train.Checkpoint(model=model)

        self.checkpoint_mngr = tf.train.CheckpointManager(
            checkpoint=self.checkpoint,
            directory=os.path.join(checkpoint_dir_base, str(type(model))),
            max_to_keep=2,
        )

        self.checkpoint_mngr.save()

        self.rebuild()

    def rebuild(self):
        """Rebuilds the checkpoint"""
        if self.checkpoint_mngr.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_mngr.latest_checkpoint)
