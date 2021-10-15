from typing import Any
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.keras.layers.core import Flatten

from models.common import (
    DescriminatorBlock,
    MeanShift,
    ResBlock,
    UpSampler,
    convolution,
    SRModel
)


class SRResNet(SRModel):
    """Defines SRResNet model"""

    def __init__(
        self,
        n_resblocks=16,
        n_features=64,
        scale=4,
        conv_f: convolution = convolution,
    ):
        super().__init__(name="sr_resnet_model")

        self.normalize = MeanShift()
        self.denormalize = MeanShift(sign=1)

        # Build Head
        head_module = [conv_f(n_features, k_size=9, name="head_conv")]

        # Build Body
        body_module = [
            ResBlock(
                conv_f,
                n_features,
                k_size=3,
                norm=True,
                name=f"body_resblock_{i}",
                activation=keras.layers.PReLU(),
            )
            for i in range(n_resblocks)
        ]
        body_module.append(conv_f(n_features, k_size=3, name="body_final_conv"))
        body_module.append(keras.layers.BatchNormalization())

        # Build Tail
        tail_module = [
            UpSampler(
                conv_f,
                scale,
                n_features,
                name="tail_upsample",
                activation=keras.layers.PReLU(),
            ),
            conv_f(3, k_size=9, name="tail_conv"),
        ]

        self.head = keras.Sequential(layers=head_module, name="head")
        self.body = keras.Sequential(layers=body_module, name="body")
        self.tail = keras.Sequential(layers=tail_module, name="tail")

    def call(self, inputs) -> Any:
        norm_head = self.normalize(inputs)
        head_result = self.head(norm_head)

        body_result = self.body(head_result)
        body_result += head_result

        tail = self.tail(body_result)
        denorm_tail = self.denormalize(tail)

        return denorm_tail


class SRDescriminator(SRModel):
    """Descriminator for SRGAN"""

    def __init__(
        self,
        n_features: int = 64,
        leaky_alpha: float = 0.2,
        conv_f: convolution = convolution,
    ):
        super().__init__(name="sr-gan-model")

        activation = keras.layers.LeakyReLU(leaky_alpha)

        # Build Head
        head_module = [conv_f(n_features, k_size=3, name="head_conv"), activation]

        # Build Body
        # TODO: Check this
        body_module = [
            DescriminatorBlock(
                conv_f,
                n_features,
                k_size=3,
                norm=i != 1,
                strides=((i + 1) % 2) + 1,
                name=f"body_descriminator_{i}",
                activation=activation,
            )
            for i in range(1, 6)
        ]

        # Build Tail
        tail_module = [
            keras.layers.Flatten(),
            keras.layers.Dense(
                1024,
                name="tail_first_dense",
            ),
            activation,
            keras.layers.Dense(1, activation="sigmoid", name="tail_last_dense"),
        ]

        self.head = keras.Sequential(layers=head_module, name="head")
        self.body = keras.Sequential(layers=body_module, name="body")
        self.tail = keras.Sequential(layers=tail_module, name="tail")

    def call(self, inputs) -> Any:
        head_result = self.head(inputs)

        body_result = self.body(head_result)

        return self.tail(body_result)
