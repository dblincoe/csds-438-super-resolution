from typing import Any
import tensorflow as tf
import tensorflow.keras as keras

from models.common import MeanShift, ResBlock, UpSampler, convolution, SRModel


class EDSR(SRModel):
    """Defines EDSR module"""

    def __init__(
        self,
        n_resblocks=16,
        n_features=64,
        scale=4,
        conv_f: convolution = convolution,
    ):
        super().__init__(name="edsr_model")

        k_size = 3

        self.normalize = MeanShift()
        self.denormalize = MeanShift(sign=1)

        # Build Head
        head_module = [conv_f(n_features, k_size, name="head_conv")]

        # Build Body
        body_module = [
            ResBlock(
                conv_f, n_features, k_size, residual_scale=0.1, name=f"body_resblock_{i}"
            )
            for i in range(n_resblocks)
        ]
        body_module.append(conv_f(n_features, k_size, name="body_final_conv"))

        # Build Tail
        tail_module = [
            UpSampler(conv_f, scale, n_features, name="tail_upsample"),
            conv_f(3, k_size, name="tail_conv"),
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
