from typing import Any, List, Tuple
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras


def convolution(features: int,
                k_size: int,
                strides: int = 1,
                bias: bool = True,
                name: str = None):
    return keras.layers.Conv2D(features,
                               k_size,
                               strides,
                               use_bias=bias,
                               padding="same",
                               name=name)

def psnr(x1, x2):
    return tf.image.psnr(x1, x2, max_val=255)

def ssim(x1, x2):
    return tf.image.ssim(x1,x2,max_val=255)


# run lr through the model and output sr
def convert(model, lr):
    lr = tf.cast(lr, tf.float32)
    sr = model(lr)
    sr = tf.clip_by_value(sr, 0, 255)
    sr = tf.round(sr)
    sr = tf.cast(sr, tf.uint8)
    return sr


# evaluate data using the input model
def evaluate(model: 'SRModel', data: List[Tuple[tf.Tensor, tf.Tensor]]) -> Tuple[float, float]:
    """ Perform evaluation on the given model and return a tuple of psnr and ssim
    """
    psnr_values = []
    ssim_values = []

    for lr, hr in data:
        lr, hr = add_num_images(lr), add_num_images(hr)

        sr = convert(model, lr)

        ssim_values.append(ssim(hr, sr)[0])
        psnr_values.append(psnr(hr, sr)[0])
    return tf.reduce_mean(psnr_values), tf.reduce_mean(ssim_values)


def add_num_images(img):
    return img.reshape(1, img.shape[0], img.shape[1], img.shape[2])


class SRModel(keras.Model):
    """Base Class for all of the modes"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def call(self, inputs, training=False):
        raise NotImplementedError


class DescriminatorBlock(tf.Module):
    """Defines a single descriminator block"""
    def __init__(
        self,
        conv_f: convolution,
        features: int,
        k_size: int,
        strides: int = 1,
        bias: bool = True,
        norm: bool = True,
        activation: keras.layers.Activation = keras.layers.ReLU(),
        name: str = None,
    ):
        super().__init__(name=name)

        block_layers = []
        block_layers.append(
            conv_f(features, k_size, strides=strides, bias=bias))

        if norm:
            block_layers.append(keras.layers.BatchNormalization())

        block_layers.append(activation)

        self.block = keras.Sequential(layers=block_layers)

    def __call__(self, inputs) -> Any:
        return self.block(inputs)


class ResBlock(tf.Module):
    """Defines a single residual block"""
    def __init__(
        self,
        conv_f: convolution,
        features: int,
        k_size: int,
        bias: bool = True,
        norm: bool = False,
        activation=keras.layers.ReLU(),
        residual_scale: int = 1,
        name: str = None,
    ):
        super().__init__(name=name)

        block_layers = []
        for i in range(2):
            block_layers.append(conv_f(features, k_size, bias=bias))

            if norm:
                block_layers.append(keras.layers.BatchNormalization())

            if i == 0:
                block_layers.append(activation)

        self.block = keras.Sequential(layers=block_layers)
        self.residual_scale = residual_scale

    def __call__(self, inputs) -> Any:
        return tf.multiply(self.block(inputs), self.residual_scale)


class MeanShift(keras.layers.Layer):
    def __init__(
            self,
            rgb_mean=(0.4488, 0.4371, 0.4040),
            rgb_std=(1.0, 1.0, 1.0),
            sign=-1,
            **kwargs,
    ):
        # TODO: Check this
        super().__init__(**kwargs)
        mean = tf.constant(rgb_mean, np.float32)
        std = tf.constant(rgb_std, np.float32)

        self.bias = sign * mean / std

    def __call__(self, inputs):
        return inputs + self.bias


class PixelShuffler(keras.layers.Layer):
    def __init__(self, factor: int, name: str = None) -> None:
        super().__init__(name=name)

        self.pixel_shuffler = lambda x: tf.nn.depth_to_space(x, factor)

    def __call__(self, inputs) -> Any:
        return self.pixel_shuffler(inputs)


class UpSampler(keras.Sequential):
    """Defines a upsample sequence"""
    def __init__(
        self,
        convolution: convolution,
        scale: int,
        features: int,
        norm: bool = False,
        activation: keras.layers.Activation = None,
        bias: bool = True,
        name: str = None,
    ):
        def __upsample_base(l: List[keras.layers.Layer],
                            factor: int,
                            name: str = None):
            # TODO: Fix this convolution features size (Check this)
            l.append(
                convolution((factor**2) * features,
                            3,
                            bias=bias,
                            name=f"{name}_convolution"))

            l.append(
                PixelShuffler(factor=factor, name=f"{name}_pixel_shuffler"))

            if norm:
                l.append(
                    keras.layers.BatchNormalization(
                        name=f"{name}_normalization"))

            if activation:
                l.append(activation)

            return l

        layers = []

        if scale == 1:
            pass
        elif scale == 2:
            layers = __upsample_base(layers,
                                     factor=2,
                                     name="upsample_1_scale_2")
        elif scale == 3:
            layers = __upsample_base(layers,
                                     factor=3,
                                     name="upsample_1_scale_3")
        elif scale == 4:
            layers = __upsample_base(layers,
                                     factor=2,
                                     name="upsample_1_scale_2")
            layers = __upsample_base(layers,
                                     factor=2,
                                     name="upsample_3_scale_2")
        else:
            raise ValueError(
                f"Scale must be either 2, 3, or 4. The set scale was: {scale}")

        super().__init__(layers=layers, name=name)
