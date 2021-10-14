import tensorflow.keras as keras


class SRModel(keras.Model):
    """Base Class for all of the modes"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def call(self, inputs, training=False):
        raise NotImplementedError
