
import tensorflow as tf
from keras.layers import Layer


class ResizeImageLayer(Layer):
    def __init__(self, output_dim, method=tf.image.ResizeMethod.BILINEAR, **kwargs):
        super(ResizeImageLayer, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.method = method

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim[0], self.output_dim[1], input_shape[3]

    def call(self, inputs):
        output = tf.image.resize_images(inputs, self.output_dim, method=self.method)
        return output

    def get_config(self):
        config = {'output_dim': self.output_dim}
        base_config = super(ResizeImageLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
