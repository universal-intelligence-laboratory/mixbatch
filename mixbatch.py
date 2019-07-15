import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.layers import Layer

class MixBatch(Layer):
    def __init__(self,batch_size, alpha=0.2, **kwargs) -> None:
        self.alpha = alpha
        self.batch_size = batch_size
        super().__init__(**kwargs)

    def get_beta(self,input_shape):
        beta_shape = [1] * len(list(input_shape))
        beta_shape[0] = self.batch_size
        return tf.distributions.Beta(self.alpha, self.alpha).sample(beta_shape)

    def mixup_tf(self, features):
        # do mixup here
        # tensorflow version
        print("features",features.shape)
        input_shape = K.int_shape(features)
        mix = self.get_beta(features.shape)
        mix = tf.maximum(mix, 1 - mix)  # contrl to let data close to x1
        features = features * mix + features[::-1] * (1 - mix)
        return features

    def call(self, x,training=None, **kwargs):
        if training:
            return self.mixup_tf(x)
        else:
            return x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'alpha': self.alpha,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))



# class KernelAttention(conv1d):
#     #https://eccv2018.org/openaccess/content_ECCV_2018/papers/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.pdf
