import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.layers import Layer
from tensorflow.python.keras.utils.tf_utils import smart_cond

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
        # print("features",features.shape)
        input_shape = K.int_shape(features)
        mix = self.get_beta(features.shape)
        try:
            mix = tf.maximum(mix, 1 - mix)  # contrl to let data close to x1
            features = features * mix + features[::-1] * (1 - mix)
        except:
            pass
        return features

    # def call(self, x,training=None, **kwargs):
    #     if training:
    #         return self.mixup_tf(x)
    #     else:
    #         return x

    def call(self, x, **kwargs):
        # if training is None:
        training = K.learning_phase()
        # print("LCV",tf.to_float(training))
        return smart_cond(training, lambda: self.mixup_tf(x), lambda: x)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'alpha': self.alpha,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


# class MixBatch(Layer):
#     def __init__(self, alpha=0.2, **kwargs) -> None:
#         self.alpha = alpha
#         self.axis = 0
#         super().__init__(**kwargs)

#     def get_beta(self,input_shape,sample_size):
#         beta_shape = [1] * len(list(input_shape))
#         beta_shape[0] = sample_size
#         return tf.distributions.Beta(self.alpha, self.alpha).sample(beta_shape)

#     def build(self, input_shape):
#         super().build(input_shape)

#     def mixup_tf(self, features):

#         input_shape = K.int_shape(features)
#         reduction_axes = list(range(len(input_shape)))
#         del reduction_axes[self.axis]

#         sample_size = K.prod([K.shape(features)[axis]
#                     for axis in reduction_axes])
#         sample_size = K.cast(sample_size, dtype=K.dtype(features))
        
#         if K.backend() == 'tensorflow' and sample_size.dtype != 'float32':
#             sample_size = K.cast(sample_size, dtype='float32')

#         # do mixup here with tensorflow version
#         mix = self.get_beta(features.shape,sample_size)
#         mix = tf.maximum(mix, 1 - mix)  # contrl to let data close to x1
#         features = features * mix + features[::-1] * (1 - mix)
#         return features

#     # @tf.function
#     def call(self, x,training=None, **kwargs):
#         if training is None:
#             training = K.learning_phase()
#         return smart_cond(training, lambda: self.mixup_tf(x), lambda: x)
#         # # if training is None:
#         # training = K.learning_phase()
#         # print(training)

#         # # if training==None:
#         # raise NotImplementedError
#         # if training:
          
#         #     return self.mixup_tf(x)
#         # else:
#         #     return x

#     def compute_output_shape(self, input_shape):
#         return input_shape

#     def get_config(self):
#         config = {
#             'alpha': self.alpha,
#         }
#         base_config = super().get_config()
#         return dict(list(base_config.items()) + list(config.items()))

