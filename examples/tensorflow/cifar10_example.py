import numpy as np
import os
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION']='1'
# from  matplotlib import pyplot as plt
np.random.seed(6) #随机数，这样做的目的是在每次运行程序时，初始值保持一致。seed的值可以随便设置。
import keras
from keras.models import Sequential
from keras.layers import Dense , Dropout ,Activation, Flatten
from keras.layers import Conv2D ,MaxPooling2D   #卷积层，池化层
from keras.utils import np_utils
from keras.datasets import mnist
from mixbatch import MixBatch
import tensorflow as tf
from keras.datasets import cifar10
import keras.backend.tensorflow_backend as KTF
import numpy as np
np.random.seed(2019)
tf.set_random_seed(2019)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
sess = tf.Session(config=config)

KTF.set_session(sess)

def get_default_callbacks(log_dir=None,comment=''):
    import time
    import os

    if log_dir==None:
        def _get_current_date():
            strDate = time.strftime('%Y%m%d_%H%M%S',
                                    time.localtime(time.time())) 
            return strDate
        log_dir = os.path.join('./',_get_current_date()+'_cifar10_train_test_exchange_'+comment)

    cb = []
    weight_path = os.path.join(log_dir, 'model.h5')
    ckpt_cb = keras.callbacks.ModelCheckpoint(weight_path,
                                            save_weights_only=True,
                                            save_best_only=True)
    cb.append(ckpt_cb)

    tb_cb = keras.callbacks.TensorBoard(log_dir=log_dir)
    cb.append(tb_cb)

    # nan_cb = keras.callbacks.TerminateOnNaN()
    # cb.append(nan_cb)

    # update_pruning = sparsity.UpdatePruningStep(),
    # pruning_summary = ssparsity.PruningSummaries(log_dir=log_dir, profile_batch=0)
    # cb.append(update_pruning)
    # cb.append(pruning_summary)
    return cb



batch_size = 32
num_classes = 10
epochs = 100
data_augmentation = False
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(MixBatch(batch_size=batch_size))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

x_train,x_test = x_test,x_train
y_train,y_test = y_test,y_train
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


cb = get_default_callbacks(comment='c1')

print('Not using data augmentation.')
model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_test, y_test),
            shuffle=True,callbacks=cb)
