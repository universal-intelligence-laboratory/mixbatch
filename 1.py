import numpy as np
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
        log_dir = os.path.join('./',_get_current_date()+comment)

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

(X_train,y_train),(X_test,y_test)=mnist.load_data()   #导入mnist数据
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
#这里大家可能不是很了解，主要是为了把mnist数据变成二维图片形式
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255     #做标准化（0，1）之间 ，深度学习对样本值敏感需要做标准化处理
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)
#把标签值变成一维数组显示，例如 标签1 ——0100000000
model = Sequential()
model.add(MixBatch(batch_size=32))
model.add(Conv2D(32, kernel_size=(3, 3),
                   activation='relu',
                   input_shape=(28,28,1)))
model.add(MixBatch(batch_size=32))
#第一层卷积层，32个特征图，每个特征图中的神经元与输入中的3×3的领域相连。参数可调
model.add(Conv2D(64, (3, 3), activation='relu'))   #第二层卷积层,64个特征图。参数可调
model.add(MixBatch(batch_size=32))
model.add(MaxPooling2D(pool_size=(2, 2)))   #池化层 2×2做映射，参数可调
model.add(MixBatch(batch_size=32))
model.add(Dropout(0.25))   #Dropou算法用于测试集的精确度的优化，这个也可以换。
model.add(Flatten())    #把二维数组拉成一维
model.add(Dense(128, activation='relu'))    #正常的神经网络层
model.add(MixBatch(batch_size=32))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax')) #输出层
model.compile(loss='categorical_crossentropy',
                         optimizer='adam',
                          metrics=['accuracy'])   #交叉熵，训练优化算法adam
cb = get_default_callbacks(comment="MBall")
model.fit(X_train, Y_train, batch_size=32, nb_epoch=10, verbose=1,callbacks=cb,validation_data=(X_test, Y_test))  # batch=32,epoch=10,参数可调，总迭代次数大家会算吗？
# score = model.evaluate(, verbose=0)