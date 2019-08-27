import numpy as np
import tensorflow as tf

from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform

import pydot
from IPython.display import SVG
import scipy.misc
from matplotlib.pyplot import imshow
import keras.backend as K
# K.set_image_data_format('channels_last')
# K.set_learning_phase(1)

# import resnets_utils



def identity_block(X,f, filters,stage,block):
    """
       实现图3的恒等块

       参数：
           X - 输入的tensor类型的数据，维度为( m, n_H_prev, n_W_prev, n_H_prev )
           f - 整数，指定主路径中间的CONV窗口的维度
           filters - 整数列表，定义了主路径每层的卷积层的过滤器数量
           stage - 整数，根据每层的位置来命名每一层，与block参数一起使用。
           block - 字符串，据每层的位置来命名每一层，与stage参数一起使用。

       返回：
           X - 恒等块的输出，tensor类型，维度为(n_H, n_W, n_C)

       """
    # 定义命名规则
    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base = "bn" + str(stage)+ block + "_branch"

    # 获取过滤器
    F1,F2,F3 = filters

    # 保存输入数据，之后用于向主路径添加捷径
    X_shortcut = X

    # 主路径的第一部分
    ## 卷积层
    X = Conv2D(filters=F1, kernel_size=(1,1), strides=(1,1), padding='valid', name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
    ## 归一化
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    ## 激活函数
    X = Activation("relu")(X)

    # 主路径的第二部分
    X = Conv2D(filters=F2, kernel_size=(f,f), strides=(1,1), padding='same', name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation("relu")(X)

    # 主路径第三部分
    X = Conv2D(filters=F3, kernel_size=(1,1), strides=(1,1), padding='valid',name=conv_name_base + '2c',kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3,name=bn_name_base + '2c')(X)

    # 最后一步
    ## 将捷径与输入加在一起
    X = Add()([X,X_shortcut])
    X = Activation("relu")(X)

    return X



if __name__ == '__main__':
    tf.reset_default_graph()
    with tf.Session() as test:
        np.random.seed(1)
        A_prev = tf.placeholder("float", [3, 4, 4, 6])
        X = np.random.randn(3, 4, 4, 6)
        A = identity_block(A_prev, f=2, filters=[2, 4, 6], stage=1, block="a")

        test.run(tf.global_variables_initializer())
        out = test.run([A], feed_dict={A_prev: X, K.learning_phase(): 0})
        print("out = " + str(out[0][1][1][0]))

        test.close()





















