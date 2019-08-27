from keras.layers import Dense, Flatten, Input, Activation, ZeroPadding2D, AveragePooling2D, BatchNormalization, Conv2D, \
    Add, MaxPooling2D
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
import resnets_utils
import numpy as np
from keras.initializers import glorot_uniform
import imageio
from matplotlib.pyplot import imshow
from keras.models import Model, load_model


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

def convolutional_block(X,f, filters,stage,block,s=2):
    """
       实现图3的恒等块

       参数：
           X - 输入的tensor类型的数据，维度为( m, n_H_prev, n_W_prev, n_H_prev )
           f - 整数，指定主路径中间的CONV窗口的维度
           filters - 整数列表，定义了主路径每层的卷积层的过滤器数量
           stage - 整数，根据每层的位置来命名每一层，与block参数一起使用。
           block - 字符串，据每层的位置来命名每一层，与stage参数一起使用。
           s - 整数，指定要使用的步幅

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
    X = Conv2D(filters=F1, kernel_size=(1,1), strides=(s,s), padding='valid', name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
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

    # 捷径
    X_shortcut = Conv2D(filters=F3, kernel_size=(1,1), strides=(s,s), padding='valid', name=conv_name_base + '1',)(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name= bn_name_base + '1')(X_shortcut)


    # 最后一步
    ## 将捷径与输入加在一起
    X = Add()([X,X_shortcut])
    X = Activation("relu")(X)

    return X

def ResNet50(input_shape=(64, 3, 3), classes=6):
    """
    实现ResNet50
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    参数：
        input_shape - 图像数据集的维度
        classes - 整数，分类数

    返回：
        model - Keras框架的模型
    """

    # 定义tensor类型的输入数据
    # X_input = input(input_shape)
    # X_input = Input(input_shape)
    X_input = Input(input_shape)

    # 0填充
    X = ZeroPadding2D((3, 3))(X_input)

    # Stage 1.
    X = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), name='conv1')(X)
    X = BatchNormalization(axis=3, name='bn1')(X)
    X = Activation("relu")(X)
    X = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(X)

    # Stage 2.
    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block="a", s=1)
    X = identity_block(X, f=3, filters=[64, 64, 256], stage=2, block="b")
    X = identity_block(X, f=3, filters=[64, 64, 256], stage=2, block="c")

    # Stage 3.
    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block="a", s=2)
    X = identity_block(X, f=3, filters=[128, 128, 512], stage=3, block="b")
    X = identity_block(X, f=3, filters=[128, 128, 512], stage=3, block="c")
    X = identity_block(X, f=3, filters=[128, 128, 512], stage=3, block="d")

    # Stage 4.
    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block="a", s=2)
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block="b")
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block="c")
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block="d")
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block="e")
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block="f")

    # Stage 5.
    X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block="a", s=2)
    X = identity_block(X, f=3, filters=[512, 512, 2048], stage=5, block="b")
    X = identity_block(X, f=3, filters=[512, 512, 2048], stage=5, block="c")


    X = AveragePooling2D(pool_size=(2, 2), name="avg_pool")(X)

    # 输出层
    X = Flatten()(X)
    X = Dense(classes, activation="softmax", name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)

    # 创建模型
    model = Model(inputs=X_input, outputs=X, name="ResNet50")

    return model



if __name__ == '__main__':
    model = load_model('test1.h5')

    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = resnets_utils.load_dataset()

    # Normalize image vectors
    X_train = X_train_orig / 255.
    X_test = X_test_orig / 255.

    # Convert training and test labels to one hot matrices
    Y_train = resnets_utils.convert_to_one_hot(Y_train_orig, 6).T
    Y_test = resnets_utils.convert_to_one_hot(Y_test_orig, 6).T

    # 评估模型
    preds = model.evaluate(X_test, Y_test)

    print("误差值 = " + str(preds[0]))
    print("准确率 = " + str(preds[1]))

    img_path = 'images/3.jpg'

    my_img = image.load_img(img_path, target_size=(64, 64))
    my_img = image.img_to_array(my_img)

    my_img = np.expand_dims(my_img, axis=0)
    my_img = preprocess_input(my_img)

    print("my_image.shape = " + str(my_img.shape))
    img = imageio.imread(img_path)
    imshow(img)
    print("class prediction vector [p(0), p(1), p(2), p(3), p(4), p(5)] = ")
    print(model.predict(my_img))


















