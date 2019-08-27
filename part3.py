from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, \
    AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.initializers import glorot_uniform
from matplotlib.pyplot import imshow

import numpy as np
import imageio

import keras.backend as K
K.set_image_data_format('channels_last')
# K.set_learning_phase(1)

from part1 import identity_block
from part2 import convolutional_block
import datetime

import resnets_utils



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

def Train_Model():
    # print(datetime.datetime.now())
    # 实体化并编译
    model = ResNet50(input_shape=(64, 64, 3), classes=6)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = resnets_utils.load_dataset()

    # Normalize image vectors
    X_train = X_train_orig / 255.
    X_test = X_test_orig / 255.

    # Convert training and test labels to one hot matrices
    Y_train = resnets_utils.convert_to_one_hot(Y_train_orig, 6).T
    Y_test = resnets_utils.convert_to_one_hot(Y_test_orig, 6).T

    print("number of training examples = " + str(X_train.shape[0]))
    print("number of test examples = " + str(X_test.shape[0]))
    print("X_train shape: " + str(X_train.shape))
    print("Y_train shape: " + str(Y_train.shape))
    print("X_test shape: " + str(X_test.shape))
    print("Y_test shape: " + str(Y_test.shape))

    """
    number of training examples = 1080
    number of test examples = 120
    X_train shape: (1080, 64, 64, 3)
    Y_train shape: (1080, 6)
    X_test shape: (120, 64, 64, 3)
    Y_test shape: (120, 6)
    """

    # 训练
    model.fit(X_train, Y_train, epochs=40, batch_size=32)

    # 评估模型
    preds = model.evaluate(X_test, Y_test)

    print("误差值 = " + str(preds[0]))
    print("准确率 = " + str(preds[1]))

    # 保存模型
    # model.save('test1.h5')
    # print(datetime.datetime.now())

def Load_Model(themodel):
    model = load_model(themodel)

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

    return model

def Predict_Model(the_trained_model,img_path):

    model = Load_Model(the_trained_model)

    img_path = img_path

    my_img = image.load_img(img_path, target_size=(64,64))
    my_img = image.img_to_array(my_img)

    my_img = np.expand_dims(my_img, axis=0)
    my_img = preprocess_input(my_img)

    print("my_image.shape = " + str(my_img.shape))
    img = imageio.imread(img_path)
    imshow(img)
    print("class prediction vector [p(0), p(1), p(2), p(3), p(4), p(5)] = ")
    print(model.predict(my_img))

if __name__ == '__main__':
    Predict_Model('ResNet50_1.h5','images/my_image.jpg')
