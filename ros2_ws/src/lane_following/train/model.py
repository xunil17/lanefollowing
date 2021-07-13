from keras.utils import plot_model
from keras import backend as K
from keras.optimizers import RMSprop, Adam
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Activation, ELU, Flatten, Add, Multiply, ReLU, Reshape, Softmax
from keras.layers import Input, Lambda, Concatenate, Permute, BatchNormalization
from keras.models import Model
import keras

def build_openpilot_model():
    vision_input = Input(shape=(70,320,3), dtype='float32', name='vision')
    desire = Input(shape=(8,), dtype='float32', name='desire')  # NEW in v0.6.6
    rnn_state = Input(shape=(512,), dtype='float32', name='rnn_state')

    # After permutation, the output shape will be (128, 256, 6)
    # vision_permute = Permute((2,3,1), input_shape=(6,128,256), name='vision_permute')(vision_input)
    vision_conv2d = Conv2D(8, 5, strides=1, padding="same", name='vision_conv2d')(vision_input)


    vision_elu = ELU(alpha=1.0, name='vision_elu')(vision_conv2d)

    # vision_max_pooling2d = MaxPooling2D(pool_size=3, strides=2, padding="same", name='vision_max_pooling2d')(vision_elu)
    vision_average_pooling2d = AveragePooling2D(pool_size=3, strides=2, padding="same", name='vision_average_pooling2d')(vision_elu)  # CHANGE in v0.6.6

    # Conv2 block1 - lane0
    vision_conv2_block1_0_conv = Conv2D(16, 1, strides=1, padding="same", name='vision_conv2_block1_0_conv')(vision_average_pooling2d)
    # Conv2 block1 - lane1
    vision_conv2_block1_1_conv = Conv2D(16, 3, strides=1, padding="same", name='vision_conv2_block1_1_conv')(vision_average_pooling2d)
    vision_conv2_block1_1_elu = ELU(alpha=1.0, name='vision_conv2_block1_1_elu')(vision_conv2_block1_1_conv)
    vision_conv2_block1_2_conv = Conv2D(16, 3, strides=1, padding="same", name='vision_conv2_block1_2_conv')(vision_conv2_block1_1_elu)
    vision_conv2_block1_2_elu = ELU(alpha=1.0, name='vision_conv2_block1_2_elu')(vision_conv2_block1_2_conv)
    vision_conv2_block1_add = Add(name='vision_conv2_block1_add')([vision_conv2_block1_0_conv, vision_conv2_block1_2_elu])
    vision_conv2_block1_out = ELU(alpha=1.0, name='vision_conv2_block1_out')(vision_conv2_block1_add)
    # Conv2 block2 - lane1
    vision_conv2_block2_1_conv = Conv2D(16, 3, strides=1, padding="same", name='vision_conv2_block2_1_conv')(vision_conv2_block1_out)
    vision_conv2_block2_1_elu = ELU(alpha=1.0, name='vision_conv2_block2_1_elu')(vision_conv2_block2_1_conv)
    vision_conv2_block2_2_conv = Conv2D(16, 3, strides=1, padding="same", name='vision_conv2_block2_2_conv')(vision_conv2_block2_1_elu)
    vision_conv2_block2_2_elu = ELU(alpha=1.0, name='vision_conv2_block2_2_elu')(vision_conv2_block2_2_conv)
    vision_conv2_block2_add = Add(name='vision_conv2_block2_add')([vision_conv2_block1_out, vision_conv2_block2_2_elu])
    vision_conv2_block2_out = ELU(alpha=1.0, name='vision_conv2_block2_out')(vision_conv2_block2_add)

    # Conv3 block1 - lane0
    vision_conv3_block1_0_conv = Conv2D(32, 1, strides=2, padding="same", name='vision_conv3_block1_0_conv')(vision_conv2_block2_out)
    # Conv3 block1 - lane1
    vision_conv3_block1_1_conv = Conv2D(32, 3, strides=2, padding="same", name='vision_conv3_block1_1_conv')(vision_conv2_block2_out)
    vision_conv3_block1_1_elu = ELU(alpha=1.0, name='vision_conv3_block1_1_elu')(vision_conv3_block1_1_conv)
    vision_conv3_block1_2_conv = Conv2D(32, 3, strides=1, padding="same", name='vision_conv3_block1_2_conv')(vision_conv3_block1_1_elu)
    vision_conv3_block1_2_elu = ELU(alpha=1.0, name='vision_conv3_block1_2_elu')(vision_conv3_block1_2_conv)
    vision_conv3_block1_add = Add(name='vision_conv3_block1_add')([vision_conv3_block1_0_conv, vision_conv3_block1_2_elu])
    vision_conv3_block1_out = ELU(alpha=1.0, name='vision_conv3_block1_out')(vision_conv3_block1_add)
    # Conv3 block2 - lane1
    vision_conv3_block2_1_conv = Conv2D(32, 3, strides=1, padding="same", name='vision_conv3_block2_1_conv')(vision_conv3_block1_out)
    vision_conv3_block2_1_elu = ELU(alpha=1.0, name='vision_conv3_block2_1_elu')(vision_conv3_block2_1_conv)
    vision_conv3_block2_2_conv = Conv2D(32, 3, strides=1, padding="same", name='vision_conv3_block2_2_conv')(vision_conv3_block2_1_elu)
    vision_conv3_block2_2_elu = ELU(alpha=1.0, name='vision_conv3_block2_2_elu')(vision_conv3_block2_2_conv)
    vision_conv3_block2_add = Add(name='vision_conv3_block2_add')([vision_conv3_block1_out, vision_conv3_block2_2_elu])
    vision_conv3_block2_out = ELU(alpha=1.0, name='vision_conv3_block2_out')(vision_conv3_block2_add)

    # Conv4 block1 - lane0
    vision_conv4_block1_0_conv = Conv2D(48, 1, strides=2, padding="same", name='vision_conv4_block1_0_conv')(vision_conv3_block2_out)
    # Conv4 block1 - lane1
    vision_conv4_block1_1_conv = Conv2D(48, 3, strides=2, padding="same", name='vision_conv4_block1_1_conv')(vision_conv3_block2_out)
    vision_conv4_block1_1_elu = ELU(alpha=1.0, name='vision_conv4_block1_1_elu')(vision_conv4_block1_1_conv)
    vision_conv4_block1_2_conv = Conv2D(48, 3, strides=1, padding="same", name='vision_conv4_block1_2_conv')(vision_conv4_block1_1_elu)
    vision_conv4_block1_2_elu = ELU(alpha=1.0, name='vision_conv4_block1_2_elu')(vision_conv4_block1_2_conv)
    vision_conv4_block1_add = Add(name='vision_conv4_block1_add')([vision_conv4_block1_0_conv, vision_conv4_block1_2_elu])
    vision_conv4_block1_out = ELU(alpha=1.0, name='vision_conv4_block1_out')(vision_conv4_block1_add)
    # Conv4 block2 - lane1
    vision_conv4_block2_1_conv = Conv2D(48, 3, strides=1, padding="same", name='vision_conv4_block2_1_conv')(vision_conv4_block1_out)
    vision_conv4_block2_1_elu = ELU(alpha=1.0, name='vision_conv4_block2_1_elu')(vision_conv4_block2_1_conv)
    vision_conv4_block2_2_conv = Conv2D(48, 3, strides=1, padding="same", name='vision_conv4_block2_2_conv')(vision_conv4_block2_1_elu)
    vision_conv4_block2_2_elu = ELU(alpha=1.0, name='vision_conv4_block2_2_elu')(vision_conv4_block2_2_conv)
    vision_conv4_block2_add = Add(name='vision_conv4_block2_add')([vision_conv4_block1_out, vision_conv4_block2_2_elu])
    vision_conv4_block2_out = ELU(alpha=1.0, name='vision_conv4_block2_out')(vision_conv4_block2_add)

    # Conv5 block1 - lane0
    vision_conv5_block1_0_conv = Conv2D(64, 1, strides=2, padding="same", name='vision_conv5_block1_0_conv')(vision_conv4_block2_out)
    # Conv5 block1 - lane1
    vision_conv5_block1_1_conv = Conv2D(64, 3, strides=2, padding="same", name='vision_conv5_block1_1_conv')(vision_conv4_block2_out)
    vision_conv5_block1_1_elu = ELU(alpha=1.0, name='vision_conv5_block1_1_elu')(vision_conv5_block1_1_conv)
    vision_conv5_block1_2_conv = Conv2D(64, 3, strides=1, padding="same", name='vision_conv5_block1_2_conv')(vision_conv5_block1_1_elu)
    vision_conv5_block1_2_elu = ELU(alpha=1.0, name='vision_conv5_block1_2_elu')(vision_conv5_block1_2_conv)
    vision_conv5_block1_add = Add(name='vision_conv5_block1_add')([vision_conv5_block1_0_conv, vision_conv5_block1_2_elu])
    vision_conv5_block1_out = ELU(alpha=1.0, name='vision_conv5_block1_out')(vision_conv5_block1_add)
    # Conv5 block2 - lane1
    vision_conv5_block2_1_conv = Conv2D(64, 3, strides=1, padding="same", name='vision_conv5_block2_1_conv')(vision_conv5_block1_out)
    vision_conv5_block2_1_elu = ELU(alpha=1.0, name='vision_conv5_block2_1_elu')(vision_conv5_block2_1_conv)
    vision_conv5_block2_2_conv = Conv2D(64, 3, strides=1, padding="same", name='vision_conv5_block2_2_conv')(vision_conv5_block2_1_elu)
    vision_conv5_block2_2_elu = ELU(alpha=1.0, name='vision_conv5_block2_2_elu')(vision_conv5_block2_2_conv)
    vision_conv5_block2_add = Add(name='vision_conv5_block2_add')([vision_conv5_block1_out, vision_conv5_block2_2_elu])
    vision_conv5_block2_out = ELU(alpha=1.0, name='vision_conv5_block2_out')(vision_conv5_block2_add)

    ############# VALIDATED ##############
    # Below are the different parts

    # conv2d 1
    vision_conv2d_1 = Conv2D(4, 1, strides=1, padding="same",
            name='vision_conv2d_1')(vision_conv5_block2_out)
    
    vision_elu_1 = ELU(alpha=1.0, name='vision_elu_1')(vision_conv2d_1)
    # vision_elu_1 = ELU(alpha=1.0, name='vision_elu_1')(vision_concatenate)
    dense = Dense(512, name='dense')(Flatten()(vision_elu_1))

    ### Meta
    meta_dense_1 = Dense(256, name='meta_dense_1')(dense)
    meta_relu_1 = ReLU(name='meta_relu_1')(meta_dense_1)
    meta_dense_2 = Dense(128, name='meta_dense_2')(meta_relu_1)
    meta_relu_2 = ReLU(name='meta_relu_2')(meta_dense_2)
    meta_dense_3 = Dense(64, name='meta_dense_3')(meta_relu_2)
    meta_relu_3 = ReLU(name='meta_relu_3')(meta_dense_3)
    meta_dense_4 = Dense(64, name='meta_dense_4')(meta_relu_3)
    meta_relu_4 = ReLU(name='meta_relu_4')(meta_dense_4)

    desire_final_dense = Dense(32, name='desire_final_dense')(meta_relu_4)
    desire_reshape = Reshape((4, 8), name='desire_reshape')(desire_final_dense)
    softmax_ = Softmax(name='softmax')(desire_reshape)
    flatten_ = Flatten(name='flatten')(softmax_)
    snpe_pleaser2 = Dense(32, name='snpe_pleaser2')(flatten_)
    
    dense_1 = Dense(1, name='dense_1')(meta_relu_4)

    model = Model(inputs=[vision_input], outputs=dense_1, name='openpilot_model')

    plot_model(model, to_file='openpilot_model.png')

    return model