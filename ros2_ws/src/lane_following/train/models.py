from keras.utils import plot_model
from keras import backend as K
from keras.optimizers import RMSprop, Adam
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Activation, ELU, Flatten, Add, Multiply, ReLU, Reshape, Softmax
from keras.layers import Input, Lambda, Concatenate, Permute, BatchNormalization
from keras.models import Model
import keras

def build_nvidia_model():
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0, input_shape=(70, 320, 3)))
    model.add(Conv2D(24, (5, 5), strides=(2, 2), padding='valid', activation='relu'))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), padding='valid', activation='relu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), padding='valid', activation='relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='valid', activation='relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='valid', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))

    return model

def build_openpilot_model():
    vision_input = Input(shape=(6,128,256), dtype='float32', name='vision')
    desire = Input(shape=(8,), dtype='float32', name='desire')  # NEW in v0.6.6
    rnn_state = Input(shape=(512,), dtype='float32', name='rnn_state')

    # After permutation, the output shape will be (128, 256, 6)
    vision_permute = Permute((2,3,1), input_shape=(6,128,256), name='vision_permute')(vision_input)
    vision_conv2d = Conv2D(8, 5, strides=1, padding="same", name='vision_conv2d')(vision_permute)


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
    main_relu = ReLU(name='main_relu')(dense)  # FIXME: find the meaning of each field

    # desire input
    snpe_desire_pleaser = Dense(8, name='snpe_desire_pleaser')(desire)

    proc_features = Concatenate(name='proc_features')([main_relu, snpe_desire_pleaser])

    # TODO: add a bunch of layers here

    # GRU, TODO: implement using primitives
    # add_3 = GRU(512, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, name='add_3')(reshape)
    rnn_z = Dense(512, name='rnn_z')(proc_features)
    rnn_rz = Dense(512, name='rnn_rz')(rnn_state)
    add = Add(name='add')([rnn_z, rnn_rz])
    activation_1 = Activation('sigmoid', name='activation_1')(add)
    snpe_pleaser = Dense(512, name='snpe_pleaser')(rnn_state)
    rnn_rh = Dense(512, name='rnn_rh')(snpe_pleaser)

    rnn_h = Dense(512, name='rnn_h')(proc_features)
    rnn_r = Dense(512, name='rnn_r')(proc_features)
    rnn_rr = Dense(512, name='rnn_rr')(rnn_state)
    add_1 = Add(name='add_1')([rnn_r, rnn_rr])
    activation = Activation('sigmoid', name='activation')(add_1)

    multiply = Multiply(name='multiply')([rnn_rh, activation])
    add_2 = Add(name='add_2')([rnn_h, multiply])
    activation_2 = Activation('tanh', name='activation_2')(add_2)

    multiply_1 = Multiply(name='multiply_1')([activation_1, snpe_pleaser])  # one end
    one_minus = Dense(512, name='one_minus')(activation_1)
    multiply_2 = Multiply(name='multiply_2')([one_minus, activation_2])  # other end
    add_3 = Add(name='add_3')([multiply_1, multiply_2])

    ### Meta
    meta_dense_1 = Dense(256, name='meta_dense_1')(main_relu)
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
    
    dense_1 = Dense(4, name='dense_1', activation='sigmoid')(meta_relu_4)

    
    # Path
    dense_1_path = Dense(256, name='dense_1_path')(add_3)
    elu_1_path = ReLU(name='elu_1_path')(dense_1_path)
    dense_2_path = Dense(192, name='dense_2_path')(elu_1_path)
    elu_2_path = ReLU(name='elu_2_path')(dense_2_path)
    dense_3_path = Dense(128, name='dense_3_path')(elu_2_path)
    elu_3_path = ReLU(name='elu_3_path')(dense_3_path)
    path = Dense(384, name='path')(elu_3_path)

    # Left lane
    dense_1_left_lane = Dense(256, name='dense_1_left_lane')(add_3)
    elu_1_left_lane = ReLU(name='elu_1_left_lane')(dense_1_left_lane)
    dense_2_left_lane = Dense(192, name='dense_2_left_lane')(elu_1_left_lane)
    elu_2_left_lane = ReLU(name='elu_2_left_lane')(dense_2_left_lane)
    dense_3_left_lane = Dense(128, name='dense_3_left_lane')(elu_2_left_lane)
    elu_3_left_lane = ReLU(name='elu_3_left_lane')(dense_3_left_lane)
    left_lane = Dense(385, name='left_lane')(elu_3_left_lane)

    # Right lane
    dense_1_right_lane = Dense(256, name='dense_1_right_lane')(add_3)
    elu_1_right_lane = ReLU(name='elu_1_right_lane')(dense_1_right_lane)
    dense_2_right_lane = Dense(192, name='dense_2_right_lane')(elu_1_right_lane)
    elu_2_right_lane = ReLU(name='elu_2_right_lane')(dense_2_right_lane)
    dense_3_right_lane = Dense(128, name='dense_3_right_lane')(elu_2_right_lane)
    elu_3_right_lane = ReLU(name='elu_3_right_lane')(dense_3_right_lane)
    right_lane = Dense(385, name='right_lane')(elu_3_right_lane)

    # Lead
    dense_1_lead = Dense(256, name='dense_1_lead')(add_3)
    elu_1_lead = ReLU(name='elu_1_lead')(dense_1_lead)
    dense_2_lead = Dense(192, name='dense_2_lead')(elu_1_lead)
    elu_2_lead = ReLU(name='elu_2_lead')(dense_2_lead)
    dense_3_lead = Dense(128, name='dense_3_lead')(elu_2_lead)
    elu_3_lead = ReLU(name='elu_3_lead')(dense_3_lead)
    lead = Dense(58, name='lead')(elu_3_lead)

    # Output dimension should be (1, 1760)
    outputs = Concatenate(name='outputs')([path, left_lane, right_lane, lead, dense_1, snpe_pleaser2, add_3])
    model = Model(inputs=[vision_input, desire, rnn_state], outputs=outputs, name='openpilot_model')

    plot_model(model, to_file='openpilot_model.png')

    return model

def build_openpilot_keras_model():
    vision_input = Input(shape=(6,80,160), dtype='float32', name='vision')
    rnn_state = Input(shape=(512,), dtype='float32', name='rnn_state')

    # After permutation, the output shape will be (80, 160, 6)
    vision_permute = Permute((2,3,1), input_shape=(6,80,160), name='vision_permute')(vision_input)
    vision_conv2d = Conv2D(16, 5, strides=1, padding="same", name='vision_conv2d')(vision_permute)
    vision_elu = ELU(alpha=1.0, name='vision_elu')(vision_conv2d)
    vision_conv2d_1 = Conv2D(16, 3, strides=1, padding="same", name='vision_conv2d_1')(vision_elu)

    # Resnet block
    vision_max_pooling2d = MaxPooling2D(pool_size=3, strides=2, padding="valid", name='vision_max_pooling2d')(vision_conv2d_1)

    vision_conv2d_2 = Conv2D(16, 3, strides=1, padding="same", name='vision_conv2d_2')(vision_max_pooling2d)
    vision_elu_1 = ELU(alpha=1.0, name='vision_elu_1')(vision_conv2d_2)
    vision_conv2d_3 = Conv2D(16, 3, strides=1, padding="same", name='vision_conv2d_3')(vision_elu_1)
    vision_add = Add(name='vision_add')([vision_max_pooling2d, vision_conv2d_3])

    # Resnet block
    vision_max_pooling2d_1 = MaxPooling2D(pool_size=3, strides=2, padding="valid", name='vision_max_pooling2d_1')(vision_add)
    vision_conv2d_4 = Conv2D(16, 3, strides=1, padding="same", name='vision_conv2d_4')(vision_max_pooling2d_1)
    vision_elu_2 = ELU(alpha=1.0, name='vision_elu_2')(vision_conv2d_4)
    vision_conv2d_5 = Conv2D(16, 3, strides=1, padding="same", name='vision_conv2d_5')(vision_elu_2)
    vision_add_1 = Add(name='vision_add_1')([vision_max_pooling2d_1, vision_conv2d_5])

    # Resnet block
    vision_max_pooling2d_2 = MaxPooling2D(pool_size=3, strides=2, padding="valid", name='vision_max_pooling2d_2')(vision_add_1)
    vision_conv2d_6 = Conv2D(16, 3, strides=1, padding="same", name='vision_conv2d_6')(vision_max_pooling2d_2)
    vision_elu_3 = ELU(alpha=1.0, name='vision_elu_3')(vision_conv2d_6)
    vision_conv2d_7 = Conv2D(16, 3, strides=1, padding="same", name='vision_conv2d_7')(vision_elu_3)
    vision_add_2 = Add(name='vision_add_2')([vision_max_pooling2d_2, vision_conv2d_7])

    vision_max_pooling2d_3 = MaxPooling2D(pool_size=3, strides=2, padding="valid", name='vision_max_pooling2d_3')(vision_add_2)
    flattened = Flatten(name='flattened')(vision_max_pooling2d_3)

    # RNN layer
    dense = Dense(576, name='dense')(flattened)
    elu = ELU(alpha=1.0, name='elu')(dense)
    rnn_kernel = Dense(512, name='rnn_kernel')(elu)
    rnn_recurrent_kernel = Dense(512, name='rnn_recurrent_kernel')(rnn_state)
    rnn_add = Add(name='rnn_add')([rnn_kernel, rnn_recurrent_kernel])
    rnn_out_state = Activation('tanh', name='rnn_out_state')(rnn_add)

    # Path
    dense_1 = Dense(256, name='dense_1')(rnn_out_state)
    elu_1 = ELU(alpha=1.0, name='elu_1')(dense_1)
    dense_2 = Dense(128, name='dense_2')(elu_1)
    elu_2 = ELU(alpha=1.0, name='elu_2')(dense_2)
    path = Dense(51, name='path')(elu_2)

    # Left lane
    dense_4 = Dense(256, name='dense_4')(rnn_out_state)
    elu_3 = ELU(alpha=1.0, name='elu_3')(dense_4)
    dense_5 = Dense(128, name='dense_5')(elu_3)
    elu_4 = ELU(alpha=1.0, name='elu_4')(dense_5)
    left_lane = Dense(53, name='left_lane')(elu_4)

    # Right lane
    dense_7 = Dense(256, name='dense_7')(rnn_out_state)
    elu_5 = ELU(alpha=1.0, name='elu_5')(dense_7)
    dense_8 = Dense(128, name='dense_8')(elu_5)
    elu_6 = ELU(alpha=1.0, name='elu_6')(dense_8)
    right_lane = Dense(53, name='right_lane')(elu_6)

    # Lead
    dense_10 = Dense(256, name='dense_10')(rnn_out_state)
    elu_7 = ELU(alpha=1.0, name='elu_7')(dense_10)
    dense_11 = Dense(128, name='dense_11')(elu_7)
    elu_8 = ELU(alpha=1.0, name='elu_8')(dense_11)
    lead = Dense(4, name='lead')(elu_8)

    # Output dimension should be (1, 673)
    outputs = Concatenate(name='outputs')([path, left_lane, right_lane, lead, rnn_out_state])

    model = Model(inputs=[vision_input, rnn_state], outputs=outputs, name='driving_model')
    #print(model.summary())
    #plot_model(model, to_file='driving_model.png')
    #model.load_weights('../testdata/driving_model.h5')
    return model


def build_modified_openpilot_model():
    openpilot_model = build_openpilot_model()
    openpilot_model.trainable = False
    openpilot_model.load_weights('/lanefollowing/ros2_ws/src/lane_following/model/driving_model.h5')

    # set all previous weights to be untrainable
    for k in openpilot_model.layers:
        k.trainable = False

    # add outputs for steering command, should output of rnn state be used here?
    dense_1_final = Dense(100, name='dense_1_final')(openpilot_model.get_layer("outputs").output)
    elu_1_final = ReLU(name = 'elu_1_final')(dense_1_final)
    dense_2_final = Dense(50, name='dense_2_final')(elu_1_final)
    elu_2_final = ReLU(name = 'elu_2_final')(dense_2_final)
    dense_3_final = Dense(10, name='dense_3_final')(elu_2_final)
    elu_3_final = ReLU(name = 'elu_3_final')(dense_3_final)
    dense_4_final = Dense(1, name = 'output')(elu_3_final)

    rnn_out = openpilot_model.get_layer("add_3").output
    # model = Model(inputs=openpilot_model.inputs, outputs=[dense_4_final, rnn_out], name='driving_model')
    model = Model(inputs=openpilot_model.inputs, outputs=[dense_4_final], name='driving_model')
    
    # steering angle
    # acceleration/speed
    # lateral/longitiudinal
    plot_model(model, to_file='driving_model.png', show_shapes=True)
    
    return model