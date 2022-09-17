from keras.models import Model, load_model
from keras.layers import *
from keras import backend as K
import tensorflow as tf

# Metrics

def TP(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    true_positives = K.sum(K.round(K.clip(y_true_f * y_pred_f, 0, 1)))
    return true_positives


def FP(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    y_pred_f01 = K.round(K.clip(y_pred_f, 0, 1))
    tp_f01 = K.round(K.clip(y_true_f * y_pred_f, 0, 1))
    false_positives = K.sum(K.round(K.clip(y_pred_f01 - tp_f01, 0, 1)))
    return false_positives


def TN(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    y_pred_f01 = K.round(K.clip(y_pred_f, 0, 1))
    all_one = K.ones_like(y_pred_f01)
    y_pred_f_1 = -1 * (y_pred_f01 - all_one)
    y_true_f_1 = -1 * (y_true_f - all_one)
    true_negatives = K.sum(K.round(K.clip(y_true_f_1 + y_pred_f_1, 0, 1)))
    return true_negatives


def FN(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    # y_pred_f01 = keras.round(keras.clip(y_pred_f, 0, 1))
    tp_f01 = K.round(K.clip(y_true_f * y_pred_f, 0, 1))
    false_negatives = K.sum(K.round(K.clip(y_true_f - tp_f01, 0, 1)))
    return false_negatives


def recall(y_true, y_pred):
    tp = TP(y_true, y_pred)
    fn = FN(y_true, y_pred)
    return tp / (tp + fn)


def precision(y_true, y_pred):
    tp = TP(y_true, y_pred)
    fp = FP(y_true, y_pred)
    return tp / (tp + fp)


def patch_whole_dice(truth, predict):
    dice = []
    count_dice = 0
    for i in range(len(truth)):
        true_positive = truth[i] > 0
        predict_positive = predict[i] > 0
        match = np.equal(true_positive, predict_positive)
        match_count = np.count_nonzero(match)

        P1 = np.count_nonzero(predict[i])
        T1 = np.count_nonzero(truth[i])

        full_back = np.zeros(truth[i].shape)
        non_back = np.invert(np.equal(truth[i], full_back))
        TP = np.logical_and(match, non_back)
        TP_count = np.count_nonzero(TP)
        # print("m:", match_count, " P:", P1, " T:", T1, " TP:", TP_count)

        if (P1 + T1) == 0:
            dice.append(0)
        else:
            dice.append(2 * TP_count / (P1 + T1))
        if P1 != 0 or T1 != 0:
            count_dice += 1
    if count_dice == 0:
        count_dice = 1e6
    return dice  # , count_dice
    # return dice

def patch_whole_dice2(truth, predict):
    y_true_f = np.reshape(truth, (1, -1))
    y_pred_f = np.reshape(predict, (1, -1))
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (np.sum(y_true_f * y_true_f) + np.sum(y_pred_f * y_pred_f) + 1)

def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + smooth)


def EML(y_true, y_pred):
    gamma = 1.1
    alpha = 0.48
    smooth = 1
    y_true = tf.math.abs(K.flatten(y_true))
    y_pred = tf.math.abs(K.flatten(y_pred))
    intersection = K.sum(y_true*y_pred)
    dice_loss = (2*intersection + smooth)/(K.sum(y_true*y_true)+K.sum(y_pred * y_pred)+smooth)
    y_pred = K.clip(y_pred, K.epsilon(), 1.0)
    pt_1 = tf.where(tf.math.greater_equal(y_true, 0.5),y_pred,tf.ones_like(y_pred))
    pt_0 = tf.where(tf.math.less(y_true, 0.5),y_pred,tf.zeros_like(y_pred))
    focal_loss = -K.mean(alpha*K.pow(1-pt_1, gamma)*K.log(pt_1),axis=-1)/ -K.mean(1-alpha*K.pow(pt_0,gamma)*K.log(1-pt_0),axis=-1)
    return K.min((focal_loss,1e-4)) - K.min((K.log(dice_loss),1e-4))

# Model Helpers

def expand(x):
    x = K.expand_dims(x, axis=-1)
    return x


def squeeze(x):
    x = K.squeeze(x, axis=-1)
    return x


def BN_block(filter_num, input):
    x = Conv2D(filter_num, 3, padding='same', kernel_initializer='he_normal')(input)
    x = BatchNormalization()(x)
    x1 = Activation('relu')(x)
    x = Conv2D(filter_num, 3, padding='same', kernel_initializer='he_normal')(x1)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def BN_block3d(filter_num, input):
    x = Conv3D(filter_num, 3, padding='same', kernel_initializer='he_normal')(input)
    x = BatchNormalization()(x)
    x1 = Activation('relu')(x)
    x = Conv3D(filter_num, 3, padding='same', kernel_initializer='he_normal')(x1)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def D_Add(filter_num, input3d, input2d):
    x = Conv3D(1, 1, padding='same', kernel_initializer='he_normal')(input3d)
    x = Lambda(squeeze)(x)
    x = Conv2D(filter_num, 3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = Add()([x, input2d])
    return x


def D_concat(filter_num, input3d, input2d):
    x = Conv3D(1, 1, padding='same', kernel_initializer='he_normal')(input3d)
    x = Lambda(squeeze)(x)
    x = Conv2D(filter_num, 3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = Concatenate()([x, input2d])
    x = Conv2D(filter_num, 1, padding='same', activation='relu', kernel_initializer='he_normal')(x)
    return x


def D_SE_concat(filter_num, input3d, input2d):
    x = Conv3D(1, 1, padding='same', kernel_initializer='he_normal')(input3d)
    x = Lambda(squeeze)(x)
    x = Conv2D(filter_num, 3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = squeeze_excite_block(x)
    input2d = squeeze_excite_block(input2d)
    x = Concatenate()([x, input2d])
    x = Conv2D(filter_num, 1, padding='same', activation='relu', kernel_initializer='he_normal')(x)
    return x


def D_Add_SE(filter_num, input3d, input2d):
    x = Conv3D(1, 1, padding='same', kernel_initializer='he_normal')(input3d)
    x = Lambda(squeeze)(x)
    x = Conv2D(filter_num, 3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = Add()([x, input2d])
    x = squeeze_excite_block(x)
    return x


def D_SE_Add(filter_num, input3d, input2d):
    x = Conv3D(1, 1, padding='same', kernel_initializer='he_normal')(input3d)
    x = Lambda(squeeze)(x)
    x = Conv2D(filter_num, 3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = squeeze_excite_block(x)
    input2d = squeeze_excite_block(input2d)
    x = Add()([x, input2d])

    return x


def D_concat_SE(filter_num, input3d, input2d):
    x = Conv3D(1, 1, padding='same', kernel_initializer='he_normal')(input3d)
    x = Lambda(squeeze)(x)
    x = Conv2D(filter_num, 3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = Concatenate()([x, input2d])
    x = squeeze_excite_block(x)
    x = Conv2D(filter_num, 1, padding='same', activation='relu', kernel_initializer='he_normal')(x)
    return x


def squeeze_excite_block(input, ratio=16):
    ''' Create a squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
        k: width factor
    Returns: a keras tensor
    '''
    init = input
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init.shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x

def conv_bn_block(x, filter):
    x = Conv3D(filter, 3, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x1 = Activation('relu')(x)
    x = Conv3D(filter, 3, padding='same', kernel_initializer='he_normal')(x1)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Concatenate()([x, x1])
    return 

# D-Unet definition

def D_Unet():
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 6))
    input3d = Lambda(expand)(inputs)
    conv3d1 = BN_block3d(32, input3d)

    pool3d1 = MaxPooling3D(pool_size=2)(conv3d1)

    conv3d2 = BN_block3d(64, pool3d1)

    pool3d2 = MaxPooling3D(pool_size=2)(conv3d2)

    conv3d3 = BN_block3d(128, pool3d2)


    conv1 = BN_block(32, inputs)
    #conv1 = D_Add(32, conv3d1, conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = BN_block(64, pool1)
    conv2 = D_SE_Add(64, conv3d2, conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = BN_block(128, pool2)
    conv3 = D_SE_Add(128, conv3d3, conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = BN_block(256, pool3)
    conv4 = Dropout(0.3)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = BN_block(512, pool4)
    conv5 = Dropout(0.3)(conv5)

    up6 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv5))
    merge6 = Concatenate()([conv4, up6])
    conv6 = BN_block(256, merge6)

    up7 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = Concatenate()([conv3, up7])
    conv7 = BN_block(128, merge7)

    up8 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = Concatenate()([conv2, up8])
    conv8 = BN_block(64, merge8)

    up9 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = Concatenate()([conv1, up9])
    conv9 = BN_block(32, merge9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)  # conv10作为输出
    output = Dense(1, activation="linear")(conv10)
    model = Model(inputs=inputs, outputs=output)

    return model

def Unet3d():
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 6))
    input3d = Lambda(expand)(inputs)
    conv1 = BN_block3d(32, input3d)
    pool1 = MaxPooling3D(pool_size=(2, 2, 1))(conv1)

    conv2 = BN_block3d(64, pool1)
    pool2 = MaxPooling3D(pool_size=(2, 2, 1))(conv2)

    conv3 = BN_block3d(128, pool2)
    pool3 = MaxPooling3D(pool_size=(2, 2, 1))(conv3)

    conv4 = BN_block3d(256, pool3)
    drop4 = Dropout(0.3)(conv4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 1))(drop4)

    conv5 = BN_block3d(512, pool4)
    drop5 = Dropout(0.3)(conv5)

    up6 = Conv3D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal', name='6')(
        UpSampling3D(size=(2, 2, 1))(drop5))
    merge6 = Concatenate()([drop4, up6])
    conv6 = BN_block3d(256, merge6)

    up7 = Conv3D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal', name='8')(
        UpSampling3D(size=(2, 2, 1))(conv6))
    merge7 = Concatenate()([conv3, up7])
    conv7 = BN_block3d(128, merge7)

    up8 = Conv3D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal', name='10')(
        UpSampling3D(size=(2, 2, 1))(conv7))
    merge8 = Concatenate()([conv2, up8])
    conv8 = BN_block3d(64, merge8)

    up9 = Conv3D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal', name='12')(
        UpSampling3D(size=(2, 2, 1))(conv8))
    merge9 = Concatenate()([conv1, up9])
    conv9 = BN_block3d(32, merge9)
    conv10 = Conv3D(1, 1, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Lambda(squeeze)(conv10)
    # '''
    # conv11 = Lambda(squeeze)(conv10)
    conv11 = BN_block(32, conv10)
    output = Conv2D(1, 1, activation='sigmoid')(conv11)
    # '''
    model = Model(input=inputs, output=output)

    return model