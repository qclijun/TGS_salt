from keras.layers import Conv2D, BatchNormalization, Activation, UpSampling2D, Add, Input
from keras.layers import concatenate, GlobalAveragePooling2D, Dense
from keras.models import Model
from keras.utils import get_file
from keras.applications.densenet import DenseNet169

from keras_resnet.models import ResNet152

from .resnet_v2 import InceptionResNetV2Same
from .xception_padding import Xception
from .nn_utils import ResizeImageLayer

resnet_filename = 'ResNet-{}-model.keras.h5'
resnet_resource = 'https://github.com/fizyr/keras-models/releases/download/v0.0.1/{}'.format(resnet_filename)


img_size_ori = 101
img_size_target = 128


def download_resnet_imagenet(v):
    v = int(v.replace('resnet', ''))

    filename = resnet_filename.format(v)
    resource = resnet_resource.format(v)
    if v == 50:
        checksum = '3e9f4e4f77bbe2c9bec13b53ee1c2319'
    elif v == 101:
        checksum = '05dc86924389e5b401a9ea0348a3213c'
    elif v == 152:
        checksum = '6ee11ef2b135592f8031058820bb9e71'

    return get_file(
        filename,
        resource,
        cache_subdir='models',
        md5_hash=checksum
    )


def conv_bn_relu(input, num_channel, kernel_size, stride, name, padding='same', bn_axis=-1, bn_momentum=0.99,
                 bn_scale=True, use_bias=True):
    x = Conv2D(filters=num_channel, kernel_size=(kernel_size, kernel_size),
               strides=stride, padding=padding,
               kernel_initializer="he_normal",
               use_bias=use_bias,
               name=name + "_conv")(input)
    x = BatchNormalization(name=name + '_bn', scale=bn_scale, axis=bn_axis, momentum=bn_momentum, epsilon=1.001e-5, )(x)
    x = Activation('relu', name=name + '_relu')(x)
    return x


def conv_bn(input, num_channel, kernel_size, stride, name, padding='same', bn_axis=-1, bn_momentum=0.99, bn_scale=True,
            use_bias=True):
    x = Conv2D(filters=num_channel, kernel_size=(kernel_size, kernel_size),
               strides=stride, padding=padding,
               kernel_initializer="he_normal",
               use_bias=use_bias,
               name=name + "_conv")(input)
    x = BatchNormalization(name=name + '_bn', scale=bn_scale, axis=bn_axis, momentum=bn_momentum, epsilon=1.001e-5, )(x)
    return x


def conv_relu(input, num_channel, kernel_size, stride, name, padding='same', use_bias=True, activation='relu'):
    x = Conv2D(filters=num_channel, kernel_size=(kernel_size, kernel_size),
               strides=stride, padding=padding,
               kernel_initializer="he_normal",
               use_bias=use_bias,
               name=name + "_conv")(input)
    x = Activation(activation, name=name + '_relu')(x)
    return x


def decoder_block(input, filters, skip, block_name):
    x = UpSampling2D()(input)
    x = conv_bn_relu(x, filters, 3, stride=1, padding='same', name=block_name + '_conv1')
    x = concatenate([x, skip], axis=-1, name=block_name + '_concat')
    x = conv_bn_relu(x, filters, 3, stride=1, padding='same', name=block_name + '_conv2')
    return x


def decoder_block_no_bn(input, filters, skip, block_name, activation='relu'):
    x = UpSampling2D()(input)
    x = conv_relu(x, filters, 3, stride=1, padding='same', name=block_name + '_conv1', activation=activation)
    x = concatenate([x, skip], axis=-1, name=block_name + '_concat')
    x = conv_relu(x, filters, 3, stride=1, padding='same', name=block_name + '_conv2', activation=activation)
    return x



def create_pyramid_features(C1, C2, C3, C4, C5, feature_size=256):
    P5 = Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='P5', kernel_initializer="he_normal")(C5)
    P5_upsampled = UpSampling2D(name='P5_upsampled')(P5)

    P4 = Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C4_reduced',
                kernel_initializer="he_normal")(C4)
    P4 = Add(name='P4_merged')([P5_upsampled, P4])
    P4 = Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P4', kernel_initializer="he_normal")(P4)
    P4_upsampled = UpSampling2D(name='P4_upsampled')(P4)

    P3 = Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C3_reduced',
                kernel_initializer="he_normal")(C3)
    P3 = Add(name='P3_merged')([P4_upsampled, P3])
    P3 = Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P3', kernel_initializer="he_normal")(P3)
    P3_upsampled = UpSampling2D(name='P3_upsampled')(P3)

    P2 = Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C2_reduced',
                kernel_initializer="he_normal")(C2)
    P2 = Add(name='P2_merged')([P3_upsampled, P2])
    P2 = Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P2', kernel_initializer="he_normal")(P2)
    P2_upsampled = UpSampling2D(size=(2, 2), name='P2_upsampled')(P2)

    P1 = Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C1_reduced',
                kernel_initializer="he_normal")(C1)
    P1 = Add(name='P1_merged')([P2_upsampled, P1])
    P1 = Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P1', kernel_initializer="he_normal")(P1)

    return P1, P2, P3, P4, P5


def prediction_fpn_block(x, name, upsample=None):
    x = conv_relu(x, 128, 3, stride=1, name="predcition_" + name + "_1")
    x = conv_relu(x, 128, 3, stride=1, name="prediction_" + name + "_2")
    if upsample:
        x = UpSampling2D(upsample)(x)
    return x


def resnet152_fpn(input_shape, channels=1, activation="softmax", black_detect=False):
    img_input = Input(input_shape)
    resize_1 = ResizeImageLayer((img_size_target, img_size_target))(img_input)
    resnet_base = ResNet152(resize_1, include_top=True)
    resnet_base.load_weights(download_resnet_imagenet("resnet152"))
    conv1 = resnet_base.get_layer("conv1_relu").output
    conv2 = resnet_base.get_layer("res2c_relu").output
    conv3 = resnet_base.get_layer("res3b7_relu").output
    conv4 = resnet_base.get_layer("res4b35_relu").output
    conv5 = resnet_base.get_layer("res5c_relu").output
    P1, P2, P3, P4, P5 = create_pyramid_features(conv1, conv2, conv3, conv4, conv5)
    x = concatenate(
        [
            prediction_fpn_block(P5, "P5", (8, 8)),
            prediction_fpn_block(P4, "P4", (4, 4)),
            prediction_fpn_block(P3, "P3", (2, 2)),
            prediction_fpn_block(P2, "P2"),
        ]
    )
    x = conv_bn_relu(x, 256, 3, (1, 1), name="aggregation")
    aggr_feat = x
    x = decoder_block_no_bn(x, 128, conv1, 'up4')
    x = UpSampling2D()(x)
    x = conv_relu(x, 64, 3, (1, 1), name="up5_conv1")
    x = conv_relu(x, 64, 3, (1, 1), name="up5_conv2")
    x = Conv2D(channels, (1, 1), name="mask", kernel_initializer="he_normal")(x)
    x = Activation(activation)(x)

    main_out = ResizeImageLayer((img_size_ori, img_size_ori), name="main_out")(x)
    if black_detect:
        black_global = GlobalAveragePooling2D(name="black_global")(aggr_feat)
        black_out = Dense(1, activation="sigmoid", name="block_out")(black_global)
        model = Model(inputs=img_input, outputs=[main_out, black_out])
    else:
        model = Model(inputs=img_input, outputs=main_out)
    return model


def xception_fpn(input_shape, channels=1, activation="sigmoid", black_detect=False):
    img_input = Input(input_shape)
    resize_1 = ResizeImageLayer((img_size_target, img_size_target))(img_input)
    xception = Xception(input_tensor=resize_1, include_top=False)
    conv1 = xception.get_layer("block1_conv2_act").output
    conv2 = xception.get_layer("block3_sepconv2_bn").output
    conv3 = xception.get_layer("block4_sepconv2_bn").output
    conv3 = Activation("relu")(conv3)
    conv4 = xception.get_layer("block13_sepconv2_bn").output
    conv4 = Activation("relu")(conv4)
    conv5 = xception.get_layer("block14_sepconv2_act").output

    P1, P2, P3, P4, P5 = create_pyramid_features(conv1, conv2, conv3, conv4, conv5)
    x = concatenate(
        [
            prediction_fpn_block(P5, "P5", (8, 8)),
            prediction_fpn_block(P4, "P4", (4, 4)),
            prediction_fpn_block(P3, "P3", (2, 2)),
            prediction_fpn_block(P2, "P2"),
        ]
    )
    x = conv_bn_relu(x, 256, 3, (1, 1), name="aggregation")
    aggr_feat = x
    x = decoder_block_no_bn(x, 128, conv1, 'up4')
    x = UpSampling2D()(x)
    x = conv_relu(x, 64, 3, (1, 1), name="up5_conv1")
    x = conv_relu(x, 64, 3, (1, 1), name="up5_conv2")
    if activation == 'softmax':
        name = 'mask_softmax'
        x = Conv2D(channels, (1, 1), activation=activation, name=name)(x)
    else:
        x = Conv2D(channels, (1, 1), activation=activation, name="mask")(x)

    main_out = ResizeImageLayer((img_size_ori, img_size_ori), name="main")(x)
    if black_detect:
        black_global = GlobalAveragePooling2D(name="black_global")(aggr_feat)
        black_out = Dense(1, activation="sigmoid", name="block")(black_global)
        model = Model(inputs=img_input, outputs=[main_out, black_out])
    else:
        model = Model(inputs=img_input, outputs=main_out)
    return model


def inception_resnet_v2_fpn(input_shape, channels=1, activation="sigmoid", black_detect=False):
    img_input = Input(input_shape)
    resize_1 = ResizeImageLayer((img_size_target, img_size_target))(img_input)
    inceresv2 = InceptionResNetV2Same(input_tensor=resize_1, include_top=False)
    conv1, conv2, conv3, conv4, conv5 = inceresv2.output

    P1, P2, P3, P4, P5 = create_pyramid_features(conv1, conv2, conv3, conv4, conv5)
    x = concatenate(
        [
            prediction_fpn_block(P5, "P5", (8, 8)),
            prediction_fpn_block(P4, "P4", (4, 4)),
            prediction_fpn_block(P3, "P3", (2, 2)),
            prediction_fpn_block(P2, "P2"),
        ]
    )
    x = conv_bn_relu(x, 256, 3, (1, 1), name="aggregation")
    aggr_feat = x
    x = decoder_block_no_bn(x, 128, conv1, 'up4')
    x = UpSampling2D()(x)
    x = conv_relu(x, 64, 3, (1, 1), name="up5_conv1")
    x = conv_relu(x, 64, 3, (1, 1), name="up5_conv2")
    if activation == 'softmax':
        name = 'mask_softmax'
        x = Conv2D(channels, (1, 1), activation=activation, name=name)(x)
    else:
        x = Conv2D(channels, (1, 1), activation=activation, name="mask")(x)
    main_out = ResizeImageLayer((img_size_ori, img_size_ori), name="main")(x)
    if black_detect:
        black_global = GlobalAveragePooling2D(name="black_global")(aggr_feat)
        black_out = Dense(1, activation="sigmoid", name="block")(black_global)
        model = Model(inputs=img_input, outputs=[main_out, black_out])
    else:
        model = Model(inputs=img_input, outputs=main_out)
    return model


def densenet_fpn(input_shape, channels=1, activation="sigmoid", black_detect=False):
    img_input = Input(input_shape)
    resize_1 = ResizeImageLayer((img_size_target, img_size_target))(img_input)
    densenet = DenseNet169(input_tensor=resize_1, include_top=False)
    conv1 = densenet.get_layer("conv1/relu").output
    conv2 = densenet.get_layer("pool2_relu").output
    conv3 = densenet.get_layer("pool3_relu").output
    conv4 = densenet.get_layer("pool4_relu").output
    conv5 = densenet.get_layer("bn").output
    conv5 = Activation("relu", name="conv5_relu")(conv5)

    P1, P2, P3, P4, P5 = create_pyramid_features(conv1, conv2, conv3, conv4, conv5)
    x = concatenate(
        [
            prediction_fpn_block(P5, "P5", (8, 8)),
            prediction_fpn_block(P4, "P4", (4, 4)),
            prediction_fpn_block(P3, "P3", (2, 2)),
            prediction_fpn_block(P2, "P2"),
        ]
    )
    x = conv_bn_relu(x, 256, 3, (1, 1), name="aggregation")
    aggr_feat = x
    x = decoder_block_no_bn(x, 128, conv1, 'up4')
    x = UpSampling2D()(x)
    x = conv_relu(x, 64, 3, (1, 1), name="up5_conv1")
    x = conv_relu(x, 64, 3, (1, 1), name="up5_conv2")
    if activation == 'softmax':
        name = 'mask_softmax'
        x = Conv2D(channels, (1, 1), activation=activation, name=name)(x)
    else:
        x = Conv2D(channels, (1, 1), activation=activation, name="mask")(x)

    main_out = ResizeImageLayer((img_size_ori, img_size_ori), name="main")(x)
    if black_detect:
        black_global = GlobalAveragePooling2D(name="black_global")(aggr_feat)
        black_out = Dense(1, activation="sigmoid", name="block")(black_global)
        model = Model(inputs=img_input, outputs=[main_out, black_out])
    else:
        model = Model(inputs=img_input, outputs=main_out)
    return model


def testnet(input_shape, channels=1, activation="sigmoid", black_detect=False):
    img_input = Input(input_shape)
    resize_1 = ResizeImageLayer((img_size_target, img_size_target))(img_input)
    x = conv_relu(resize_1, channels, 3, 1, "test", activation=activation)
    main_out = ResizeImageLayer((img_size_ori, img_size_ori), name="main")(x)
    if black_detect:
        black_global = GlobalAveragePooling2D(name="block_global")(x)
        black_out = Dense(1, activation="sigmoid", name="black")(black_global)
        model = Model(inputs=img_input, outputs=[main_out, black_out])
    else:

        model = Model(inputs=img_input, outputs=main_out)
    return model


if __name__ == '__main__':
    # resnet152_fpn((256, 256, 3)).summary()
    xception_fpn((101, 101, 3)).summary()

    # inception_resnet_v2_fpn((256, 256, 3)).summary()