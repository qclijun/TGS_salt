from models.unets import xception_fpn, resnet152_fpn, inception_resnet_v2_fpn, densenet_fpn, testnet
from keras.applications.resnet50 import ResNet50
from keras.applications.xception import Xception
from keras.applications.inception_resnet_v2 import InceptionResNetV2


_preprocessing_modes = {
    "xception_fpn": "tf",
    "resnet152": "torch",
    "inception_resnet_v2": "tf",
    "densenet169": "torch",
    "testnet": "torch"
}


def make_model(network, black_detect=False):
    input_shape = (None, None, 3)
    if network == "xception_fpn":
        return xception_fpn(input_shape, channels=1, activation="sigmoid", black_detect=black_detect)
    elif network == "resnet152":
        return resnet152_fpn(input_shape, channels=1, activation="sigmoid", black_detect=black_detect)
    elif network == "inception_resnet_v2":
        return inception_resnet_v2_fpn(input_shape, channels=1, activation="sigmoid", black_detect=black_detect)
    elif network == "densenet169":
        return densenet_fpn(input_shape, channels=1, activation="sigmoid", black_detect=black_detect)
    elif network == "testnet":
        return testnet(input_shape, channels=1, activation="sigmoid", black_detect=black_detect)
    else:
        raise NotImplementedError("Network {} not implement.".format(network))


def get_preprocessing_mode(network):
    return _preprocessing_modes[network]
