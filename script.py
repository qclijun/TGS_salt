
import math
import time
from pathlib import Path
from contextlib import contextmanager

import numpy as np
import pandas as pd

import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array
from keras.layers import Conv2D, BatchNormalization, Dropout, Concatenate, Layer
from keras.layers import MaxPool2D, UpSampling2D, Conv2DTranspose, Input
from keras.models import Model, load_model
from keras.losses import binary_crossentropy
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# from tqdm import tqdm
from sklearn.model_selection import train_test_split
# from skimage.transform import resize

import keras.backend as K
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import seaborn as sns
sns.set_style("white")

@contextmanager
def timer(title):
    start = time.time()
    yield
    print('{} - Done in {:.2f} secs'.format(title, time.time() - start))


img_size_ori = 101
img_size_target = 128

data_dir = Path("../input")

train_csv = str(data_dir / "train.csv")
depths_csv = str(data_dir / "depths.csv")
train_imgs_dir = data_dir / "train/images"
train_masks_dir = data_dir / "train/masks"
test_imgs_dir = data_dir / "test/images"

ACTIVATION = "relu"
USE_BN = True
USE_RESIDUAL = False
DROP_RATE = 0.5
USE_UPSAMPLING = True

epochs = 200
batch_size = 32
verbose = 0

model_file = "keras.model"

# def upsample(img):
#     return resize(img, (img_size_target, img_size_target), mode="constant", preserve_range=True)
#
#
# def downsample(img):
#     return resize(img, (img_size_ori, img_size_ori), mode="constant", preserve_range=True)


def load_csv():
    train_df = pd.read_csv(train_csv, index_col="id", usecols=[0])
    depths_df = pd.read_csv(depths_csv, index_col="id")
    train_df['z'] = depths_df.loc[train_df.index, "z"]
    test_df = depths_df[~depths_df.index.isin(train_df.index)]

    return train_df, test_df, depths_df


def load_images(img_paths):
    return np.asarray([np.array(load_img(str(img_path), grayscale=True)) / 255.0 for img_path in img_paths])


def cal_iou(y_true, y_pred):
    hist2d = np.histogram2d(y_true.flatten(), y_pred.flatten(), bins=(2, 2))[0]
    intersection = hist2d[1, 1]
    union = intersection + hist2d[0, 1] + hist2d[1, 0]

    # both y_true and y_pred are black (tn)
    if union == 0:
        iou = -1.0
    else:
        iou = intersection / union
    return iou


def iou_metric_batch(ground_trues, preds):
    n_samples = len(ground_trues)
    iou_arr = np.asarray([cal_iou(ground_trues[i], preds[i]) for i in range(n_samples)], np.float32)

    precisions = []
    for iou_thres in np.arange(0.5, 1.0, 0.05):
        tp = np.sum(iou_arr > iou_thres)
        tn = np.sum(iou_arr == -1.0)
        prec = tp / (n_samples - tn)
        precisions.append(prec)
    return np.mean(precisions)


def split_dataset(train_df, test_size=0.2, random_state=None):
    ids_train, ids_valid = train_test_split(train_df.index.values, test_size=test_size, random_state=random_state,
                                            stratify=train_df['coverage_class'])
    x_train = np.stack(train_df.loc[ids_train, 'image'], axis=0)[..., np.newaxis]
    y_train = np.stack(train_df.loc[ids_train, 'mask'], axis=0)[..., np.newaxis]

    x_valid = np.stack(train_df.loc[ids_valid, "image"], axis=0)[..., np.newaxis]
    y_valid = np.stack(train_df.loc[ids_valid, "mask"], axis=0)[..., np.newaxis]

    return ids_train, ids_valid, x_train, x_valid, y_train, y_valid


class ResizeImageLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        super(ResizeImageLayer, self).__init__(**kwargs)
        self.output_dim = output_dim

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim[0], self.output_dim[1], input_shape[3]

    def call(self, inputs):
        output = tf.image.resize_images(inputs, self.output_dim)
        return output



# input: n_channels, (mxm)
# output: n_channels, (m x m)
def conv_block(inputs, n_channels, dropout=0.0):
    n = Conv2D(n_channels, kernel_size=3, activation=ACTIVATION, padding="same")(inputs)
    n = BatchNormalization()(n) if USE_BN else n
    n = Dropout(dropout)(n) if dropout else n
    n = Conv2D(n_channels, kernel_size=3, activation=ACTIVATION, padding="same")(n)
    n = BatchNormalization()(n) if USE_BN else n
    return n
    # Concatenate()([inputs, n]) if USE_RESIDUAL else n


def up_block(inputs, conv_outputs, conv_channels):
    x = inputs

    for conv_ch, conv_x in zip(conv_channels[::-1], conv_outputs[::-1]):
        if USE_UPSAMPLING:
            x = UpSampling2D()(x)
            x = Conv2D(conv_ch, kernel_size=2, activation=ACTIVATION, padding="same")(x)
        else:
            x = Conv2DTranspose(conv_ch, kernel_size=3, strides=2, activation=ACTIVATION, padding="same")(x)
        x = Concatenate()([conv_x, x])
        x = conv_block(x, conv_ch)
    return x


def UNet():
    inputs = Input(shape=(img_size_ori, img_size_ori, 1))
    x = ResizeImageLayer(output_dim=(img_size_target, img_size_target))(inputs)

    # 128x128x1
    # 64x64x16
    # 32x32x32
    # 16x16x64
    # 8x8x128
    # 4x4x256
    conv_channels = [16, 32, 64, 128, 256]
    conv_outputs = []
    for n_channels in conv_channels:
        x = conv_block(x, n_channels)
        conv_outputs.append(x)
        x = MaxPool2D()(x)
    # middle: 4x4x512
    x = conv_block(x, 512, dropout=DROP_RATE)

    # 128x128x16
    x = up_block(x, conv_outputs, conv_channels)

    # last layer: 128x128x16
    outputs = Conv2D(1, kernel_size=1, activation='sigmoid')(x)
    outputs = ResizeImageLayer(output_dim=(img_size_ori, img_size_ori))(outputs)
    return Model(inputs=inputs, outputs=outputs)


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred = K.cast(y_pred, 'float32')
    y_pred_f = K.cast(K.greater(K.flatten(y_pred), 0.5), 'float32')
    intersection = y_true_f * y_pred_f
    score = 2.0 * K.sum(intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))
    return score


def dice_loss(y_true, y_pred):
    smooth = 1.0
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2.0 * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1 - score


def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)


def build_model():
    model = UNet()
    model.compile(loss=bce_dice_loss, optimizer='adam', metrics=['accuracy'])
    return model


def train(model, x_train, y_train, x_valid, y_valid):
    early_stopping = EarlyStopping(patience=10, verbose=1)
    model_checkpoint = ModelCheckpoint(model_file, save_best_only=True, verbose=verbose)
    reduce_lr = ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=verbose)

    with timer("Training"):
        history = model.fit(x_train, y_train,
                            validation_data=[x_valid, y_valid],
                            epochs=epochs,
                            batch_size=batch_size,
                            callbacks=[early_stopping, model_checkpoint, reduce_lr],
                            verbose=verbose)

    plot_history(history)
    return history


def plot_history(history):
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(15, 5))
    ax_loss.plot(history.epoch, history.history["loss"], label="Train loss")
    ax_loss.plot(history.epoch, history.history["val_loss"], label="Validation loss")
    ax_acc.plot(history.epoch, history.history["acc"], label="Train accuracy")
    ax_acc.plot(history.epoch, history.history["val_acc"], label="Validation accuracy")
    fig.savefig("history.png")


def evaluate(model, x_valid, y_valid):
    pred_valid = model.predict(x_valid).reshape(-1, img_size_ori, img_size_ori)

    prob_thresholds = np.linspace(0, 1, 50)
    precisions = [iou_metric_batch(y_valid, np.int32(pred_valid > threshold)) for threshold in prob_thresholds]
    best_index = np.argmax(precisions)
    precision_best = precisions[best_index]
    threshold_best = prob_thresholds[best_index]

    print('threshold_best = {:.3f}, precison_best = {:.3f}'.format(threshold_best, precision_best))
    return threshold_best, precision_best


# Source https://www.kaggle.com/bguberfain/unet-with-depth
def RLenc(img, order='F', format=True):
    """
    img is binary mask image, shape (r,c)
    order is down-then-right, i.e. Fortran
    format determines if the order needs to be preformatted (according to submission rules) or not

    returns run length as an array or string (if format is True)
    """
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = []  ## list of run lengths
    r = 0  ## the current run length
    pos = 1  ## count starts from 1 per WK
    for c in bytes:
        if (c == 0):
            if r != 0:
                runs.append((pos, r))
                pos += r
                r = 0
            pos += 1
        else:
            r += 1

    # if last run is unsaved (i.e. data ends with 1)
    if r != 0:
        runs.append((pos, r))
        pos += r
        r = 0

    if format:
        z = ''

        for rr in runs:
            z += '{} {} '.format(rr[0], rr[1])
        return z[:-1]
    else:
        return runs


def do_predict(model, test_df, threshold_best):
    with timer("Load test images"):
        test_imgs_files = [str(test_imgs_dir / "{}.png".format(idx)) for idx in test_df.index]
        n_test = len(test_imgs_files)
        x_test = np.empty((n_test, img_size_ori, img_size_ori), np.float32)
        for i, file in enumerate(test_imgs_files):
            x_test[i] = np.array(load_img(file, grayscale=True)) / 255.0
        x_test = x_test[..., np.newaxis]

    with timer("Do predict"):
        preds_test = model.predict(x_test, batch_size=64)
        rle_mask = [RLenc(np.int32(pred > threshold_best)) for pred in preds_test]
        test_df['rle_mask'] = rle_mask

        sub_df = test_df['rle_mask']
    return sub_df



if __name__ == '__main__':
    with timer("Load csv"):
        train_df, test_df, depths_df = load_csv()

    n_train = len(train_df)
    n_test = len(test_df)
    print('num of train samples: {}'.format(n_train))
    print('num of test samples: {}'.format(n_test))

    tr_imgs_path = (train_imgs_dir / "{}.png".format(idx) for idx in train_df.index)
    tr_masks_path = (train_masks_dir / "{}.png".format(idx) for idx in train_df.index)

    with timer("Load train images and masks"):
        tr_images = load_images(tr_imgs_path)
        tr_masks = load_images(tr_masks_path)

    train_df["coverage"] = np.mean(tr_masks.reshape(n_train, -1), axis=1)
    train_df["coverage_class"] = train_df["coverage"].apply(lambda x: math.ceil(x * 10))

    ids_train, ids_valid = train_test_split(range(n_train), test_size=0.2, random_state=43,
                                            stratify=train_df['coverage_class'])

    x_train = tr_images[ids_train][..., np.newaxis]
    x_valid = tr_images[ids_valid][..., np.newaxis]
    y_train = tr_masks[ids_train][..., np.newaxis]
    y_valid = tr_masks[ids_valid][..., np.newaxis]
    with timer("Data augmentation"):
        x_train = np.concatenate([x_train, x_train[:, :, ::-1, :]], axis=0)
        y_train = np.concatenate([y_train, y_train[:, :, ::-1, :]], axis=0)

    with timer("Build model"):
        model = build_model()

    history = train(model, x_train, y_train, x_valid, y_valid)

    threshold_best, precision_best = evaluate(model, x_valid, y_valid)
    sub_df = do_predict(model, test_df, threshold_best)

    sub_df.to_csv("submission.csv")

