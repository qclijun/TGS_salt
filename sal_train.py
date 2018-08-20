import time
import argparse
import gc
import json
from collections import namedtuple
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager

import numpy as np
import pandas as pd
from keras.losses import binary_crossentropy
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.optimizers import Adam
from keras.utils.training_utils import multi_gpu_model
from keras.models import Model
from keras.preprocessing.image import array_to_img
import keras.backend as K
import matplotlib.pyplot as plt

from datasets import SalDataset
from model_factory import make_model, get_preprocessing_mode

plt.style.use('seaborn-white')


from utils import timer


parser = argparse.ArgumentParser()
parser.add_argument("experiment", help="Experiment name")
parser.add_argument("--network", default="xception_fpn")
parser.add_argument("--bs", default=32, type=int)
parser.add_argument("--bs_inference", default=64, type=int)
parser.add_argument("--epochs", default=100, type=int)
parser.add_argument("--workers", default=8, type=int)
parser.add_argument("--lr", default=1e-4, type=float)
parser.add_argument("--decay", default=0.0, type=float)
parser.add_argument("--ds_ratio", default=1.0, type=float)
parser.add_argument("--fold", default="0,1,2,3,4")
parser.add_argument("--weights", default="")
parser.add_argument("--dice_weight", default=1.0, type=float)
parser.add_argument("--do_eval", action="store_true")
parser.add_argument("--multi_gpu", action="store_true")
parser.add_argument("--black_detect", action="store_true")
parser.add_argument("--black_loss_weight", default=1.0, type=float)
ARGS = parser.parse_args()

data_dir = Path("data")
out_dir = Path("output")

gpus = [x.name for x in K.device_lib.list_local_devices() if "gpu" in x.name.lower()]


def calc_iou(gt, pred):
    intersection = np.sum(gt * pred)
    union = np.sum(gt + pred > 0)

    # gt, pred are all zeros
    if union == 0:
        return 1.0
    return intersection / union


def calc_ious(gts, preds):
    return np.array([calc_iou(gt, pred) for gt, pred in zip(gts, preds)])


THRESHOLD = np.arange(0.5, 1.0, 0.05)


def calc_metric(gts, preds):
    ious = calc_ious(gts, preds)[..., np.newaxis]
    scores = np.mean(ious > THRESHOLD, axis=1)
    return scores


def hard_dice_coef(y_true, y_pred, smooth=1e-3):
    y_true_f = K.flatten(K.round(y_true[..., 0]))
    y_pred_f = K.flatten(K.round(y_pred[..., 0]))
    intersection = K.sum(y_true_f * y_pred_f)
    return 100. * (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    smooth = 1e-3
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2.0 * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1 - score


def bce_dice_loss(y_true, y_pred, dice_weight=1.0):
    dice_w = dice_weight / (1 + dice_weight)
    bce_w = 1 / (1 + dice_weight)

    return bce_w * binary_crossentropy(y_true, y_pred) + dice_w * dice_loss(y_true, y_pred)


class ModelCheckpointMGPU(ModelCheckpoint):
    def __init__(self, original_model, filepath, monitor='val_loss', verbose=0, save_best_only=False,
                 save_weights_only=False, mode='auto', period=1):
        self.original_model = original_model
        super().__init__(filepath, monitor, verbose, save_best_only, save_weights_only, mode, period)

    def on_epoch_end(self, epoch, logs=None):
        self.model = self.original_model
        super().on_epoch_end(epoch, logs)


def train():
    network = ARGS.network
    model_dir = out_dir / ARGS.experiment
    model_dir.mkdir(exist_ok=True, parents=True)
    print(str(ARGS), file=open(model_dir / "config.txt", "w"))

    dataset = SalDataset(data_dir=data_dir, preprocessing_mode=get_preprocessing_mode(network),
                         sz_ratio=ARGS.ds_ratio, black_detect=ARGS.black_detect)

    folds = [int(x) for x in ARGS.fold.split(",")]

    for fold_id in folds:
        print()
        print("-" * 50)
        print("Fold ", fold_id)

        with timer("Build model"):
            if ARGS.multi_gpu:
                with K.tf.device("/cpu:0"):
                    model = make_model(network, black_detect=ARGS.black_detect)
            else:
                model = make_model(network, black_detect=ARGS.black_detect)

        if ARGS.weights:
            model_path = str(model_dir / ARGS.weights.format(fold_id))
            print("load weights from ", model_path)
            with timer("Load weights"):
                model.load_weights(model_path, by_name=True)
        else:
            print('No weights passed, training from scratch')
        optimizer = Adam(lr=ARGS.lr, decay=ARGS.decay)
        loss_fn = {"main": lambda y_true, y_pred: bce_dice_loss(y_true, y_pred, dice_weight=ARGS.dice_weight)}
        loss_weights = {"main": 1.0}
        if ARGS.black_detect:
            loss_fn["black"] = binary_crossentropy
            loss_weights["black"] = ARGS.black_loss_weight

        if ARGS.multi_gpu:
            model = multi_gpu_model(model, len(gpus))
        model.compile(optimizer=optimizer, loss=loss_fn, metrics=[binary_crossentropy, hard_dice_coef],
                      loss_weights=loss_weights)

        train_gen = dataset.train_gen(fold_id=fold_id, batch_size=ARGS.bs)
        val_gen = dataset.val_gen(fold_id=fold_id, batch_size=ARGS.bs_inference)

        early_stopping = EarlyStopping(patience=10, verbose=1)
        reduce_lr = ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-6, verbose=1)

        black_str = "_black" if ARGS.black_detect else ""
        model_path = model_dir / f"weights_{network}{black_str}_fold{fold_id}.h5"
        model_checkpoint = ModelCheckpointMGPU(model,
                                               filepath=str(model_path),
                                               mode="min",
                                               save_best_only=True,
                                               verbose=1,
                                               save_weights_only=True
                                               )

        # tb_log_file = model_dir / "logs/{}_{}".format(ARGS.network, fold_id)
        # tb_callback = TensorBoard(log_dir=str(tb_log_file),
        #                           batch_size=ARGS.bs)

        print("Start Training...")
        with timer("Training"):
            history = model.fit_generator(generator=train_gen,
                                          steps_per_epoch=len(train_gen),
                                          epochs=ARGS.epochs,
                                          verbose=1,
                                          callbacks=[early_stopping, model_checkpoint, reduce_lr],
                                          validation_data=val_gen,
                                          validation_steps=len(val_gen),
                                          workers=ARGS.workers,
                                          shuffle=False
                                          )
            history_df = pd.DataFrame.from_dict(history.history)
            history_df['epoch'] = history.epoch
            history_df.set_index("epoch")
            history_df.to_csv(str(model_dir / "history_fold{}.csv".format(fold_id)))
            plot_history(history, str(model_dir / "loss_fold{}.png".format(fold_id)))

            if ARGS.do_eval:
                with timer("Evaluation"):
                    output_img_dir = model_dir / "masks_pred"
                    output_img_dir.mkdir(exist_ok=True)
                    eval_df = evaluate(model, dataset, fold_id, ARGS.bs_inference, output_imgs_dir=output_img_dir)
                    eval_df.to_csv(model_dir / "eval_fold{}.csv".format(fold_id), index=None)
                    print("Evaluate Fold", fold_id)
                    mean_iou = eval_df['iou'].mean()
                    mean_precision = eval_df['precision'].mean()
                    s = "mean_iou ={:.3f}, mean_precision={:.3f}".format(mean_iou, mean_precision)
                    print(s)
                    print(s, file=open("score_fold{}.txt".format(fold_id), "w"))

        with timer("Clear session and gc"):
            del model
            K.clear_session()
            gc.collect()


def plot_history(history, outfile):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    ax = axs[0]
    ax.plot(history.epoch, history.history["loss"], label="Train loss")
    ax.plot(history.epoch, history.history["val_loss"], label="Validation loss")
    ax.set_title("Loss")
    ax.legend(loc="upper right")

    ax = axs[1]
    ax.plot(history.epoch, history.history["binary_crossentropy"], label="BCE")
    ax.plot(history.epoch, history.history["val_binary_crossentropy"], label="Validation BCE")
    ax.set_title("BCE")
    ax.legend(loc="upper right")

    ax = axs[2]
    ax.plot(history.epoch, history.history["hard_dice_coef"], label="Dice Coef")
    ax.plot(history.epoch, history.history["val_hard_dice_coef"], label="Validation Dice Coef")
    ax.set_title("dice_coef")
    ax.legend(loc="upper right")
    fig.savefig(outfile)


def evaluate(model: Model, ds: SalDataset, fold_id, batch_size=64, output_imgs_dir=None):
    val_gen = ds.val_gen(fold_id, batch_size=batch_size)
    n_batchs = len(val_gen)
    ious = []
    val_ids = ds.get_val_ids(fold_id)
    img_names = ds.get_img_names(val_ids)

    for batch_id in range(n_batchs):
        x, y_true = val_gen[batch_id]
        y_pred = model.predict_on_batch(x)
        y_pred = (y_pred >= 0.5)

        ious.append(calc_ious(y_true, y_pred))

        if output_imgs_dir is not None:
            for fname, y_p in zip(img_names, y_pred):
                y_out = y_p.astype(np.uint8) * 255
                img = array_to_img(y_out, scale=False)
                img.save(str(output_imgs_dir / (fname + ".png")))

    ious = np.concatenate(ious, axis=0)
    precisions = np.mean(ious[..., np.newaxis] > THRESHOLD, axis=1)
    df = pd.DataFrame({"id": img_names, "iou": ious, "precision": precisions})
    return df


if __name__ == '__main__':
    train()
