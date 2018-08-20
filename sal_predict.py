import argparse
import os.path
from pathlib import Path

import numpy as np
import pandas as pd
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing.image import load_img
from keras.utils import Sequence
from tqdm import tqdm

from model_factory import make_model, get_preprocessing_mode
from utils import timer
from datasets import SalDataset

parser = argparse.ArgumentParser()
parser.add_argument("--network")
parser.add_argument("--model_dir")
parser.add_argument("--models")
# parser.add_argument("--workers", default=4, type=int)
# parser.add_argument("--bs", default=64, type=int)
parser.add_argument("--threshold", default=0.5, type=float)
parser.add_argument("submission_file")
ARGS = parser.parse_args()


test_imgs_dir = Path("data/test/images")
submission_file = ARGS.submission_file
parent_dir = Path(submission_file).parent
if not parent_dir.exists():
    parent_dir.mkdir(parent=True, exist_ok=True)

img_size_ori = 101


def preprocess_inputs(x):
    return preprocess_input(x, mode=get_preprocessing_mode(ARGS.network))


def rle(img):
    x = np.pad(img.reshape(-1, order="F"), [2, 1], mode="constant")
    diff_x = np.diff(x)
    non_zero = np.nonzero(diff_x)[0]
    if len(non_zero) == 0:
        return ""
    return ' '.join(str(non_zero[i]) if i % 2 == 0 else str(non_zero[i] - non_zero[i - 1])
                    for i in range(len(non_zero)))


class FileSequence(Sequence):
    def __init__(self, files, batch_size):
        self.files = files
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.files) / self.batch_size))

    def __getitem__(self, idx):
        files_batch = self.files[idx * self.batch_size:(idx + 1) * self.batch_size]
        n_files = len(files_batch)
        imgs = np.empty((n_files, img_size_ori, img_size_ori), np.float32)
        for i, file in enumerate(files_batch):
            imgs[i] = np.array(load_img(str(file), grayscale=False)) / 255.0

        return imgs[..., np.newaxis]


def main():
    with timer("Load models"):
        models = []
        weights = [os.path.join(ARGS.model_dir, w) for w in ARGS.models.split(",")]
        for weight in weights:
            model = make_model(ARGS.network)
            print("Build model {} from weights {}".format(ARGS.network, weight))
            model.load_weights(weight, by_name=True)
            models.append(model)

    test_imgs_files = list(test_imgs_dir.iterdir())
    # dataset = SalDataset(data_dir=Path("data"), preprocessing_mode=get_preprocessing_mode(ARGS.network),
    #                      sz_ratio=1.0, black_detect=False)

    # ids = dataset.get_val_ids(1)
    # imgs = dataset.images[ids]

    rle_masks = []

    # preds = []
    # for img in tqdm(imgs):
    #     x = preprocess_input(img)
    #     x = x[np.newaxis, ...]
    #
    #     pred = np.zeros(shape=[img_size_ori, img_size_ori], dtype=np.float32)
    #
    #     for model in models:
    #         p = model.predict_on_batch(x)
    #         # print("p.shape:", p.shape)
    #         pred += p.squeeze()
    #         preds.append(p.squeeze())
    #
    #     pred /= len(models)
    #     rle_mask = rle(pred > ARGS.threshold)
    #     rle_masks.append(rle_mask)

    for img_file in tqdm(test_imgs_files):
        img = np.array(load_img(str(img_file))).astype(np.float32) / 255.0
        # print(img[0, :3])
        x = preprocess_input(img)
        x = x[np.newaxis, ...]
        # print('x.shape: ', x.shape)
        pred = models[0].predict(x, batch_size=1)
        # pred = np.zeros(shape=[img_size_ori, img_size_ori], dtype=np.float32)
        #
        # for model in models:
        #     p = model.predict_on_batch(x)
        #     pred += p.squeeze()
        #
        # pred /= len(models)
        rle_mask = rle(pred >= ARGS.threshold)
        rle_masks.append(rle_mask)
    #     i += 1

    ids = [f.stem for f in test_imgs_files]
    sub_df = pd.DataFrame(data=ids, columns=['id'])
    sub_df['rle_mask'] = rle_masks
    sub_df.to_csv(ARGS.submission_file, index=None)

    # print(preds[0])
    # print(preds[1])

if __name__ == '__main__':
    with timer("Total predict"):
        main()
