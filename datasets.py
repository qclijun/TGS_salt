
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from keras.preprocessing.image import Iterator, load_img
from keras.applications.imagenet_utils import preprocess_input
import imgaug.augmenters as iaa
from tqdm import tqdm


from sal_aug import no_operation_aug_fn, affine_seq_aug_fn, intensity_seq_aug_fn


class SalIterator(Iterator):
    img_aug_seed = 1234
    img_with_mask_aug_seed = 4321

    def __init__(self, images, masks, preprocessing_mode,
                 batch_size=32,
                 img_aug_fn=None,
                 img_with_mask_aug_fn=None,
                 shuffle=False,
                 seed=None,
                 is_black=None):
        self.images = images
        self.masks = masks
        self.preprocessing_mode = preprocessing_mode
        self.img_aug_fn = img_aug_fn or no_operation_aug_fn
        self.img_with_mask_aug_fn = img_with_mask_aug_fn or no_operation_aug_fn
        self.x_aug = iaa.Sequential([self.img_with_mask_aug_fn(SalIterator.img_with_mask_aug_seed),
                                     self.img_aug_fn(SalIterator.img_aug_seed)])
        self.y_aug = self.img_with_mask_aug_fn(SalIterator.img_with_mask_aug_seed)
        self.is_black = is_black

        super(SalIterator, self).__init__(len(images), batch_size, shuffle, seed)

    def _get_batches_of_transformed_samples(self, index_array):
        x_batch = self.images[index_array]
        x_batch = self.x_aug.augment_images(x_batch)
        if self.preprocessing_mode:
            x_batch = preprocess_input(x_batch.astype(np.float32), mode=self.preprocessing_mode)
        y_batch = None
        if self.masks is not None:
            y_batch = self.masks[index_array]
            y_batch = self.y_aug.augment_images(y_batch)
            y_batch = y_batch.astype(np.float32) / 255

        if self.is_black is not None:
            y = [y_batch, self.is_black[index_array]]
        else:
            y = y_batch
        return x_batch, y


class SalDataset:
    def __init__(self, data_dir: Path, preprocessing_mode, kfolds=5, sz_ratio=1.0, black_detect=False):
        self.data_dir = data_dir
        self.imgs_dir = data_dir / "train/images"
        self.masks_dir = data_dir / "train/masks"

        self.train_df = pd.read_csv(str(data_dir / "train_df.csv"), index_col='id')
        self.train_df['is_black'] = self.train_df['coverage_class'] == 0

        n = len(self.train_df)
        print("train samples:", n)
        if sz_ratio < 1.0:
            print("Use {}% train set for testing.".format(sz_ratio * 100))
            n = int(sz_ratio * n)
            self.train_df = self.train_df[:n]

        self.preprocessing_mode = preprocessing_mode
        self.kfolds = 5
        kf = StratifiedKFold(n_splits=kfolds)
        self.ids_trn_val = list(kf.split(np.arange(n), self.train_df['coverage_class']))

        self.bs_train = 32
        self.bs_inference = 64
        self.black_detect = black_detect
        self._load_images_and_masks()

    def _load_images_and_masks(self):
        print("load train images...")
        img_fnames = [str(self.imgs_dir / (idx + ".png")) for idx in self.train_df.index]

        # shape: [4000, 101, 101, 3], dtype: uint8
        self.images = SalDataset.load_images(img_fnames, grayscale=False)

        print("load train masks...")
        mask_fnames = [str(self.masks_dir / (idx + ".png")) for idx in self.train_df.index]

        # shape: [4000, 101, 101, 1], dtype: uint8
        self.masks = SalDataset.load_images(mask_fnames, grayscale=True)[..., np.newaxis]

    def train_gen(self, fold_id=0, batch_size=32, seed=None):
        train_ids = self.ids_trn_val[fold_id][0]
        return SalIterator(self.images[train_ids], self.masks[train_ids],
                           preprocessing_mode=self.preprocessing_mode,
                           batch_size=batch_size,
                           shuffle=True,
                           seed=seed,
                           img_aug_fn=intensity_seq_aug_fn,
                           img_with_mask_aug_fn=affine_seq_aug_fn,
                           is_black=self.train_df['is_black'].values[train_ids] if self.black_detect else None)

    def val_gen(self, fold_id=0, batch_size=32):
        val_ids = self.ids_trn_val[fold_id][1]
        return SalIterator(self.images[val_ids], self.masks[val_ids],
                           preprocessing_mode=self.preprocessing_mode,
                           batch_size=batch_size,
                           shuffle=False,
                           img_aug_fn=None,
                           img_with_mask_aug_fn=None,
                           is_black=self.train_df['is_black'].values[val_ids] if self.black_detect else None)

    def get_val_ids(self, fold_id):
        return self.ids_trn_val[fold_id][1]

    def get_img_names(self, ids):
        return self.train_df.index[ids]

    @staticmethod
    def load_images(fnames, grayscale):
        imgs = [np.array(load_img(fname, grayscale=grayscale)) for fname in tqdm(fnames)]
        imgs = np.stack(imgs, axis=0)
        return imgs

    @staticmethod
    def calc_border(masks):
        pass


if __name__ == '__main__':
    dataset = SalDataset(Path("./data"), preprocessing_mode="tf", kfolds=5, sz_ratio=0.1)
    val_gen = dataset.val_gen(batch_size=16)

    for i in range(2):
        x_val, y_val = val_gen[i]
        print(x_val.shape, y_val.shape)
        print(x_val.dtype, y_val.dtype)
        print(x_val)
        print(y_val)
