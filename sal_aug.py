import imgaug.augmenters as iaa


def affine_seq_aug_fn(seed):
    affine_seq = iaa.Sequential([
        # General
        iaa.SomeOf((1, 2),
                   [iaa.Fliplr(0.5),
                    iaa.Affine(rotate=(-10, 10),
                               translate_percent={"x": (-0.25, 0.25)}, mode='symmetric'),
                    ]),
        # Deformations
        iaa.Sometimes(0.3, iaa.PiecewiseAffine(scale=(0.04, 0.08))),
        iaa.Sometimes(0.3, iaa.PerspectiveTransform(scale=(0.05, 0.1))),
    ], random_order=True)
    affine_seq.reseed(seed)
    return affine_seq


def intensity_seq_aug_fn(seed):
    intensity_seq = iaa.Sequential([
        iaa.Invert(0.3),
        iaa.Sometimes(0.3, iaa.ContrastNormalization((0.5, 1.5))),
        iaa.OneOf([
            iaa.Noop(),
            iaa.Sequential([
                iaa.OneOf([
                    iaa.Add((-10, 10)),
                    iaa.AddElementwise((-10, 10)),
                    iaa.Multiply((0.95, 1.05)),
                    iaa.MultiplyElementwise((0.95, 1.05)),
                ]),
            ]),
            iaa.OneOf([
                iaa.GaussianBlur(sigma=(0.0, 1.0)),
                iaa.AverageBlur(k=(2, 5)),
                iaa.MedianBlur(k=(3, 5))
            ])
        ])
    ], random_order=False)
    intensity_seq.reseed(seed)
    return intensity_seq


def no_operation_aug_fn(_):
    return iaa.Noop()