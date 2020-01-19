import pandas as pd
import numpy as np
from matplotlib.image import imsave
import cv2
import os
import gc


def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax


def crop_resize(img0, size=(128, 128), pad=16):
    height, width = img0.shape
    # crop a box around pixels large than the threshold
    # some images contain line at the sides
    ymin, ymax, xmin, xmax = bbox(img0[5:-5, 5:-5] > 80)
    # cropping may cut too much, so we need to add it back
    xmin = xmin - 13 if (xmin > 13) else 0
    ymin = ymin - 10 if (ymin > 10) else 0
    xmax = xmax + 13 if (xmax < width - 13) else width
    ymax = ymax + 10 if (ymax < height - 10) else height
    img = img0[ymin:ymax, xmin:xmax]
    # remove lo intensity pixels as noise
    img[img < 28] = 0
    lx, ly = xmax - xmin, ymax - ymin
    l = max(lx, ly) + pad
    # make sure that the aspect ratio is kept in rescaling
    img = np.pad(img, [((l - ly) // 2, ), ((l - lx) // 2, )], mode='constant')
    return cv2.resize(img, size)


def save_imgs(train_dir: str, size=(236, 137)):
    # x_tot, x2_tot = [], []
    for fname in os.listdir(train_dir):
        if fname.endswith(".parquet"):
            print(f"Unpacking {fname}...")
            df = pd.read_parquet(os.path.join(train_dir, fname))
            # the input is inverted
            data = 255 - df.iloc[:, 1:].values.reshape(
                -1, size[1], size[0]).astype(np.uint8)
            for idx in range(len(df)):
                name = df.iloc[idx, 0]
                # normalize each image by its max val
                img = (data[idx] * (255.0 / data[idx].max())).astype(np.uint8)
                img = crop_resize(img)

                # x_tot.append((img / 255.0).mean())
                # x2_tot.append(((img / 255.0)**2).mean())
                imsave(os.path.join(train_dir, name + '.png'),
                       img,
                       cmap='gray')
            gc.collect()
            print("Done.")
