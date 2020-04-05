from torch.utils.data import DataLoader, Dataset
from models.fastai import Rnet_1ch
import torch
import numpy as np
import os
import cv2
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser


#check https://www.kaggle.com/iafoss/image-preprocessing-128x128
HEIGHT = 137
WIDTH = 236
def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax

def crop_resize(img0, size=128, pad=16):

    #crop a box around pixels large than the threshold
    #some images contain line at the sides
    ymin,ymax,xmin,xmax = bbox(img0[5:-5,5:-5] > 80)
    #cropping may cut too much, so we need to add it back
    xmin = xmin - 13 if (xmin > 13) else 0
    ymin = ymin - 10 if (ymin > 10) else 0
    xmax = xmax + 13 if (xmax < WIDTH - 13) else WIDTH
    ymax = ymax + 10 if (ymax < HEIGHT - 10) else HEIGHT
    img = img0[ymin:ymax,xmin:xmax]
    #remove lo intensity pixels as noise
    img[img < 28] = 0
    lx, ly = xmax-xmin,ymax-ymin
    l = max(lx,ly) + pad
    #make sure that the aspect ratio is kept in rescaling
    img = np.pad(img, [((l-ly)//2,), ((l-lx)//2,)], mode='constant')
    return cv2.resize(img,(size,size))

class GraphemeDataset(Dataset):
    def __init__(self, fname):
        self.df = pd.read_parquet(fname)
        self.data = 255 - self.df.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH).astype(np.uint8)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        name = self.df.iloc[idx,0]
        #normalize each image by its max val
        img = (self.data[idx]*(255.0/self.data[idx].max())).astype(np.uint8)
        img = crop_resize(img)
        img = (img.astype(np.float32)/255.0 - stats[0])/stats[1]
        return img, name



if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('--batch_size', type=int, default=128)
    argparser.add_argument('--models_path', type=str, required=True)
    argparser.add_argument('--num_workers', type=int, default=2)
    argparser.add_argument('--parquet_dir', type=str, required=True)
    args = argparser.parse_args()

    stats = (0.0692, 0.2051)

    parquets = [f'{args.parquet_dir}/test_image_data_{fold}.parquet' for fold in [0, 1, 2, 3]]

    LABELS = '../input/bengaliai-cv19/train.csv'

    df = pd.read_csv(LABELS)
    nunique = list(df.nunique())[1:-1]

    row_id, target = [], []
    for fold in [1,2,3]:
        model = Rnet_1ch(pre=False).cuda()
        model.load_state_dict(torch.load(os.path.join(args.models_path, f'model{fold}.pth'), map_location=torch.device('cpu')));
        model.eval();
        parquet = f'{args.parquet_dir}/test_image_data_{fold}.parquet'
        ds = GraphemeDataset(parquet)
        dl = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
        with torch.no_grad():
            for x, y in tqdm(dl):
                x = x.unsqueeze(1).cuda()
                p1, p2, p3 = model(x)
                p1 = p1.argmax(-1).view(-1).cpu()
                p2 = p2.argmax(-1).view(-1).cpu()
                p3 = p3.argmax(-1).view(-1).cpu()
                for idx, name in enumerate(y):
                    row_id += [f'{name}_grapheme_root', f'{name}_vowel_diacritic',
                               f'{name}_consonant_diacritic']
                    target += [p1[idx].item(), p2[idx].item(), p3[idx].item()]

    sub_df = pd.DataFrame({'row_id': row_id, 'target': target})
    sub_df.to_csv('submission.csv', index=False)
    sub_df.head()