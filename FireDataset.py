import os
from swinunetr import SwinUNETR

import numpy as np
import torch
from monai.metrics import MeanIoU, DiceMetric
from torch.utils.data import Dataset, DataLoader


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        for i in range(len(self.mean)):
            sample[:, i, ...] = (sample[:, i, ...] - self.mean[i]) / self.std[i]
        return sample

class FireDataset(Dataset):
    def __init__(self, image_path, label_path, transform=None, n_channel=8, label_sel=2):
        self.image_path, self.label_path = image_path, label_path
        self.num_samples = np.load(self.image_path).shape[0]
        self.transform = transform
        self.n_channel = n_channel
        self.label_sel = label_sel

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # load a chunk of data from disk
        data_chunk, label_chunk = self.load_data(idx)
        if self.transform:
            data_chunk = self.transform(data_chunk)
        sample = {
            'data': data_chunk,
            'labels': label_chunk,
        }

        return sample

    # define a function to load a batch of data from disk
    def load_data(self, indices):
        # load a chunk of data from disk
        data_chunk = np.load(self.image_path, mmap_mode='r')[indices]
        label_chunk = np.load(self.label_path, mmap_mode='r')[indices]

        if self.n_channel==6:
            img_dataset = data_chunk[2:, :8, :, :]
        else:
            img_dataset = data_chunk[:, :8, :, :]
        label_dataset = label_chunk[[self.label_sel], :8, :, :]
        # 0 NIFC 1 VIIRS AF ACC 2 combine
        y_dataset = np.zeros((2, 8,256,256))
        # y_dataset = np.where(label_dataset > 0, 1, 0)
        # y_dataset = np.where(af_dataset > 0, 2, y_dataset)
        y_dataset[0, :, :, :] = label_dataset[..., :] == 0
        y_dataset[1, :, :, :] = label_dataset[..., :] > 0

        x_array, y_array = img_dataset, y_dataset
        x_array_copy = x_array.copy()
        # convert the data to a PyTorch tensor
        x = torch.from_numpy(x_array_copy)
        y = torch.from_numpy(y_array).long()

        return x, y

if __name__ == '__main__':
    root_path = '/geoinfo_vol1/home/z/h/zhao2/CalFireMonitoring'
    ts_length = 8
    image_path = os.path.join(root_path, 'data_train_proj5/proj5_train_img_seqtoseq_alll_' + str(ts_length) + '.npy')
    label_path = os.path.join(root_path, 'data_train_proj5/proj5_train_label_seqtoseq_alll_' + str(ts_length) + '.npy')
    # transform = Normalize(mean=[-0.02396825, -0.00982363, -0.03872192, -0.04996127, -0.0444024, -0.04294463],
    #                       std=[0.9622167, 0.9731459, 0.96916544, 0.96462715, 0.9488478, 0.965671])
    transform = Normalize(mean=[19.13, 25.65, 22.53, 264.53, 257.15, 18.07, 241.96, 240.91],
                          std=[10.25, 12.63, 12.46, 119.31, 115.99, 12.00, 108.09, 107.48])
    train_dataset = FireDataset(image_path=image_path, label_path=label_path, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    num_heads = 2
    hidden_size = 36
    mean_iou = MeanIoU(include_background=True, reduction="mean")
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    image_size = (8, 256, 256)
    patch_size = (1, 2, 2)
    window_size = (8, 7, 7)

    model = SwinUNETR(
        image_size=image_size,
        patch_size=patch_size,
        window_size=window_size,
        in_channels=8,
        out_channels=1,
        depths=(2, 2, 2, 2),
        num_heads=(num_heads, 2*num_heads, 3*num_heads, 4*num_heads),
        feature_size=hidden_size,
        norm_name='batch',
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        attn_version='v2',
        normalize=True,
        use_checkpoint=False,
        spatial_dims=3
    )

    for epoch in range(1):
        for batch in train_dataloader:
            data_batch = batch['data']
            label_batch = batch['labels']
            print(data_batch.shape, label_batch.shape)
            outputs = model(data_batch)
            mean_iou(outputs, label_batch).mean().item()
            dice_metric(y_pred=outputs, y=label_batch).mean().item()