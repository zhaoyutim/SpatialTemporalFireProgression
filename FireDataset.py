import os

import numpy as np
import torch
from monai.transforms import (
    LoadImage,
)
from torch.utils.data import Dataset, DataLoader

from data_processor.tokenize_processor import TokenizeProcessor


class FireDataset(Dataset):
    def __init__(self, image_path, label_path):
        self.array, self.labels = self.get_dataset(image_path, label_path)

    def __len__(self):
        return len(self.array)

    def __getitem__(self, idx):
        sample = {
            'data': torch.from_numpy(self.array[idx]).float(),
            'labels': torch.from_numpy(self.labels[idx]).float(),
        }
        return sample

    def get_dataset(self, image_path, label_path):
        img_dataset = np.load(image_path).transpose((0,2,1,3,4))
        label_dataset = np.load(label_path).transpose((0,2,1,3,4))
        img_dataset = img_dataset[:, :, :8, :, :]
        label_dataset = label_dataset[:, :, :8, :, :]
        # y_dataset = np.zeros((label_dataset.shape))
        y_dataset = label_dataset[..., :]>0
        # y_dataset[..., 1] = label_dataset[..., 0] > 0

        x_array, y_array = img_dataset, y_dataset,

        return x_array, y_array

if __name__ == '__main__':
    root_path = 'data'
    ts_length = 10
    image_path = os.path.join(root_path, 'proj5_train_img_seqtoseq_l' + str(ts_length) + '.npy')
    label_path = os.path.join(root_path, 'proj5_train_label_seqtoseq_l' + str(ts_length) + '.npy')
    train_dataset = FireDataset(image_path=image_path, label_path=label_path)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    for epoch in range(1):
        for batch in train_dataloader:
            data_batch = batch['data']
            labels_batch = batch['labels']
            print(data_batch.shape, labels_batch.shape)