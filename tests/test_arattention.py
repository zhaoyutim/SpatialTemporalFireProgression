import numpy as np
import torch
import matplotlib.pyplot as plt
from utils.utils import get_temporal_mask

if __name__ == '__main__':
    dims = (8, 128, 128)
    window_size = (8,4,4)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    temp_mask_windows = get_temporal_mask(dims, window_size, device=device)
    temp_mask_windows = temp_mask_windows.cpu().numpy()
    for i in range(128):
        plt.imshow(temp_mask_windows[:, :, i], vmin=0, vmax=1)
        plt.savefig('../temp_mask_test/'+str(i)+'.png')
        plt.show()

    # aratt = AutoregressiveAttntion(dim=36, num_heads=3, window_size=(2,7,7), qkv_bias= False, attn_drop=0.0, proj_drop=0.0)
    # aratt(np.zeros(1444,98,36))