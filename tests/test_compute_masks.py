from utils.utils import compute_mask

if __name__=='__main__':
    dims=(8,128,128)
    window_size=(8,4,4)
    shift_size=(0,2,2)
    device='cpu'
    compute_mask(dims,window_size,shift_size,device)