import os
import pandas as pd
import numpy as np
def slice_dataset(arr, ts_length, interval):
    new_arr = []
    length = arr.shape[0]
    for i in range(0, length, interval):
        if i+8>=length:
            print('Drop tail')
            break
        new_arr.append(arr[i:i+8,:,:,:])
    new_arr = np.stack(new_arr, axis=0)
    return new_arr

print('dataset_slice train, validation')
ts_length = 8
interval = 3
root_path = '/geoinfo_vol1/home/z/h/zhao2/CalFireMonitoring/'
image_path = os.path.join(root_path, 'data_train_proj5/proj5_train_imgs.npy')
label_path = os.path.join(root_path, 'data_train_proj5/proj5_train_labels.npy')
val_image_path = os.path.join(root_path, 'data_val_proj5/proj5_val_imgs.npy')
val_label_path = os.path.join(root_path, 'data_val_proj5/proj5_val_labels.npy')
image = slice_dataset(np.load(image_path), ts_length, interval)
label = slice_dataset(np.load(label_path), ts_length, interval)
val_image = slice_dataset(np.load(val_image_path), ts_length, interval)
val_label = slice_dataset(np.load(val_label_path), ts_length, interval)


save_image_path = os.path.join(root_path, 'data_train_proj5/proj5_train_img_seqtoseq_alll_' + str(ts_length) + '.npy')
save_label_path = os.path.join(root_path, 'data_train_proj5/proj5_train_label_seqtoseq_alll_' + str(ts_length) + '.npy')
save_val_image_path = os.path.join(root_path, 'data_val_proj5/proj5_val_img_seqtoseql_' + str(ts_length) + '.npy')
save_val_label_path = os.path.join(root_path, 'data_val_proj5/proj5_val_label_seqtoseql_' + str(ts_length) + '.npy')

print(image.shape)
print(label.shape)
print(val_image.shape)
print(val_label.shape)

np.save(save_image_path, image)
np.save(save_label_path, label)
np.save(save_val_image_path, val_image)
np.save(save_val_label_path, val_label)

print('Done')
dfs=[]
for year in ['2021']:
    filename = 'roi/us_fire_' + year + '_out_new.csv'
    df = pd.read_csv(filename)
    dfs.append(df)
df = pd.concat(dfs, ignore_index=True)
ids = df['Id'].values.astype(str)
ts_length=8
f1_all = 0
iou_all = 0
# ids = ['US_2021_CA3451712013120211011', 'US_2021_CA4086312235520210630', 'US_2021_CA3658211879520210912', 'US_2021_CA3627811855020210815', 'US_2021_CA3604711863120210910', 'US_2021_CA3568711855020210818']
for id in ids:
    print('dataset_slice', id)
    test_image_path = os.path.join(root_path,
                                    'data_test_proj5/proj5_'+id+'.npy')
    test_label_path = os.path.join(root_path,
                                    'data_test_proj5/proj5_'+id+'.npy')

    test_img = slice_dataset(np.load(test_image_path), ts_length, interval)
    test_label = slice_dataset(np.load(test_label_path), ts_length, interval)
    
    np.save(os.path.join(root_path,
                                    'data_test_proj5/proj5_'+id+'_img_seqtoseql_' + str(ts_length) + '.npy'), test_img)
    np.save(os.path.join(root_path,
                                    'data_test_proj5/proj5_'+id+'_label_seqtoseql_' + str(ts_length) + '.npy'), test_label)
print('done')