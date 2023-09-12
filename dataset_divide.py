import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
def divide_dataset(arr):
    new_arr = []
    for k in range(arr.shape[0]):
        for i in range(2):
            for j in range(2):
                new_arr.append(arr[k,:,:,i*128:(i+1)*128,j*128:(j+1)*128])
    new_arr = np.stack(new_arr, axis=0)

    eva_arr = np.zeros((arr.shape[1], arr.shape[2], arr.shape[3], arr.shape[4]))
    eva_arr[:,:,:128,:128] = new_arr[0, ...]
    eva_arr[:,:,:128,128:256] = new_arr[1, ...]
    eva_arr[:,:,128:256,:128] = new_arr[2, ...]
    eva_arr[:,:,128:256,128:256] = new_arr[3, ...]
    assert (arr[0, ...] == eva_arr).all()
    return new_arr

if __name__ == '__main__':
    print('train, validation')
    ts_length = 8
    root_path = '/geoinfo_vol1/home/z/h/zhao2/CalFireMonitoring/'
    image_path = os.path.join(root_path, 'data_train_proj5/proj5_train_img_seqtoseq_alll_' + str(ts_length) + '.npy')
    label_path = os.path.join(root_path, 'data_train_proj5/proj5_train_label_seqtoseq_alll_' + str(ts_length) + '.npy')
    val_image_path = os.path.join(root_path, 'data_val_proj5/proj5_val_img_seqtoseql_' + str(ts_length) + '.npy')
    val_label_path = os.path.join(root_path, 'data_val_proj5/proj5_val_label_seqtoseql_' + str(ts_length) + '.npy')

    image = np.load(image_path)
    label = np.load(label_path)
    val_image = np.load(val_image_path)
    val_label = np.load(val_label_path)

    image = divide_dataset(image)
    label = divide_dataset(label)
    val_image = divide_dataset(val_image)
    val_label = divide_dataset(val_label)
    print(image.shape)
    print(label.shape)
    print(val_image.shape)
    print(val_label.shape)

    for i in range(image.shape[0]):
        for j in range(image.shape[2]):
            data = (image[i,3,j,:,:]-image[i,3,j,:,:].min())/(image[i,3,j,:,:].max()-image[i,3,j,:,:].min())
            label_vis = label[i,2,j,:,:]
            f, axarr = plt.subplots(1, 2)
            axarr[0].imshow(data)
            axarr[0].set_title('data')
            axarr[0].axis('off')

            axarr[1].imshow(label_vis)
            axarr[1].set_title('label')
            axarr[1].axis('off')
            f.savefig('dataset_inspect/i_{}_j_{}.png'.format(i, j), bbox_inches='tight')
            plt.show()
            plt.close()


    np.save(os.path.join(root_path, 'data_train_proj5/proj5_train_img_seqtoseq_alll_' + str(ts_length) + '_devided.npy'), image)
    np.save(os.path.join(root_path, 'data_train_proj5/proj5_train_label_seqtoseq_alll_' + str(ts_length) + '_devided.npy'), label)
    np.save(os.path.join(root_path, 'data_val_proj5/proj5_val_img_seqtoseql_' + str(ts_length) + '_devided.npy'), val_image)
    np.save(os.path.join(root_path, 'data_val_proj5/proj5_val_label_seqtoseql_' + str(ts_length) + '_devided.npy'), val_label)

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
        print(id)
        test_image_path = os.path.join(root_path,
                                        'data_test_proj5/proj5_'+id+'_img_seqtoseql_' + str(ts_length) + '.npy')
        test_label_path = os.path.join(root_path,
                                        'data_test_proj5/proj5_'+id+'_label_seqtoseql_' + str(ts_length) + '.npy')

        test_img = np.load(test_image_path)
        test_label = np.load(test_label_path)

        test_img = divide_dataset(test_img)
        test_label = divide_dataset(test_label)
        np.save(os.path.join(root_path,
                                        'data_test_proj5/proj5_'+id+'_img_seqtoseql_' + str(ts_length) + '_divided.npy'), test_img)
        np.save(os.path.join(root_path,
                                        'data_test_proj5/proj5_'+id+'_label_seqtoseql_' + str(ts_length) + '_divided.npy'), test_label)
    print('done')