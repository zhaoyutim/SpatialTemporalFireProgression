import numpy as np
import matplotlib.pyplot as plt

class TokenizeProcessor:

    def tokenizing(self, data_path, window_size):
        array = np.load(data_path).transpose((0, 3, 4, 1, 2))
        output_shape = (array.shape[0], array.shape[1], array.shape[2], array.shape[3], (array.shape[4]-2) * pow(window_size, 2)+2)

        output_array = np.zeros(output_shape)
        padding = window_size // 2
        padded_array = np.pad(array, pad_width=((0,0),(padding,padding),(padding,padding),(0,0),(0,0)), mode='constant', constant_values=0)
        shape = padded_array.shape[1]
        for num_sample in range(array.shape[0]):
            for i in range(padding, shape-padding):
                for j in range(padding, shape-padding):
                    output_array[num_sample, i-padding, j-padding, :, :] = self.flatten_window(padded_array[num_sample, i-padding:i+padding+1, j-padding:j+padding+1, :, :], window_size)
        # print(output_array.shape)
        return output_array
    def flatten_window(self, array, window_size):
        # print(array.shape)
        output_array = np.zeros((array.shape[2], (array.shape[3]-2)*pow(window_size,2)+2))
        for time in range(array.shape[2]):
            output_array[time, :output_array.shape[1]-2] = array[:, :, time, :5].flatten('F')
            output_array[time, output_array.shape[1]-2] = array[window_size//2, window_size//2, time, 5]
            output_array[time, output_array.shape[1]-1] = array[window_size // 2, window_size // 2, time, 6]
        return output_array



if __name__=='__main__':
    window_size = 1
    locations= ['sydney']
    for location in locations:
        tokenize_processor = TokenizeProcessor()
        tokenized_array = tokenize_processor.tokenizing(data_path='/Users/zhaoyu/PycharmProjects/CalFireMonitoring/data_train_proj5/proj5_'+location+'_img.npy', window_size=window_size)
        np.nan_to_num(tokenized_array)
        np.save('../data/proj5_'+location+'_w'+str(window_size)+'.npy', tokenized_array.reshape(-1,10,pow(window_size,2)*5+2))
    # np.save('../data/proj3_test_w' + str(window_size) + 'patch_seg.npy',
    #         tokenized_array)
    # np.save('../data/proj3_test_w' + str(window_size) + 'label.npy',
    #         label_array)
    # for i in range(tokenized_array.shape[0]):
    #     for j in range(tokenized_array.shape[3]):
    #         plt.subplot(211)
    #         ch_label = pow(window_size,2)*5
    #         ch_i4 = pow(window_size,2)*3+pow(window_size,2)//2
    #         plt.imshow(
    #             (tokenized_array[i, :, :, j, ch_label] - tokenized_array[i, :, :, j, ch_label].min()) - (
    #                         tokenized_array[i, :, :, j, ch_label].max() - tokenized_array[i, :, :, j, ch_label].min()))
    #         plt.subplot(212)
    #         plt.imshow(
    #             (tokenized_array[i, :, :, j, ch_i4] - tokenized_array[i, :, :, j, ch_i4].min()) - (
    #                         tokenized_array[i, :, :, j, ch_i4].max() - tokenized_array[i, :, :, j, ch_i4].min()))
    #         plt.savefig('../plt/' + str(i) + str(j) + '.png')
    #         plt.show()
    # def standardize(array):
    #     return (array-array.min())/(array.max()-array.min())
    #
    #
    # img = np.zeros((226, 226))
    # label = np.zeros((226, 226))
    # for num in range(58):
    #     for k in range(10):
    #         for i in range(1, 224, 2):
    #             for j in range(1, 224, 2):
    #                 iidx = int((i - 1) / (1 * 2))
    #                 jidx = int((j - 1) / (1 * 2))
    #                 label[i - window_size // 2:i + window_size // 2 + 1,
    #                 j - window_size // 2:j + window_size // 2 + 1] = label_array[iidx*112 + jidx][k, :].reshape(
    #                     (window_size, window_size), order='F')
    #                 img[i - window_size // 2:i + window_size // 2 + 1,
    #                 j - window_size // 2:j + window_size // 2 + 1] = tokenized_array[iidx * 112 + jidx][k, 27:36].reshape(
    #                     (window_size, window_size), order='F')
    #
    #         plt.subplot(211)
    #         plt.imshow(standardize(img))
    #         plt.subplot(212)
    #         plt.imshow(standardize(label))
    #         plt.savefig('../plt_test/' + str(num) + str(k) + '.png')
    #         plt.show()

