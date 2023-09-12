import argparse
import heapq
import platform
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "7,8,9,10"
import numpy as np
import torch
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
from monai.losses.dice import DiceLoss
from monai.metrics import MeanIoU, DiceMetric
from torch import nn, optim
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm
import wandb
from FireDataset import FireDataset
from swinunetr import SwinUNETR
from sklearn.metrics import f1_score, jaccard_score
import torchvision.transforms as transforms
import pandas as pd


if platform.system() == 'Darwin':
    root_path = 'data'

else:
    root_path = '/geoinfo_vol1/home/z/h/zhao2/CalFireMonitoring/'

def wandb_config(model_name, num_heads, hidden_size, batch_size):
    wandb.login()
    # wandb.init(project="tokenized_window_size" + str(window_size) + str(model_name) + 'run' + str(run), entity="zhaoyutim")
    wandb.init(project="proj5_"+model_name+"_grid_search", entity="zhaoyutim")
    wandb.run.name = 'num_heads_' + str(num_heads) +'hidden_size_'+str(hidden_size)+'batchsize_'+str(batch_size)
    wandb.config = {
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "epochs": MAX_EPOCHS,
        "batch_size": batch_size,
    }


if __name__=='__main__':
    import os
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-m', type=str, help='Model to be executed')
    parser.add_argument('-b', type=int, help='batch size')
    parser.add_argument('-r', type=int, help='run')
    parser.add_argument('-lr', type=float, help='learning rate')
    parser.add_argument('-av', type=str, help='attension version')

    parser.add_argument('-nh', type=int, help='number-of-head')
    # parser.add_argument('-md', type=int, help='mlp-dimension')
    parser.add_argument('-ed', type=int, help='embedding dimension')
    parser.add_argument('-nc', type=int, help='n_channel')
    parser.add_argument('-test', dest='binary_flag', action='store_true', help='embedding dimension')
    parser.set_defaults(binary_flag=False)
    # parser.add_argument('-nl', type=int, help='num_layers')

    args = parser.parse_args()
    model_name = args.m
    batch_size = args.b

    num_heads=args.nh
    # mlp_dim=args.md
    # num_layers=args.nl
    hidden_size=args.ed


    run = args.r
    lr = args.lr
    MAX_EPOCHS = 200
    learning_rate = lr
    weight_decay = lr / 10
    num_classes = 2
    n_channel = args.nc
    ts_length = 8
    top_n_checkpoints = 3
    train = args.binary_flag
    attn_version=args.av

    class Normalize(object):
        def __init__(self, mean, std):
            self.mean = mean
            self.std = std

        def __call__(self, sample):
            for i in range(len(self.mean)):
                sample[:, i, ...] = (sample[:, i, ...] - self.mean[i]) / self.std[i]
            return sample

    # transform = Normalize(mean=[-0.02396825, -0.00982363, -0.03872192, -0.04996127, -0.0444024, -0.04294463],
    #                       std=[0.9622167, 0.9731459, 0.96916544, 0.96462715, 0.9488478, 0.965671])
    # transform = Normalize(mean=[24.27, 29.49, 22.56, 314.99, 308.35, 16.86, 289.21, 287.58],
                        #   std=[6.38, 5.59, 6.21, 9.52, 6.29, 10.73, 11.18, 6.13])
    transform = Normalize(mean=[47.041927,53.98012,36.33056,318.13885,308.42276,29.793797,291.0422,288.78125],
                        std=[22.357374,21.575853,14.279626,11.570329,10.646594,14.370376,11.274373,6.923434])


    # Dataloader
    if not train:
        wandb_config(model_name, num_heads, hidden_size, batch_size)
        image_path = os.path.join(root_path, 'data_train_proj5/proj5_train_img_seqtoseq_alll_' + str(ts_length) + '.npy')
        label_path = os.path.join(root_path, 'data_train_proj5/proj5_train_label_seqtoseq_alll_' + str(ts_length) + '.npy')
        val_image_path = os.path.join(root_path, 'data_val_proj5/proj5_val_img_seqtoseql_' + str(ts_length) + '.npy')
        val_label_path = os.path.join(root_path, 'data_val_proj5/proj5_val_label_seqtoseql_' + str(ts_length) + '.npy')
        train_dataset = FireDataset(image_path=image_path, label_path=label_path, transform=transform, n_channel=n_channel)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataset = FireDataset(image_path=val_image_path, label_path=val_label_path, transform=transform, n_channel=n_channel)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    image_size = (n_channel, 128, 128)
    patch_size = (1, 2, 2)
    window_size = (ts_length, 4, 4)

    model = SwinUNETR(
        image_size=image_size,
        patch_size=patch_size,
        window_size=window_size,
        in_channels=n_channel,
        out_channels=2,
        depths=(2, 2, 2, 2),
        num_heads=(num_heads, 2*num_heads, 3*num_heads, 4*num_heads),
        feature_size=hidden_size,
        norm_name='batch',
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        attn_version=attn_version,
        normalize=True,
        use_checkpoint=False,
        spatial_dims=3
    )
    model = nn.DataParallel(model)
    model.to(device)

    summary(model, (n_channel, 8, 256, 256), batch_dim=0, device=device)
    criterion = DiceLoss(reduction='mean')
    mean_iou = MeanIoU(include_background=True, reduction="mean")
    dice_metric = DiceMetric(include_background=True, reduction="mean")

    optimizer = optim.SGD(model.parameters(), lr=lr)
    scaler = GradScaler()
    model.to(device)
    best_checkpoints = []
    if not train:
        # create a progress bar for the training loop
        for epoch in range(MAX_EPOCHS):
            model.train()
            train_loss = 0.0
            train_bar = tqdm(train_dataloader, total=len(train_dataloader))
            for i, batch in enumerate(train_bar):
                data_batch = batch['data'].to(device)
                labels_batch = batch['labels'].to(torch.long).to(device)

                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    outputs = model(data_batch)
                    loss = criterion(outputs, labels_batch)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.detach().item() * data_batch.size(0)
                train_bar.set_description(f"Epoch {epoch}/{MAX_EPOCHS}, Loss: {train_loss/((i+1)* data_batch.size(0)):.4f}")

            train_loss /= len(train_dataset)
            wandb.log({'train_loss': train_loss})

            print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}")
            wandb.log({'epoch': epoch})

            model.eval()
            val_loss = 0.0
            iou_values = []
            dice_values = []
            val_bar = tqdm(val_dataloader, total=len(val_dataloader))
            for j, batch in enumerate(val_bar):
                val_data_batch = batch['data'].to(device)
                val_labels_batch = batch['labels'].to(device)

                outputs = model(val_data_batch)
                loss = criterion(outputs, val_labels_batch)

                val_loss += loss.detach().item() * val_data_batch.size(0)
                iou_values.append(mean_iou(outputs, val_labels_batch).mean().item())
                dice_values.append(dice_metric(y_pred=outputs, y=val_labels_batch).mean().item())
                val_bar.set_description(
                    f"Epoch {epoch}/{MAX_EPOCHS}, Loss: {val_loss / ((j + 1) * val_data_batch.size(0)):.4f}")

            val_loss /= len(val_dataset)
            mean_iou_val = np.mean(iou_values)
            mean_dice_val = np.mean(dice_values)
            wandb.log({'val_loss': val_loss, 'miou': mean_iou_val, 'mdice': mean_iou_val})
            print(f"Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}, Mean IoU: {mean_iou_val:.4f}, Mean Dice: {mean_dice_val:.4f}")

            # Save the top N model checkpoints based on validation loss
            if len(best_checkpoints) < top_n_checkpoints or val_loss < best_checkpoints[0][0]:
                save_path = f"num_heads_{num_heads}_hidden_size_{hidden_size}_batchsize_{batch_size}_checkpoint_epoch_{epoch + 1}_attention_{attn_version}_nc_{n_channel}.pth"

                if len(best_checkpoints) == top_n_checkpoints:
                    # Remove the checkpoint with the highest validation loss
                    _, remove_checkpoint = heapq.heappop(best_checkpoints)
                    if os.path.exists(remove_checkpoint):
                        os.remove(remove_checkpoint)

                # Save the new checkpoint
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, save_path)

                # Add the new checkpoint to the priority queue
                heapq.heappush(best_checkpoints, (val_loss, save_path))

                # Ensure that the priority queue has at most N elements
                best_checkpoints = heapq.nlargest(top_n_checkpoints, best_checkpoints)
        print("Top N best checkpoints:")
        for _, checkpoint in best_checkpoints:
            print(checkpoint)
    else:
        # ids = ['24332628', '24461623']
        # ids = []
        # for i in range(5):
        #     for j in range(13):
        #         ids.append('5_13_'+str(i)+'_'+str(j)+'Alberta')
        # ids = ['5_13_0_7Alberta', '5_13_1_8Alberta', '5_13_2_9Alberta', '5_13_3_10Alberta', '5_13_4_11Alberta']
        dfs=[]
        for year in ['2021']:
            filename = 'roi/us_fire_' + year + '_out_new.csv'
            df = pd.read_csv(filename)
            dfs.append(df)
        df = pd.concat(dfs, ignore_index=True)
        ids = df['Id'].values.astype(str)
        label_sel = df['label_sel'].values.astype(int)
        ts_length=8
        interval=3
        f1_all = 0
        iou_all = 0
        mean_iou = MeanIoU(include_background=True, reduction="mean", ignore_empty=False)
        dice_metric = DiceMetric(include_background=True, reduction="mean", ignore_empty=False)
        ids = ['US_2021_NM3676810505920211120']
        for i, id in enumerate(ids):
            test_image_path = os.path.join(root_path,
                                           'data_test_proj5/proj5_'+id+'_img_seqtoseql_' + str(ts_length) + '.npy')
            test_label_path = os.path.join(root_path,
                                           'data_test_proj5/proj5_'+id+'_label_seqtoseql_' + str(ts_length) + '.npy')
            test_dataset = FireDataset(image_path=test_image_path, label_path=test_label_path, transform=transform, n_channel=n_channel, label_sel=label_sel[i])
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            # Load the model checkpoint
            load_epoch = 199
            load_path = f"num_heads_{num_heads}_hidden_size_{hidden_size}_batchsize_{batch_size}_checkpoint_epoch_{load_epoch}_attention_{attn_version}_nc_{n_channel}.pth"

            checkpoint = torch.load(load_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            loaded_epoch = checkpoint['epoch']
            loaded_loss = checkpoint['loss']

            # Make sure to set the model to eval or train mode after loading
            model.eval()  # or model.train()
            def normalization(array):
                return (array-array.min()) / (array.max() - array.min())


            output_stack = np.zeros((256, 256))
            f1=0
            iou=0
            length=0
            for j, batch in enumerate(test_dataloader):
                test_data_batch = batch['data']
                test_labels_batch = batch['labels']
                outputs = model(test_data_batch.to(device)).cpu().detach().numpy()
                import matplotlib.pyplot as plt
                # if j==0:
                #     length+=ts_length
                # else:
                #     length+=interval
                length += test_data_batch.shape[0] * ts_length
                # output_origin = np.zeros((ts_length, 256,256))
                # label_origin = np.zeros((ts_length, 256,256))

                # output_origin[:,:128,:128] = outputs[0, 0, :, :, :]>0.5
                # output_origin[:,:128,128:256] = outputs[1, 0, :, :, :]>0.5
                # output_origin[:,128:256,:128] = outputs[2, 0, :, :, :]>0.5
                # output_origin[:,128:256,128:256] = outputs[3, 0, :, :, :]>0.5

                # label_origin[:,:128,:128] = (test_labels_batch[0, 0, :, :, :]>0).numpy()
                # label_origin[:,:128,128:256] = (test_labels_batch[1, 0, :, :, :]>0).numpy()
                # label_origin[:,128:256,:128] = (test_labels_batch[2, 0, :, :, :]>0).numpy()
                # label_origin[:,128:256,128:256] = (test_labels_batch[3, 0, :, :, :]>0).numpy()
                for k in range(test_data_batch.shape[0]):
                    for i in range(ts_length):
                        # output_stack = output_origin[[i],...]
                        # label = label_origin[[i],...]
                        output_stack = np.logical_or(output_stack, outputs[k, 1, i, :, :]>0.5)
                        label = test_labels_batch[k, 1, i, :, :]>0
                        label = label.numpy()

                        f1_ts = f1_score(label.flatten(), output_stack.flatten(), zero_division=1.0)
                        f1 += f1_ts
                        iou_ts = jaccard_score(label.flatten(), output_stack.flatten(), zero_division=1.0)
                        iou += iou_ts
                        print(f1_ts, iou_ts)
                        # print(f1_ts, iou_ts)
                        # f1_ts = dice_metric(torch.tensor(output_stack), torch.tensor(label)).mean().item()
                        # if j==0 or i>=ts_length-interval:
                        #     step+=1
                        #     f1 += f1_ts
                        # iou_ts = mean_iou(torch.tensor(output_stack), torch.tensor(label)).mean().item()
                        # if j==0 or i>=ts_length-interval:
                        #     iou += iou_ts


                        # print('Batch{}, TS{}, with F1 Score{}, IoU Score{}'.format(k, i, f1_ts, iou_ts))
                        # print('Batch{}, TS{}, with F1_AF Score{}, IoU_AF Score{}'.format(k, i, f1_af_ts, iou_af_ts))
                        # ba_img = np.zeros((128,128,3))
                        # ba_img[:,:,0] = normalization(test_data_batch[k, 5, i, :, :])
                        # ba_img[:, :, 1] = normalization(test_data_batch[k, 1, i, :, :])
                        # ba_img[:, :, 2] = normalization(test_data_batch[k, 0, i, :, :])

                        f, axarr = plt.subplots(2, 2)
                        axarr[0,0].imshow(output_stack)
                        axarr[0,0].set_title('Prediction')
                        axarr[0,0].axis('off')

                        axarr[0,1].imshow(label)
                        axarr[0,1].set_title('Ground Truth')
                        axarr[0,1].axis('off')

                        if n_channel!=6:
                            axarr[1, 0].imshow(normalization(test_data_batch[k, 3, i, :, :]))
                            axarr[1, 0].set_title('VIIRS I4 Day')
                            axarr[1, 0].axis('off')

                            axarr[1, 1].imshow(normalization(test_data_batch[k, 6, i, :, :]))
                            axarr[1, 1].set_title('VIIRS I4 Night')
                            axarr[1, 1].axis('off')
                        else:
                            axarr[1, 0].imshow(normalization(test_data_batch[k, 1, i, :, :]))
                            axarr[1, 0].set_title('VIIRS I4 Day')
                            axarr[1, 0].axis('off')
                            
                            axarr[1, 1].imshow(normalization(test_data_batch[k, 4, i, :, :]))
                            axarr[1, 1].set_title('VIIRS I4 Night')
                            axarr[1, 1].axis('off')

                        f.savefig('results_0912/id_{}_nhead_{}_hidden_{}_nbatch_{}_nts_{}_ts_{}_av_{}_nc_{}.png'.format(id, num_heads, hidden_size, j, k, i, attn_version, n_channel), bbox_inches='tight')
                        plt.show()
                        plt.close()
            iou_all += iou/length
            f1_all += f1/length
            print('ID{} IoU Score of the whole TS:{}'.format(id, iou/length))
            print('ID{} F1 Score of the whole TS:{}'.format(id, f1/length))
        print('model F1 Score: {} and iou score: {}'.format(f1_all/len(ids), iou_all/len(ids)))


