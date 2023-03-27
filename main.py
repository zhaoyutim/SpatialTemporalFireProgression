import argparse
import heapq
import platform


import torch
import wandb
from swinunetr import SwinUNETR
from torch import nn, optim
from torch.utils.data import DataLoader
from torchinfo import summary

from FireDataset import FireDataset

if platform.system() == 'Darwin':
    root_path = 'data'

else:
    root_path = '/geoinfo_vol1/zhao2/proj5_dataset'

def wandb_config(model_name, run, num_heads, num_layers, mlp_dim, hidden_size):
    wandb.login()
    # wandb.init(project="tokenized_window_size" + str(window_size) + str(model_name) + 'run' + str(run), entity="zhaoyutim")
    wandb.init(project="proj3_"+model_name+"_grid_search", entity="zhaoyutim")
    wandb.run.name = 'num_heads_' + str(num_heads) + 'num_layers_'+ str(num_layers)+ 'mlp_dim_'+str(mlp_dim)+'hidden_size_'+str(hidden_size)+'batchsize_'+str(batch_size)+'run_'+str(run)
    wandb.config = {
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "epochs": MAX_EPOCHS,
        "batch_size": batch_size,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "mlp_dim": mlp_dim,
        "embed_dim": hidden_size
    }

if __name__=='__main__':
    import os
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-m', type=str, help='Model to be executed')
    parser.add_argument('-b', type=int, help='batch size')
    parser.add_argument('-r', type=int, help='run')
    parser.add_argument('-lr', type=float, help='learning rate')

    # parser.add_argument('-nh', type=int, help='number-of-head')
    # parser.add_argument('-md', type=int, help='mlp-dimension')
    # parser.add_argument('-ed', type=int, help='embedding dimension')
    # parser.add_argument('-nl', type=int, help='num_layers')

    args = parser.parse_args()
    model_name = args.m
    batch_size = args.b

    # num_heads=args.nh
    # mlp_dim=args.md
    # num_layers=args.nl
    # hidden_size=args.ed

    run = args.r
    lr = args.lr
    MAX_EPOCHS = 50
    learning_rate = lr
    weight_decay = lr / 10
    num_classes=2
    ts_length=10
    top_n_checkpoints = 3

    # wandb_config(model_name, run, num_heads, mlp_dim, num_layers, hidden_size)

    # Dataloader
    image_path = os.path.join(root_path, 'proj5_train_img_seqtoseq_l' + str(ts_length) + '.npy')
    label_path = os.path.join(root_path, 'proj5_train_label_seqtoseq_l' + str(ts_length) + '.npy')
    val_image_path = os.path.join(root_path, 'proj5_train_img_seqtoseq_l' + str(ts_length) + '.npy')
    val_label_path = os.path.join(root_path, 'proj5_train_label_seqtoseq_l' + str(ts_length) + '.npy')
    train_dataset = FireDataset(image_path=image_path, label_path=label_path)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = FireDataset(image_path=val_image_path, label_path=val_label_path)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_size = (8, 512, 512)
    patch_size = (1, 2, 2)
    window_size = (2, 7, 7)

    model = SwinUNETR(
        image_size=image_size,
        patch_size=patch_size,
        window_size=window_size,
        in_channels=6,
        out_channels=2,
        depths=(2, 2, 2, 2),
        num_heads=(4, 8, 12, 16),
        feature_size=24,
        norm_name='batch',
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        attn_version='v2',
        normalize=True,
        use_checkpoint=False,
        spatial_dims=3
    ).to(device)

    summary(model, (6, 8, 512, 512), batch_dim=0)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    best_checkpoints = []
    for epoch in range(MAX_EPOCHS):
        model.train()
        train_loss = 0.0

        for batch in train_dataloader:
            data_batch = batch['data'].to(device)
            labels_batch = batch['labels'].to(device)

            optimizer.zero_grad()

            outputs = model(data_batch)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * data_batch.size(0)

        train_loss /= len(train_dataset)
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}")

        model.eval()
        val_loss = 0.0

        for batch in val_dataloader:
            data_batch = batch['data'].to(device)
            labels_batch = batch['labels'].to(device)

            outputs = model(data_batch)
            loss = criterion(outputs, labels_batch)

            val_loss += loss.item() * data_batch.size(0)

        val_loss /= len(val_dataset)
        print(f"Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}")

        # Save the top N model checkpoints based on validation loss
        if len(best_checkpoints) < top_n_checkpoints or val_loss < best_checkpoints[0][0]:
            save_path = f"checkpoint_epoch_{epoch + 1}.pth"

            if len(best_checkpoints) == top_n_checkpoints:
                # Remove the checkpoint with the highest validation loss
                _, remove_checkpoint = heapq.heappop(best_checkpoints)
                os.remove(remove_checkpoint)

            # Save the new checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, save_path)

            # Add the new checkpoint to the priority queue
            heapq.heappush(best_checkpoints, (-val_loss, save_path))

            # Ensure that the priority queue has at most N elements
            best_checkpoints = heapq.nsmallest(top_n_checkpoints, best_checkpoints)
    print("Top N best checkpoints:")
    for _, checkpoint in best_checkpoints:
        print(checkpoint)
# # Load the model checkpoint
# load_path = "checkpoint.pth"
#
# checkpoint = torch.load(load_path)
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# loaded_epoch = checkpoint['epoch']
# loaded_loss = checkpoint['loss']
#
# # Make sure to set the model to eval or train mode after loading
# model.eval()  # or model.train()


