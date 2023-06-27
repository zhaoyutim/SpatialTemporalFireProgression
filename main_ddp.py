import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import torch
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import Dataset, DataLoader
from FireDataset import FireDataset
from swinunetr import SwinUNETR
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from tqdm import tqdm

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12358"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

class Trainer:
    def __init__(
            self,
            model: torch.nn.Module,
            train_data: DataLoader,
            val_data: DataLoader,
            optimizer: torch.optim.Optimizer,
            gpu_id: int,
            save_every: int,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.model = DDP(model, device_ids=[gpu_id])

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        self.optimizer.step()
        return loss

    def _run_batch_val(self, source, targets):
        output = self.model(source)
        val_loss = F.cross_entropy(output, targets)
        return val_loss

    def _run_epoch(self, epoch, wandb_logger):
        b_sz = next(iter(self.train_data))['data'].size(0)
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)

        self.model.train()
        train_loss = 0.0
        train_bar = tqdm(self.train_data, total=len(self.train_data))
        for i, batch in enumerate(train_bar):
            source = batch['data'].to(self.gpu_id)
            targets = batch['labels'].to(self.gpu_id)
            train_loss += self._run_batch(source, targets).detach().item() * source.size(0)
            train_bar.set_description(
                f"Epoch {epoch}/{self.max_ephoches}, Loss: {train_loss / ((i + 1) * source.size(0)):.4f}")
        train_loss /= len(self.train_data)
        wandb_logger.log({'train_loss': train_loss})
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}")
        wandb_logger.log({'epoch': epoch})

        self.model.eval()
        val_loss = 0.0
        for batch in self.val_data:
            source = batch['data'].to(self.gpu_id)
            targets = batch['labels'].to(self.gpu_id)
            val_loss += self._run_batch_val(source, targets).detach().item() * source.size(0)

        val_loss /= len(self.val_data)
        wandb_logger.log({'val_loss': val_loss})
        print(f"Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}")

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = "checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int, wandb_logger):
        wandb.init(project="proj3_SwinUNETR_grid_search", entity="zhaoyutim", group="DDP")
        self.max_ephoches = max_epochs
        for epoch in range(self.max_ephoches):
            self._run_epoch(epoch, wandb_logger)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)


def load_train_objs(lr: float):
    root_path = '/geoinfo_vol1/home/z/h/zhao2/CalFireMonitoring/data_train_proj5/'
    ts_length=8
    image_path = os.path.join(root_path, 'proj5_train_img_seqtoseq_alll_' + str(ts_length) + '.npy')
    label_path = os.path.join(root_path, 'proj5_train_label_seqtoseq_alll_' + str(ts_length) + '.npy')
    val_image_path = os.path.join(root_path, 'proj5_train_img_seqtoseq_alll_' + str(ts_length) + '.npy')
    val_label_path = os.path.join(root_path, 'proj5_train_label_seqtoseq_alll_' + str(ts_length) + '.npy')
    train_dataset = FireDataset(image_path=image_path, label_path=label_path)
    val_dataset = FireDataset(image_path=val_image_path, label_path=val_label_path)


    image_size = (8, 256, 256)
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
    )
    optimizer = optim.Adam(model.parameters(), lr=lr)
    return train_dataset, val_dataset, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )


def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int, lr: float, wandb_logger):
    ddp_setup(rank, world_size)
    dataset, val_dataset, model, optimizer = load_train_objs(lr)
    train_data_loader = prepare_dataloader(dataset, batch_size)
    val_data_loader = prepare_dataloader(val_dataset, batch_size)
    trainer = Trainer(model, train_data_loader, val_data_loader, optimizer, rank, save_every)
    trainer.train(total_epochs, wandb_logger)
    destroy_process_group()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('-m', type=str, help='Model to be executed')
    parser.add_argument('-b', type=int, help='batch size')
    parser.add_argument('-r', type=int, help='run')
    parser.add_argument('-lr', type=float, help='learning rate')

    args = parser.parse_args()
    model_name = args.m
    batch_size = args.b
    run = args.r
    lr = args.lr
    lr = lr
    weight_decay = lr / 10
    total_epoches = 50
    save_every = 5
    wandb.login()
    wandb_logger = wandb.init(project="proj3_"+model_name+"_grid_search", entity="zhaoyutim", group="DDP")
    wandb.config = {
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "epochs": total_epoches,
        "batch_size": batch_size,
    }

    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, save_every, total_epoches, batch_size, lr, wandb_logger), nprocs=world_size)