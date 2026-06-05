import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision.models import resnet50, ResNet50_Weights
import argparse

# ✅ LazyFakeDataset: 메모리 아끼는 가짜 데이터셋
class LazyFakeDataset(Dataset):
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        x = torch.randn(3, 224, 224)
        y = torch.randint(0, 100, (1,)).item()
        return x, y


def setup(rank, world_size, use_cuda):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29501'
    backend = "nccl" if use_cuda else "gloo"
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    if use_cuda:
        torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size, args, use_cuda):
    setup(rank, world_size, use_cuda)

    if use_cuda:
        device = torch.device(f"cuda:{rank}")
        print(f"[Rank {rank}] Starting training on GPU {rank}")
    else:
        device = torch.device("cpu")
        print(f"[Rank {rank}] Starting training on CPU")

    start_time = time.time()

    model = resnet50(weights=None, num_classes=args.num_classes).to(device)

    if world_size > 1:
        if use_cuda:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
        else:
            model = nn.parallel.DistributedDataParallel(model)

    dataset = LazyFakeDataset(args.dataset_size)
    pin_memory = use_cuda
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=pin_memory)
    else:
        sampler = None
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin_memory)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        epoch_loss = 0.0
        model.train()

        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"[Rank {rank}] Epoch {epoch:3d} - Avg Loss: {avg_loss:.4f}")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"[Rank {rank}] Training completed in {total_time:.2f} seconds")

    if rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)
        save_path = os.path.join(args.save_dir, "model_final.pth")
        state_dict = model.module.state_dict() if world_size > 1 else model.state_dict()
        torch.save(state_dict, save_path)
        print(f"[Rank 0] Model saved to {save_path}")

    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch DDP ResNet50 AstraGo 워크로드 테스트용")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--dataset-size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--save-dir', type=str, default='./checkpoints')
    parser.add_argument('--num-classes', type=int, default=100)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args([])

    gpu_count = torch.cuda.device_count()
    print(f"Total GPUs available: {gpu_count}")

    use_cuda = not args.cpu and gpu_count > 0

    if use_cuda:
        world_size = gpu_count
        if world_size == 1:
            print("Single GPU detected. Running in single-GPU mode.")
        else:
            print(f"Multiple GPUs detected ({world_size}). Running in DDP mode.")
    else:
        world_size = 1
        if args.cpu:
            print("CPU mode enabled by --cpu flag.")
        else:
            print("No GPU available. Running in CPU mode.")

    mp.spawn(train, args=(world_size, args, use_cuda), nprocs=world_size, join=True)
