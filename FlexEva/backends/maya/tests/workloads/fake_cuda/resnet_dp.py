#!/usr/bin/env python3
"""
ResNet Data Parallel training with DDP.

This demonstrates vision model training with DDP.
Note: Requires cuDNN for convolutions - may not work with fake-cuda.

Run with:
    torchrun --nproc_per_node=2 resnet_dp.py --steps 10

With fake-cuda (likely fails - needs cuDNN):
    ./frun torchrun --nproc_per_node=2 test/resnet_dp.py --steps 10
"""
import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def parse_args():
    ap = argparse.ArgumentParser(description="ResNet DDP training")
    ap.add_argument("--steps", type=int, default=50, help="Training steps")
    ap.add_argument("--batch-size", type=int, default=32, help="Per-rank batch size")
    ap.add_argument("--image-size", type=int, default=224, help="Input image size")
    ap.add_argument("--num-classes", type=int, default=1000, help="Number of classes")
    ap.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "resnet34", "resnet50"],
                    help="ResNet variant")
    ap.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    ap.add_argument("--log-interval", type=int, default=10, help="Log every N steps")
    return ap.parse_args()


def setup_dist():
    if "RANK" not in os.environ:
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    return rank, world_size, local_rank, device


class BasicBlock(nn.Module):
    """Basic residual block for ResNet-18/34."""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    """Bottleneck residual block for ResNet-50+."""
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    """ResNet model."""
    def __init__(self, block, layers, num_classes=1000):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def build_resnet(name: str, num_classes: int):
    configs = {
        "resnet18": (BasicBlock, [2, 2, 2, 2]),
        "resnet34": (BasicBlock, [3, 4, 6, 3]),
        "resnet50": (Bottleneck, [3, 4, 6, 3]),
    }
    block, layers = configs[name]
    return ResNet(block, layers, num_classes)


def main():
    args = parse_args()
    rank, world_size, local_rank, device = setup_dist()

    torch.manual_seed(42 + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42 + rank)

    model = build_resnet(args.model, args.num_classes).to(device)

    # Wrap with DDP
    model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    num_params = sum(p.numel() for p in model.parameters())
    if rank == 0:
        print({
            "world_size": world_size,
            "model": args.model,
            "image_size": args.image_size,
            "batch_size": args.batch_size,
            "params_m": round(num_params / 1e6, 2),
        })

    # Training loop
    start_time = time.time()
    for step in range(1, args.steps + 1):
        # Synthetic image data (batch, channels, height, width)
        images = torch.randn(args.batch_size, 3, args.image_size, args.image_size, device=device)
        labels = torch.randint(0, args.num_classes, (args.batch_size,), device=device)

        # Forward
        logits = model(images)
        loss = F.cross_entropy(logits, labels)

        # Backward
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Compute accuracy
        with torch.no_grad():
            pred = logits.argmax(dim=1)
            acc = (pred == labels).float().mean()

        if rank == 0 and (step % args.log_interval == 0 or step == 1 or step == args.steps):
            elapsed = time.time() - start_time
            print(f"step {step:5d}/{args.steps} | loss={loss.item():.4f} | acc={acc.item():.4f}")

    dist.barrier()
    dist.destroy_process_group()

    if rank == 0:
        print("Training complete.")


if __name__ == "__main__":
    main()
