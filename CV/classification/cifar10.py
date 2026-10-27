import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from optimizer import TCG


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=False)
        self.downsample = None
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        identity = x
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu2(out + identity)


class CIFARResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], 1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], 2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], 2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        layers = []
        for s in [stride] + [1] * (num_blocks - 1):
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def convnet(num_classes=10):
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, 1, 1),
        nn.ReLU(),
        nn.AvgPool2d(2, 2),
        nn.Conv2d(32, 64, 3, 1, 1),
        nn.ReLU(),
        nn.AvgPool2d(2, 2),
        nn.Conv2d(64, 64, 3, 1, 1),
        nn.ReLU(),
        nn.AvgPool2d(2, 2),
        nn.Conv2d(64, 128, 3, 1, 1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(128, num_classes),
    )


def build_model(args):
    if args.arch == "cnn":
        return convnet()
    if args.arch == "resnet18":
        return CIFARResNet(BasicBlock, [2, 2, 2, 2])
    raise ValueError(f"Unknown architecture: {args.arch}")


def disable_inplace_relu(module):
    for layer in module.modules():
        if isinstance(layer, nn.ReLU):
            layer.inplace = False


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(_worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def accuracy(logits, labels):
    return (logits.argmax(dim=1) == labels).float().mean().item()


def build_dataloaders(args):
    normalize = [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
    augment = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    train_transform = transforms.Compose((augment if args.use_augmentation else []) + normalize)
    test_transform = transforms.Compose(normalize)

    train_dataset = datasets.CIFAR10(args.data_root, train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(args.data_root, train=False, download=True, transform=test_transform)

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size_test,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
    )
    return train_dataset, train_loader, test_loader


def build_optimizer(model, args):
    if args.base_opt == "sgd":
        base_optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr_sgd,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    elif args.base_opt == "adam":
        base_optimizer = optim.Adam(
            model.parameters(),
            lr=args.lr_adam,
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            weight_decay=args.weight_decay,
        )
    else:
        raise ValueError(f"Unknown base optimizer: {args.base_opt}")

    if args.method == "baseline":
        return base_optimizer
    if args.method == "tcg":
        return TCG(
            base_optimizer=base_optimizer,
            beta_fast=args.beta_fast,
            beta_slow=args.beta_slow,
            gate_type=args.gate_type,
            gate_tau=args.gate_tau,
        )
    raise ValueError(f"Unknown method: {args.method}")


def maybe_update_lr(optimizer, args, epoch):
    if args.lr_schedule == "constant":
        return
    base_lr = args.lr_sgd if args.base_opt == "sgd" else args.lr_adam
    lr = base_lr * 0.5 * (1.0 + math.cos(math.pi * epoch / (args.epochs + 1)))
    for group in optimizer.param_groups:
        group["lr"] = lr


def unwrap_optimizer(optimizer):
    return optimizer.original_optimizer if hasattr(optimizer, "original_optimizer") else optimizer


def run_group_name(args):
    return (
        f"cifar10_{args.arch}_{args.base_opt}_{args.method}"
        f"_eps{args.target_epsilon}_seed{args.seed}"
    )


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses = []
    accs = []
    for images, labels in tqdm(loader, leave=False):
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        losses.append(criterion(logits, labels).item())
        accs.append(accuracy(logits, labels))
    return float(np.mean(losses)), float(np.mean(accs))


def make_private(model, optimizer, train_loader, args):
    if args.disable_dp:
        return model, optimizer, train_loader, None

    privacy_engine = PrivacyEngine(
        accountant=args.opacus_accountant,
        secure_mode=args.secure_mode,
    )
    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        target_epsilon=args.target_epsilon,
        target_delta=args.delta,
        epochs=args.epochs,
        max_grad_norm=args.max_per_sample_grad_norm,
        poisson_sampling=args.poisson_sampling,
        clipping=args.opacus_clipping,
        grad_sample_mode=args.grad_sample_mode,
    )
    return model, optimizer, train_loader, privacy_engine


def collect_tcg_metrics(optimizer):
    wrapped = unwrap_optimizer(optimizer)
    metrics = {}
    if hasattr(wrapped, "current_gate_mean"):
        metrics["gate_mean"] = float(wrapped.current_gate_mean)
    return metrics


def run_single_seed(args):
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_dataset, train_loader, test_loader = build_dataloaders(args)

    if args.delta is None:
        args.delta = 1.0 / len(train_dataset)

    model = build_model(args)
    disable_inplace_relu(model)
    model = ModuleValidator.fix(model)
    disable_inplace_relu(model)
    model = model.to(device)

    optimizer = build_optimizer(model, args)
    model, optimizer, train_loader, privacy_engine = make_private(model, optimizer, train_loader, args)

    output_dir = Path(args.output_dir) / run_group_name(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        group=run_group_name(args),
        name=run_group_name(args),
        tags=["cifar10", args.arch, args.base_opt, args.method, "opacus"],
        config=vars(args),
        reinit="finish_previous",
    )

    criterion = nn.CrossEntropyLoss()
    rows = []
    best_test_acc = -1.0
    best_metrics = None

    for epoch in range(1, args.epochs + 1):
        maybe_update_lr(optimizer, args, epoch)
        model.train()
        train_losses = []
        train_accs = []

        for images, labels in tqdm(train_loader, leave=False):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            train_accs.append(accuracy(logits.detach(), labels))

        if epoch % args.eval_epochs != 0 and epoch != args.epochs:
            continue

        test_loss, test_acc = evaluate(model, test_loader, device)
        row = {
            "epoch": epoch,
            "seed": args.seed,
            "arch": args.arch,
            "base_opt": args.base_opt,
            "method": args.method,
            "train_loss": float(np.mean(train_losses)),
            "train_acc": float(np.mean(train_accs)),
            "test_loss": test_loss,
            "test_acc": test_acc,
            "current_lr": float(optimizer.param_groups[0]["lr"]),
            "target_epsilon": float(args.target_epsilon),
            **collect_tcg_metrics(optimizer),
        }

        if privacy_engine is not None:
            try:
                row["epsilon_spent"] = float(privacy_engine.get_epsilon(args.delta))
            except Exception:
                pass
            if hasattr(optimizer, "noise_multiplier"):
                row["noise_multiplier"] = float(optimizer.noise_multiplier)

        rows.append(row)
        wandb.log(row)
        print(row)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_metrics = row.copy()
            model_to_save = model._module if hasattr(model, "_module") else model
            torch.save(model_to_save.state_dict(), output_dir / "best_model.pt")

    if rows:
        with open(output_dir / f"seed{args.seed}.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        final = rows[-1]
        run.summary["best_test_acc"] = best_test_acc
        run.summary["final_test_acc"] = final["test_acc"]
        run.summary["final_test_loss"] = final["test_loss"]
        if best_metrics is not None:
            run.summary["best_epoch"] = best_metrics["epoch"]

    result = {
        "seed": args.seed,
        "best_test_acc": best_test_acc,
        "best_metrics": best_metrics,
        "final_metrics": rows[-1] if rows else None,
    }
    (output_dir / f"seed{args.seed}_results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    wandb.finish()


def parse_args():
    parser = argparse.ArgumentParser("CIFAR10 DP classification with TCG")
    parser.add_argument("--data-root", type=str, default="./data/cifar10")
    parser.add_argument("--output-dir", type=str, default="./outputs/cifar10")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--batch-size-test", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--eval-epochs", type=int, default=1)
    parser.add_argument("--arch", choices=["cnn", "resnet18"], default="cnn")
    parser.add_argument("--base-opt", choices=["adam", "sgd"], default="sgd")
    parser.add_argument("--method", choices=["baseline", "tcg"], required=True)
    parser.add_argument("--lr-adam", type=float, default=1e-3)
    parser.add_argument("--lr-sgd", type=float, default=0.25)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lr-schedule", choices=["constant", "cos"], default="cos")
    parser.add_argument("--target-epsilon", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--max-per-sample-grad-norm", type=float, default=1.5)
    parser.add_argument("--opacus-accountant", choices=["prv", "rdp", "gdp"], default="rdp")
    parser.add_argument("--grad-sample-mode", choices=["hooks"], default="hooks")
    parser.add_argument("--opacus-clipping", choices=["flat", "per_layer", "adaptive"], default="flat")
    parser.add_argument("--poisson-sampling", dest="poisson_sampling", action="store_true", default=True)
    parser.add_argument("--no-poisson-sampling", dest="poisson_sampling", action="store_false")
    parser.add_argument("--secure-mode", action="store_true", default=False)
    parser.add_argument("--beta-fast", type=float, default=0.8)
    parser.add_argument("--beta-slow", type=float, default=0.98)
    parser.add_argument("--gate-type", choices=["hard", "soft"], default="hard")
    parser.add_argument("--gate-tau", type=float, default=0.0)
    parser.add_argument("--use-augmentation", action="store_true", default=False)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--disable-dp", action="store_true", default=False)
    parser.add_argument("--wandb-project", type=str, default="tcg-cifar10-classification")
    parser.add_argument("--wandb-entity", type=str, default="")
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    run_single_seed(parse_args())
