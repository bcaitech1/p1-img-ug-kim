import os
import random
import argparse
import re
import time
import datetime
from glob import glob
from importlib import import_module
from pathlib import Path
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import MaskBaseDataset
from loss import create_criterion
from adamp import AdamP


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def increment_path(path, exist_ok=False):
    """
    Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """

    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob(f"{path}")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}_{n}"


def get_current_time():
    utcnow = datetime.datetime.utcnow()
    time_gap = datetime.timedelta(hours=9)
    kor_time = utcnow + time_gap

    print(
        f"Start Training Time is {kor_time.month}/{kor_time.day} {kor_time.hour}:{kor_time.minute}:{kor_time.second}"
    )


def train(img_dir, model_dir, args):
    seed_everything(args.seed)

    start = time.time()
    get_current_time()

    save_dir = increment_path(os.path.join(model_dir, args.name))

    # settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset
    dataset_module = getattr(import_module("dataset"), args.dataset)
    dataset = dataset_module(
        img_dir=img_dir,
        val_ratio=args.val_ratio,
    )
    num_classes = dataset.num_classes

    transform_module = getattr(import_module("dataset"), args.augmentation)
    transform = transform_module(mean=dataset.mean, std=dataset.std)

    train_dataset, val_dataset = dataset.split_dataset()
    train_dataset.dataset.set_transform(transform["train"])
    val_dataset.dataset.set_transform(transform["val"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=2,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.valid_batch_size,
        num_workers=2,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    model_module = getattr(import_module("model"), args.model)
    model = model_module(num_classes=num_classes).to(device)

    model = torch.nn.DataParallel(model)

    criterion = create_criterion(args.criterion)

    optimizer = None
    if args.optimizer == "AdamP":
        optimizer = AdamP(model.parameters(), lr=args.lr)
    else:
        opt_module = getattr(import_module("torch.optim"), args.optimizer)
        optimizer = opt_module(
            model.parameters(),
            # filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            # weight_decay=5e-4,
        )

    # scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)

    logger = SummaryWriter(log_dir=save_dir)

    best_val_acc = 0
    best_val_loss = np.inf
    best_val_f1 = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        train_acc = 0
        train_f1 = 0
        for i, data in enumerate(tqdm(train_loader)):
            imgs, labels = data
            imgs = imgs.float().to(device)
            labels = labels.long().to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = torch.argmax(outputs, 1)
            acc = (preds == labels).sum().item() / len(imgs)
            t_f1_score = f1_score(
                labels.cpu().detach().numpy(),
                preds.cpu().detach().numpy(),
                average="macro",
            )

            train_loss += loss
            train_acc += acc
            train_f1 += t_f1_score

            if (i + 1) % args.log_interval == 0:
                train_loss /= args.log_interval
                train_acc /= args.log_interval
                train_f1 /= args.log_interval
                current_lr = get_lr(optimizer)
                print(
                    f"Epoch[{epoch + 1}/{args.epochs}]({i + 1}/{len(train_loader)}) || trainin_loss {train_loss:.4f} || training acc {train_acc:.4f} || train f1_score {train_f1:.4f} || lr {current_lr}"
                )

                logger.add_scalar(
                    "Train/loss", train_loss, epoch * len(train_loader) + i
                )
                logger.add_scalar(
                    "Train/accuracy", train_acc, epoch * len(train_loader) + i
                )
                logger.add_scalar(
                    "Train/F1-score", train_f1, epoch * len(train_loader) + i
                )

                train_loss = 0
                train_acc = 0
                train_f1 = 0

        # scheduler.step()

        # training은 1 epoch이 끝나야 완료된 것
        # 학습이 끝난 각 epoch에서 최고의 score를 가진 것을 저장하는 것
        with torch.no_grad():
            print("Validation step---------------------")
            model.eval()
            val_loss_items = []
            val_acc_items = []
            val_f1_items = []

            for data in tqdm(val_loader):
                imgs, labels = data
                imgs = imgs.float().to(device)
                labels = labels.long().to(device)

                outputs = model(imgs)
                preds = torch.argmax(outputs, 1)

                loss = criterion(outputs, labels).item()
                acc = (labels == preds).sum().item()
                val_f1 = f1_score(
                    labels.cpu().detach().numpy(),
                    preds.cpu().detach().numpy(),
                    average="macro",
                )

                val_loss_items.append(loss)
                val_acc_items.append(acc)
                val_f1_items.append(val_f1)

            val_loss = np.sum(val_loss_items) / len(val_loader)
            val_acc = np.sum(val_acc_items) / len(val_dataset)
            val_f1 = np.sum(val_f1_items) / len(val_loader)

            print(f"val_loader: {len(val_loader)} | val_dataset: {len(val_dataset)}")

            best_val_loss = min(best_val_loss, val_loss)
            best_val_f1 = max(val_f1, best_val_f1)
            best_val_acc = max(val_acc, best_val_acc)

            # if val_acc > best_val_acc:
            # print(
            #     f"New best model for val acc: {val_acc:4.2%}! saving the best model..."
            # )
            #     torch.save(model.module.state_dict(), f"{save_dir}/best.pth")
            #     best_val_acc = val_acc

            if val_f1 > best_val_f1:
                print(
                    f"New best model for val f1: {val_f1:.4f}! saving the best model..."
                )
                torch.save(model.module.state_dict(), f"{save_dir}/best.pth")
                best_val_f1 = val_f1

            # TODO: last model 저장이 여기 위치가 맞나 ??
            # torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
            print(
                f"[Val] acc: {val_acc:.4f}, loss: {val_loss:.4f} || best acc: {best_val_acc:.4f}, best loss: {best_val_loss:.4f}"
            )

            logger.add_scalar("Val/loss", val_loss, epoch)
            logger.add_scalar("Val/accuracy", val_acc, epoch)
            logger.add_scalar("Val/f1-score", val_f1, epoch)
            print()

    torch.save(model.module.state_dict(), f"{save_dir}/last.pth")

    # How much time training taken
    times = time.time() - start
    minute, sec = divmod(times, 60)
    print(f"Finish Training! Taken time is {minute} minutes {sec} seconds")


def train_no_val(img_dir, model_dir, args):
    seed_everything(args.seed)

    start = time.time()
    get_current_time()

    save_dir = increment_path(os.path.join(model_dir, args.name))

    # settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset
    dataset_module = getattr(import_module("dataset"), args.dataset)
    dataset = dataset_module(
        img_dir=img_dir,
        val_ratio=args.val_ratio,
    )
    num_classes = dataset.num_classes

    transform_module = getattr(import_module("dataset"), args.augmentation)
    transform = transform_module(mean=dataset.mean, std=dataset.std)

    dataset.set_transform(transform["train"])

    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=2,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    model_module = getattr(import_module("model"), args.model)
    model = model_module(num_classes=num_classes).to(device)

    model = torch.nn.DataParallel(model)

    criterion = create_criterion(args.criterion)

    optimizer = None
    if args.optimizer == "AdamP":
        optimizer = AdamP(model.parameters())
    else:
        opt_module = getattr(import_module("torch.optim"), args.optimizer)
        optimizer = opt_module(
            model.parameters(),
            # filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            # weight_decay=5e-4,
        )

    # scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)

    logger = SummaryWriter(log_dir=save_dir)

    best_val_acc = 0
    best_val_loss = np.inf
    best_val_f1 = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        train_acc = 0
        train_f1 = 0
        for i, data in enumerate(tqdm(train_loader)):
            imgs, labels = data
            imgs = imgs.float().to(device)
            labels = labels.long().to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = torch.argmax(outputs, 1)
            acc = (preds == labels).sum().item() / len(imgs)
            t_f1_score = f1_score(
                labels.cpu().detach().numpy(),
                preds.cpu().detach().numpy(),
                average="macro",
            )

            train_loss += loss
            train_acc += acc
            train_f1 += t_f1_score

            if (i + 1) % args.log_interval == 0:
                train_loss /= args.log_interval
                train_acc /= args.log_interval
                train_f1 /= args.log_interval
                current_lr = get_lr(optimizer)
                print(
                    f"Epoch[{epoch + 1}/{args.epochs}]({i + 1}/{len(train_loader)}) || trainin_loss {train_loss:.4f} || training acc {train_acc:.4f} || train f1_score {train_f1:.4f} || lr {current_lr}"
                )

                logger.add_scalar(
                    "Train/loss", train_loss, epoch * len(train_loader) + i
                )
                logger.add_scalar(
                    "Train/accuracy", train_acc, epoch * len(train_loader) + i
                )
                logger.add_scalar(
                    "Train/F1-score", train_f1, epoch * len(train_loader) + i
                )

                train_loss = 0
                train_acc = 0
                train_f1 = 0

    torch.save(model.module.state_dict(), f"{save_dir}/last.pth")

    # How much time training taken
    times = time.time() - start
    minute, sec = divmod(times, 60)
    print(f"Finish Training! Taken time is {minute} minutes {sec} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed", type=int, default=42, help="random seed (default: 42)"
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="number of epochs to train (default: 1)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="MaskBaseDataset",
        help="dataset augmentation type (default: MaskBaseDataset)",
    )
    parser.add_argument(
        "--augmentation",
        type=str,
        default="get_transforms",
        help="data augmentation type (default: get_tranform)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="input batch size for training (default: 32)",
    )
    parser.add_argument(
        "--valid_batch_size",
        type=int,
        default=64,
        help="input batch size for validating (default: 64)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Efficientnet_b0",
        help="model type (default: efficientnet_b0)",
    )
    parser.add_argument(
        "--optimizer", type=str, default="Adam", help="optimizer type (default: Adam)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="learning rate (defualt: 1e-4)"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="ratio for validation (default: 0.2)",
    )
    parser.add_argument(
        "--criterion",
        type=str,
        default="cross_entropy",
        help="criterion type (default: cross_entropy)",
    )
    parser.add_argument(
        "--lr_decay_step",
        type=int,
        default=20,
        help="learning rate scheduler decay step (default: 20)",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=20,
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--name", type=str, default="exp", help="model save at {SM_MODEL_DIR}/{name}"
    )
    parser.add_argument(
        "--img_dir",
        type=str,
        default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train/images"),
    )
    parser.add_argument(
        "--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR", "./model")
    )

    # TODO: SM_MODEL_DIR이 뭐지?? -> os.environ.get으로 환경변수 설정

    args = parser.parse_args()
    print(args)

    img_dir = args.img_dir
    model_dir = args.model_dir

    # train(img_dir, model_dir, args)
    train_no_val(img_dir, model_dir, args)
