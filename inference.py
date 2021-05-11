import os
import argparse
from importlib import import_module
from tqdm import tqdm

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import TestDataset


def load_model(saved_model, num_classes, device):
    model_cls = getattr(import_module("model"), args.model)
    model = model_cls(num_classes=num_classes)

    model_path = os.path.join(saved_model, "last.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


# TODO: @ 데코레이터로 이렇게 하면 어찌 되는거지??
# Also functions as a decorator. (Make sure to instantiate with parenthesis.)
@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = 18

    info_path = os.path.join(data_dir, "info.csv")
    submission = pd.read_csv(info_path)
    img_dir = os.path.join(data_dir, "images")
    img_paths = [os.path.join(img_dir, img_id) for img_id in submission.ImageID]

    test_dataset = TestDataset(img_paths)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=8,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    model = load_model(model_dir, num_classes, device).to(device)
    model.eval()

    all_predictions = []
    with torch.no_grad():
        for imgs in tqdm(test_loader):
            imgs = imgs.float().to(device)
            pred = model(imgs)
            pred = pred.argmax(dim=-1)
            all_predictions.extend(pred.cpu().numpy())

    submission["ans"] = all_predictions

    # 제출할 파일을 저장합니다.
    submission.to_csv(
        os.path.join(output_dir, f"submission_{args.name}.csv"), index=False
    )
    print("Test inference is done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="input batch size for testing (default: 64)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Efficientnet_b0",
        help="model type (default: efficientnet_b0)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.environ.get("SM_CHANNEL_EVAL", "/opt/ml/input/data/eval"),
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=os.environ.get("SM_CHANNEL_MODEL", "./model"),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.environ.get("SM_OUTPUT_DATA_DIR", "./output"),
    )
    parser.add_argument("--name", type=str, default="test", help="submission file name")

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    inference(data_dir, model_dir, output_dir, args)