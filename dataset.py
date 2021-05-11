import os
from glob import glob

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

from albumentations import *
from albumentations.pytorch import ToTensorV2

# 테스트 데이터셋 폴더 경로
test_dir = "/opt/ml/input/data/eval"
log_dir = "/opt/ml/code/results"


class Cfg:
    data_dir = "../input/data/train"
    img_dir = f"{data_dir}/images"
    df_path = f"{data_dir}/train.csv"


df = pd.read_csv(os.path.join(Cfg.df_path))
num2class = ["incorrect_mask", "mask1", "mask2", "mask3", "mask4", "mask5", "normal"]
class2num = {k: v for v, k in enumerate(num2class)}


def get_extension(img_dir, img_id, class_id):
    """
    img_dir = Cfg.img_dir
    img_id = "000001_female_Asian_45" (example)
    class_id = mask1
    """
    filename = glob(os.path.join(img_dir, img_id, "*"))
    for f in filename:
        _class_id = f.split("/")[-1].split(".")[0]
        if class_id == _class_id:
            ext = os.path.splitext(f)[-1].lower()
            break
    return ext


def get_img_stats(img_dir, img_ids):
    img_info = dict(heights=[], widths=[], means=[], stds=[])
    for img_id in tqdm(img_ids):
        for path in glob(os.path.join(img_dir, img_id, "*")):
            img = np.array(Image.open(path))
            h, w, _ = img.shape
            img_info["heights"].append(h)
            img_info["widths"].append(w)
            img_info["means"].append(img.mean(axis=(0, 1)))
            img_info["stds"].append(img.std(axis=(0, 1)))
    return img_info


# 시간 너무 오래 걸려서 주석 처리
# img_info = get_img_stats(Cfg.img_dir, df.path.values)

# mean = np.mean(img_info["means"], axis=0) / 255.
# std = np.mean(img_info["stds"], axis=0) / 255.

# mean, std
# (array([0.56019358, 0.52410121, 0.501457  ]),
#  array([0.23318603, 0.24300033, 0.24567522]))

mean = np.array([0.56019358, 0.52410121, 0.501457])
std = np.array([0.23318603, 0.24300033, 0.24567522])


def get_transforms(
    need=("train", "val"),
    img_size=(384, 384),
    mean=(0.548, 0.504, 0.479),
    std=(0.237, 0.247, 0.246),
):
    """
    Args:
        need: 'train' 혹은 'val' 혹은 둘 다에 대한 augmentation 함수 얻을 건지에 대한 옵션
        img_size: augmentation 이후 얻을 이미지 사이즈
        mean: 이미지 normalize 할 때 사용될 RGB 평균값
        std: 이미지를 normalize 할 때 사용할 RGB 표준편차
    """

    transformations = {}
    if "train" in need:
        transformations["train"] = Compose(
            [
                CenterCrop(img_size[0], img_size[1], p=1.0),
                HorizontalFlip(p=0.5),
                ShiftScaleRotate(p=0.5),
                HueSaturationValue(
                    hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5
                ),
                RandomBrightnessContrast(
                    brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=-0.5
                ),
                GaussNoise(p=0.5),
                Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        )

    if "val" in need:
        transformations["val"] = Compose(
            [
                CenterCrop(img_size[0], img_size[1]),
                Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        )

    return transformations


def base_transforms(
    need=("train", "val"),
    img_size=(384, 384),
    mean=(0.548, 0.504, 0.479),
    std=(0.237, 0.247, 0.246),
):
    """
    Args:
        need: 'train' 혹은 'val' 혹은 둘 다에 대한 augmentation 함수 얻을 건지에 대한 옵션
        img_size: augmentation 이후 얻을 이미지 사이즈
        mean: 이미지 normalize 할 때 사용될 RGB 평균값
        std: 이미지를 normalize 할 때 사용할 RGB 표준편차
    """

    transformations = {}
    if "train" in need:
        transformations["train"] = Compose(
            [
                CenterCrop(img_size[0], img_size[1], p=1.0),
                HorizontalFlip(p=0.5),
                HueSaturationValue(
                    hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5
                ),
                RandomBrightnessContrast(
                    brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=-0.5
                ),
                Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        )

    if "val" in need:
        transformations["val"] = Compose(
            [
                CenterCrop(img_size[0], img_size[1]),
                Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        )

    return transformations


class MaskLabels:
    mask = 0
    incorrect = 1
    normal = 2


class GenderLabels:
    male = 0
    female = 1


class AgeGroup:
    map_label = lambda x: 0 if int(x) < 30 else 1 if int(x) < 60 else 2


class MaskBaseDataset(Dataset):
    num_classes = 3 * 2 * 3

    _file_names = {
        "mask1": MaskLabels.mask,
        "mask2": MaskLabels.mask,
        "mask3": MaskLabels.mask,
        "mask4": MaskLabels.mask,
        "mask5": MaskLabels.mask,
        "incorrect_mask": MaskLabels.incorrect,
        "normal": MaskLabels.normal,
    }

    image_paths = []
    mask_labels = []
    gender_labels = []
    age_labels = []

    def __init__(self, img_dir, transform=None, val_ratio=0.2):
        self.img_dir = img_dir
        self.mean = mean
        self.std = std
        self.transform = transform
        self.val_ratio = val_ratio

        self.setup()

    def set_transform(self, transform):
        self.transform = transform

    def setup(self):
        profiles = glob(os.path.join(self.img_dir, "*"))
        for profile in profiles:
            for file_name, label in self._file_names.items():
                ext = get_extension(self.img_dir, profile.split("/")[-1], file_name)
                img_path = os.path.join(profile, file_name + ext)
                if os.path.exists(img_path):
                    self.image_paths.append(img_path)
                    self.mask_labels.append(label)

                    _, gender, race, age = profile.split("_")
                    # getattr : object의 속성값
                    # GenderLabels.male 과 같음
                    gender_label = getattr(GenderLabels, gender)
                    age_label = AgeGroup.map_label(age)

                    self.gender_labels.append(gender_label)
                    self.age_labels.append(age_label)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path)

        mask_label = self.mask_labels[index]
        gender_label = self.gender_labels[index]
        age_label = self.age_labels[index]
        multi_class_label = mask_label * 6 + gender_label * 3 + age_label

        image_transform = self.transform(image=np.array(image))["image"]
        return image_transform, multi_class_label

    def __len__(self):
        return len(self.image_paths)

    def split_dataset(self):
        n_val = int(len(self) * self.val_ratio)
        n_train = len(self) - n_val
        train_set, val_set = torch.utils.data.random_split(self, [n_train, n_val])
        return train_set, val_set

    def denormalize_image(image, mean, std):
        img_cp = image.copy()
        img_cp *= std
        img_cp += mean
        img_cp *= 255.0
        img_cp = np.clip(img_cp, 0, 255).astype(np.uint8)
        return img_cp


class TestDataset(Dataset):
    def __init__(self, img_paths, transform=None):
        self.img_paths = img_paths
        self.transform = Compose(
            [
                CenterCrop(384, 384, p=1.0),
                Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        )

    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])
        image_transform = self.transform(image=np.array(image))["image"]
        return image_transform

    def __len__(self):
        return len(self.img_paths)