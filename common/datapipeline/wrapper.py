from typing import Any, List
import os

import cv2
import numpy as np
import albumentations as A
from common.datapipeline.dataset import DatasetItem, get_dataloader


class DatasetWrapper:
    def __init__(
        self,
        root_dir: str,
        morph_type: str,
        height: int = 224,
        width: int = 224,
        classes: int = 2,
    ) -> None:
        self.root_dir = root_dir
        self.morph_type = morph_type
        self.height = height
        self.width = width
        self.classes = classes

    def loop_through_dir(
        self,
        dir: str,
        label: int,
        augment_times: int = 0,
    ) -> List[DatasetItem]:
        items: List[DatasetItem] = []
        for image in os.listdir(dir):
            if image.lower().endswith("jpg") or image.lower().endswith("png"):
                path = os.path.join(dir, image)
                items.append(DatasetItem(path, False, label))
                for _ in range(augment_times):
                    items.append(DatasetItem(path, True, label))
        return items

    def transform(self, data: DatasetItem) -> Any:
        image = cv2.imread(data.path, cv2.IMREAD_COLOR)  # pylint: disable=E1101
        if image is None:
            print(data.path)
        if image.shape[0] > image.shape[1]:
            image = image.transpose()  # pylint: disable=E1101
        image = cv2.resize(image, (self.width, self.height))  # pylint: disable=E1101
        image = (image - image.min()) / ((image.max() - image.min()) or 1.0)
        image = image.astype("float")
        image = np.expand_dims(image, axis=0)
        # image = np.vstack([image, image, image])

        label = np.zeros((self.classes))
        label[data.label] = 1  # One hot encoding
        # return image, label
        return image.astype(np.float32), label

    def augment(self, image, label) -> Any:
        transform = A.Compose(
            [
                A.HorizontalFlip(p=0.25),
                A.VerticalFlip(p=0.25),
                A.RandomBrightnessContrast(p=0.2),
                A.InvertImg(p=0.05),
                A.PixelDropout(p=0.02),
            ],
        )
        transformed = transform(image=image)
        transformed_image = transformed["image"]
        return transformed_image, label

    def get_train_dataset(self, augment_times: int, batch_size: int):
        data: List[DatasetItem] = []
        data.extend(
            self.loop_through_dir(
                os.path.join(
                    self.root_dir,
                    "bonafide",
                    "train",
                ),
                0,
                augment_times * 2,
            )
        )
        data.extend(
            self.loop_through_dir(
                os.path.join(
                    self.root_dir,
                    "morph",
                    self.morph_type,
                    "train",
                ),
                1,
                augment_times,
            )
        )
        return get_dataloader(data, self.transform, self.augment, batch_size)

    def get_test_dataset(self, augment_times: int, batch_size: int):
        data: List[DatasetItem] = []
        data.extend(
            self.loop_through_dir(
                os.path.join(
                    self.root_dir,
                    "bonafide",
                    "test",
                ),
                0,
                augment_times * 2,
            )
        )
        data.extend(
            self.loop_through_dir(
                os.path.join(
                    self.root_dir,
                    "morph",
                    self.morph_type,
                    "test",
                ),
                1,
                augment_times,
            )
        )
        return get_dataloader(data, self.transform, self.augment, batch_size)
