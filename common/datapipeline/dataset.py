"""

Dataset generator for pytorch
"""
from dataclasses import dataclass
from typing import Any, List

from torch.utils.data import DataLoader


@dataclass
class DatasetItem:
    path: str
    to_augment: bool
    label: Any


def get_dataloader(
    data: List[DatasetItem],
    transform,
    augment,
    batch_size,
    shuffle: bool = True,
    num_workers: int = 2,
) -> DataLoader:
    """
    Creates Dataset loader.
    """
    dataset_generator = DatasetGenerator(data, transform, augment)
    return DataLoader(
        dataset_generator,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


class DatasetGenerator:
    """A generator which gives the dataset sequentially"""

    def __init__(
        self,
        data: List[DatasetItem],
        transform: Any,
        augment: Any,
    ) -> None:
        self.transform = transform
        self.augment = augment
        self.data: List[DatasetItem] = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        x, y = self.transform(self.data[idx])
        if self.data[idx].to_augment:
            x, y = self.augment(x, y)
        return (x, y)
        # return (*[split[idx] for split in self.data],)
