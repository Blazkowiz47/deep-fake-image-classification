"""
Main evaluation file.
Evaluates the pretrained model based on a particular dataset.
"""
import argparse

import json
from torch.nn import Module
from timm.loss import SoftTargetCrossEntropy
from common.metrics.eer import EER
from common.trainpipeline.train import cuda_info
from common.util.logger import logger
from train import get_config
from typing import Any
import os
import numpy as np
import torch
from torch.optim import lr_scheduler as lr_scheduler
from tqdm import tqdm
from torchmetrics import Metric
from torchmetrics.classification import Accuracy

from common.datapipeline.wrapper import DatasetWrapper
from common.trainpipeline.model.model import get_model
import matlab
import matlab.engine

parser = argparse.ArgumentParser(
    description="Evaluation Config",
    add_help=True,
)

parser.add_argument(
    "-c",
    "--config",
    default="vig_pyramid_compact",
    type=str,
    help="Put model config name from common_config",
)

parser.add_argument(
    "--batch-size",
    default=32,
    type=int,
    help="Add batch_size.",
)

parser.add_argument(
    "--morph-type",
    type=str,
    default="cvmi",
    help="Morph Type",
)

parser.add_argument(
    "--printer",
    type=str,
    default="rico",
    help="Printer type: rico , digital or DNP",
)

parser.add_argument(
    "--root-dir",
    type=str,
    default="/home/ubuntu/cluster/nbl-users/Shreyas-Sushrut-Raghu/PRINT_SCAN/",
    help="Dataset name.",
)

parser.add_argument(
    "--act",
    type=str,
    default="gelu",
    help="Defines activation function.",
)

parser.add_argument(
    "--pred-type",
    type=str,
    default="conv",
    help="Defines predictor type.",
)

parser.add_argument(
    "--height",
    type=int,
    default=224,
    help="Defines height of the image.",
)

parser.add_argument(
    "--width",
    type=int,
    default=224,
    help="Defines width of the image.",
)

parser.add_argument(
    "-f",
    "--model-path",
    type=str,
    default=None,
    help="Defines pretrained model's path.",
)

parser.add_argument(
    "--total-layers",
    type=int,
    default=5,
    help="TOTAL LAYERS FOR DSC STEM",
)


parser.add_argument(
    "--grapher-units",
    type=str,
    default="2,1,6,2",
    help="Number of grapher units",
)


def get_loss() -> Module:
    """
    Gets a loss function for training.
    """
    return SoftTargetCrossEntropy()


def get_metrics(n_classes: int, eng: Any) -> list[Metric]:
    """
    Returns list of testing metrics.
    """
    return [
        Accuracy(
            task="multiclass",
            num_classes=n_classes,
        ),
        EER(eng, genuine_class_label=0),
        # ConfusionMatrix().to(device),
    ]


def main():
    args = parser.parse_args()
    batch_size = args.batch_size
    logger.info("BATCHSIZE: %s", batch_size)
    config = get_config(
        args.config,
        args.act,
        args.pred_type,
        args.n_classes,
        args.num_heads,
        args.height,
        args.width,
        args.total_layers,
        args.grapher_units,
    )

    try:
        eng = matlab.engine.start_matlab()
        script_dir = "/home/ubuntu/finger-vein-quality-assessement/EER"
        eng.addpath(script_dir)
    except Exception:
        eng = None
        logger.exception("Cannot initialise matlab engine")

    # set device
    device = cuda_info()
    # Load the model
    model = get_model(config)
    model.load_state_dict(torch.load(args.model_path))
    model = model.to(device)
    logger.info(model)
    logger.info("Total parameters: %s", sum(p.numel() for p in model.parameters()))
    logger.info(
        "Total trainable parameters: %s",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
    # Set model to evaluation mode
    model.eval()
    # Load the dataset
    wrapper = DatasetWrapper(
        os.path.join(args.root_dir, args.printer),
        args.morph_type,
        args.height,
        args.width,
    )

    train_dataset = wrapper.get_train_dataset(0, batch_size)
    test_dataset = wrapper.get_test_dataset(0, batch_size)

    loss_fn = get_loss().to(device)

    metrics = [metric.to(device) for metric in get_metrics(2, eng)]
    results = {}
    for dataset, dataset_name in zip(
        [train_dataset, test_dataset], ["Train Set", "Test Set"]
    ):
        losses = []
        for inputs, labels in tqdm(dataset, desc=f"Evaluating {dataset_name}: "):
            inputs = inputs.to(device).float()
            labels = labels.to(device).float()
            outputs = model(inputs)  # pylint: disable=E1102
            loss = loss_fn(outputs, labels)  # pylint: disable=E1102
            losses.append(loss.item())

            metrics[1].update(outputs, labels)

            predicted = outputs.argmax(dim=1)
            labels = labels.argmax(dim=1)
            metrics[0].update(predicted, labels)
            eer, one, pointone, pointzeroone = metrics[1].compute()
            results[dataset_name] = (
                {
                    "accuracy": metrics[0].compute().item(),
                    "loss": np.mean(losses),
                    "eer": eer,
                    "tar1": one,
                    "tar0.1": pointone,
                    "tar0.01": pointzeroone,
                },
            )

        for metric in metrics:
            metric.reset()
    model_name = args.model_path.removeprefix("best_eer_test").split(".")[0]

    with open(
        f"results/{model_name}_{args.printer}_{args.morph_type}.json", "w+"
    ) as fp:
        json.dump(
            results,
            fp,
        )


if __name__ == "__main__":
    main()
