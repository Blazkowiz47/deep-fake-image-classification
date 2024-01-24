"""
Main training file.
calls the train pipeline with configs.
"""
import argparse

import os

# python train.py --config="grapher_12_conv_gelu_config" --wandb-run-name="grapher only"

parser = argparse.ArgumentParser(
    description="Training Config",
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
    "--epochs",
    default=30,
    type=int,
    help="Add number of epochs.",
)

parser.add_argument(
    "--batch-size",
    default=32,
    type=int,
    help="Add batch_size.",
)

parser.add_argument(
    "--wandb-run-name",
    type=str,
    default=None,
    help="Wandb run name",
)

parser.add_argument(
    "--validate-after-epochs",
    type=int,
    default=1,
    help="Validate after epochs.",
)

parser.add_argument(
    "--learning-rate",
    type=float,
    default=1e-4,
    help="Learning rate.",
)

parser.add_argument(
    "--continue-model",
    type=str,
    default=None,
    help="Give path to the model to continue learning.",
)

parser.add_argument(
    "--augment-times",
    type=int,
    default=4,
    help="Number of augmented images per image",
)

parser.add_argument(
    "--printer",
    type=str,
    default=None,
    help=" wil continue from Printer type: rico , digital or DNP",
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
    "--n-classes",
    type=int,
    default=2,
    help="Defines total classes to predict.",
)


parser.add_argument(
    "--num-heads",
    type=int,
    default=4,
    help="Defines total number of heads in attention.",
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

parser.add_argument(
    "--reverse",
    "-r",
    type=bool,
    default=False,
    help="Will train in reverse order of (printer, morph-type)",
)


def main():
    """
    Wrapper for the driver.
    """
    args = parser.parse_args()
    morph_types = [
        "Morphing_Diffusion_2024",
        "cvmi",
        "lma",
        "lmaubo",
        "mipgan1",
        "mipgan2",
        "mordiff",
        "pipe",
        "regen",
        "stylegan",
    ]
    if args.reverse:
        morph_types = reversed(morph_types)
    for morph_type in morph_types:
        wandb_run_name = args.config + "_" + args.printer + "_" + morph_type
        process = f"python train.py -c {args.config} --printer={args.printer} --morph-type={morph_type} --epochs={args.epochs} --batch-size={args.batch_size}  --grapher-units={args.grapher_units} --total-layers={args.total_layers} --validate-after-epochs={args.validate_after_epochs} --wandb-run-name={wandb_run_name}"
        os.system(process)


if __name__ == "__main__":
    main()
