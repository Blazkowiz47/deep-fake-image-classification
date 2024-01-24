"""
Evaluation loop
Evaluates the pretrained model based on a particular dataset.
"""
import argparse
import os
from typing import List


parser = argparse.ArgumentParser(
    description="Evaluation Config",
    add_help=True,
)

parser.add_argument(
    "-c",
    "--config",
    default="test_dsc_custom",
    type=str,
    help="Put model config name from common_config",
)

parser.add_argument(
    "-o",
    "--save-results",
    default=None,
    type=str,
    help="Path to save results",
)

parser.add_argument(
    "--batch-size",
    default=128,
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
    "--num-heads",
    type=int,
    default=5,
    help="Total heads",
)

parser.add_argument(
    "--grapher-units",
    type=str,
    default="2,1,6,2",
    help="Number of grapher units",
)


def evaluate(
    printer,
    morph_type,
    mprinter,
    mmorph_type,
    config,
    batch_size,
    total_layers,
    num_heads,
    grapher_units,
    height,
    width,
    pred_type,
    act,
):
    process: str = f"python evaluate.py -c {config} --morph-type={morph_type} --printer={printer} -f models/checkpoints/best_eer_{config}_{mprinter}_{mmorph_type}.pt -o results/eval_{mprinter}_{mmorph_type}_on_{printer}_{morph_type}.json --height={height} --width={width} --grapher-units={grapher_units} --num-heads={num_heads} --total-layers={total_layers} --act={act} --pred-type={pred_type} --batch-size={batch_size}"
    os.system(process)


morph_types: List[str] = [
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
printers: List[str] = ["DNP", "rico", "Digital"]


def main():
    args = parser.parse_args()
    for mprinter in printers:
        for mmorph in morph_types:
            print(
                f"Evaluation models trained on printer: {mprinter} and morph type: {mmorph}"
            )
            for printer in printers:
                for morph in morph_types:
                    evaluate(
                        printer,
                        morph,
                        mprinter,
                        mmorph,
                        args.config,
                        args.batch_size,
                        args.total_layers,
                        args.num_heads,
                        args.grapher_units,
                        args.height,
                        args.width,
                        args.pred_type,
                        args.act,
                    )


if __name__ == "__main__":
    main()
