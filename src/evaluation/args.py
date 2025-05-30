import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from core.globals import DATASETS_DIR


@dataclass
class CMDArgs:
    dataset_dir: Path
    eval_details: Optional[str]


def parse_arguments() -> CMDArgs:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run evaluation on PDF files')
    parser.add_argument('--dataset', '-dat', type=str, default=None,
                        help='Path to the dataset directory containing PDF files for evaluation')
    parser.add_argument('--eval_details', '-det', type=str, default=None,
                        help='Description of the evaluation run')

    args = parser.parse_args()
    if args.dataset is None:
        raise Exception("Please provide a dataset path")

    dataset_dir = Path(args.dataset)
    if not dataset_dir.exists():
        dataset_dir = DATASETS_DIR / args.dataset

    if not dataset_dir.exists():
        raise Exception(f"dataset_dir {dataset_dir} does not exist")

    return CMDArgs(
        dataset_dir=dataset_dir,
        eval_details=args.eval_details
    )
