"""Train L2 two-branch VA predictors for each modality.

Manifest format (JSONL, one sample per line)::

    {"feature_path": "path/to/feature.npy", "v_self": 0.2, "a_self": -0.5, "c_self": 0.8,
     "v_inter": 0.1, "a_inter": -0.3, "c_inter": 0.9}

Use ``--mock`` for synthetic data when D2 annotations are unavailable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.layer2_predict.trainer import train_modality
from src.layer2_predict.training_data import supported_modalities
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

MANIFEST_EPILOG = """
Manifest JSONL fields:
  feature_path  Path to a 1-D .npy feature vector
  v_self        Self-branch valence in [-1, 1]
  a_self        Self-branch arousal in [-1, 1]
  c_self        Self-branch confidence in [0, 1]
  v_inter       Inter-branch valence in [-1, 1]
  a_inter       Inter-branch arousal in [-1, 1]
  c_inter       Inter-branch confidence in [0, 1]
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train L2 TwoBranchMLP models with multitask self/inter losses.",
        epilog=MANIFEST_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--modality",
        required=True,
        choices=supported_modalities(),
        help="Modality to train",
    )
    parser.add_argument("--mock", action="store_true", help="Use synthetic training data")
    parser.add_argument("--data", type=str, default=None, help="Path to JSONL manifest")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--val-split", type=float, default=None)
    parser.add_argument("--mock-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Checkpoint directory (default: models/l2/{modality})",
    )
    parser.add_argument("--resume", type=str, default=None, help="Resume from last.pt")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.mock and args.data is None:
        parser.error("Either --mock or --data must be provided")

    models_config = load_config("models")
    pipeline_config = load_config("pipeline")

    result = train_modality(
        args.modality,
        mock=args.mock,
        manifest_path=args.data,
        models_config=models_config,
        pipeline_config=pipeline_config,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        val_split=args.val_split,
        mock_samples=args.mock_samples,
        seed=args.seed,
        device=args.device,
        resume_path=args.resume,
    )

    logger.info(
        "Training complete for '%s'. best=%s best_mae_va=%.4f metrics=%s",
        args.modality,
        result.best_checkpoint,
        result.best_mae_va,
        result.metrics_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
