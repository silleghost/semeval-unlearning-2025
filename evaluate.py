import logging
import sys
from pathlib import Path

from evaluation.evaluate_generations import main as run_evaluation

log_dir = Path("evaluation")
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename="evaluation/evaluation.log",
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    filemode="a",
    force=True,
)

logger = logging.getLogger(__name__)


def evaluate_single_model(
    model_path,
    evaluation_data_path,
    output_dir="./results",
    mia_data_path=None,
    mmlu_metrics_file_path=None,
    batch_size=32,
    max_new_tokens=256,
):
    """
    Evaluate a single merged model

    Parameters:
    - model_path: Path to the merged model
    - evaluation_data_path: Path to evaluation data
    - output_dir: Where to save results
    - mia_data_path: Optional path to MIA data
    - mmlu_metrics_file_path: Optional path to MMLU metrics
    """
    try:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting evaluation of model at: {model_path}")

        sys.argv = [
            "evaluate_generations.py",
            "--data_path",
            evaluation_data_path,
            "--checkpoint_path",
            model_path,
            "--output_dir",
            output_dir,
            "--batch_size",
            str(batch_size),
            "--max_new_tokens",
            str(max_new_tokens),
            "--debug",
        ]

        if mia_data_path:
            sys.argv.extend(["--mia_data_path", mia_data_path])
        if mmlu_metrics_file_path:
            sys.argv.extend(["--mmlu_metrics_file_path", mmlu_metrics_file_path])

        logger.info("Running evaluation...")
        run_evaluation()

        logger.info(
            f"Evaluation complete! Results saved to: {output_dir}/evaluation_results.jsonl"
        )

    except Exception as e:
        logger.exception(f"Error during evaluation: {str(e)}")
        raise


if __name__ == "__main__":
    evaluate_single_model(
        model_path="models/trained/olmo-model-final/",
        evaluation_data_path="evaluation/validation/",
        mia_data_path="evaluation/mia_data/",
        mmlu_metrics_file_path="evaluation_results/metrics.json",
        output_dir="./evaluation_results",
    )
