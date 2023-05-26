import argparse
from pathlib import Path

from src.io import read_fasta_as_dict, read_predictions
from src.logs import logger
from src.predictions import aggregate_score_by_genomic_seq
from src.scoring import length_score, consecutive_score, fraction_score
from src.utils import check_path

score_methods = {
    "length_score": length_score,
    "consecutive_score": consecutive_score,
    "fraction_score": fraction_score,
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--prediction_file', type=Path, required=True,
                        help="Path to prediction file (result of scan_proteome.py)")
    parser.add_argument('--scoring_method', default="length_score",
                        type=str, choices=list(score_methods.keys()), help="Scoring method")
    parser.add_argument('--output_file', type=Path, required=True,
                        help="Path to file where scored predictions are saved")

    args = parser.parse_args()

    logger.info(f"Started filtering predictions {args.prediction_file} with scoring method '{args.scoring_method}'")

    scoring_func = score_methods[args.scoring_method]

    predictions_data = read_predictions(args.prediction_file)
    scores = predictions_data.apply(lambda row: scoring_func(row.sequence, row.prediction_mask), axis=1)
    predictions_data = aggregate_score_by_genomic_seq(predictions_data.assign(score=scores))

    save_path = check_path(args.output_file)
    predictions_data.to_csv(save_path, sep="\t", index=False)
    logger.success(f"Predictions with scores successfully saved to '{save_path}'")
