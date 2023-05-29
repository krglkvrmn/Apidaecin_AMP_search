import argparse
from pathlib import Path

from src.io import read_fasta_as_dict, read_predictions, save_predictions_as_fasta
from src.logs import logger
from src.predictions import aggregate_score_by_genomic_seq, split_predictions_by_threshold, extract_nucl_predictions, \
    extract_fp_aa_predictions
from src.scoring import length_score, consecutive_score, fraction_score, calculate_score_threshold
from src.utils import check_path

score_methods = {
    "length_score": length_score,
    "consecutive_score": consecutive_score,
    "fraction_score": fraction_score,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--genome_file', type=Path, required=True,
                        help="Path to genomic fasta file")
    parser.add_argument('--prediction_file', type=Path, required=True,
                        help="Path to prediction file (result of scan_proteome.py)")
    parser.add_argument('--scoring_method', default="fraction_score",
                        type=str, choices=list(score_methods.keys()), help="Scoring method")
    parser.add_argument('--fp_patch_size', default=33, type=int,
                        help="Patch size which is used for selection of false positive fragments")
    parser.add_argument('--prediction_offset', nargs=2, default=(10000, 10000), type=int,
                        help="Number of nucleotides to offset from predicted region to left and right")
    parser.add_argument('--threshold', default=0.0, type=float,
                        help="Specify cutoff manually")
    parser.add_argument('--genes_output_file', required=True, type=Path,
                        help="Path to file where found potential gene regions are saved")
    parser.add_argument('--fp_output_file', type=Path,
                        help="Path to file where false positive predictions are saved")

    args = parser.parse_args()

    logger.info(f"Started filtering predictions {args.prediction_file} with scoring method '{args.scoring_method}'")

    scoring_func = score_methods[args.scoring_method]

    genome_map = read_fasta_as_dict(args.genome_file)
    predictions_data = read_predictions(args.prediction_file)
    if "score" not in predictions_data.columns:
        logger.critical(f"Predictions file {args.prediction_file} does not contain required column: score. "
                        "Be sure to run 'make_scores.py' script to create it")
    score_threshold = max(calculate_score_threshold(predictions_data["score"]), args.threshold)

    tp_predictions_data, fp_predictions_data = split_predictions_by_threshold(predictions_data,
                                                                              score_threshold=score_threshold)

    logger.info(f"Score threshold is {score_threshold}. "
                f"Selected {len(tp_predictions_data)}/{len(predictions_data)} fragments")
    print(tp_predictions_data.to_string(columns=["t_record_id", "score"], index=False))

    logger.info(f"Extracting potential gene regions from genome: {args.genome_file}")

    genomic_predictions = extract_nucl_predictions(
        tp_predictions_data,
        genome_map=genome_map, skip_non_positive=True,
        trimming_offset=args.prediction_offset
    )
    fp_predictions = extract_fp_aa_predictions(fp_predictions_data, kernel_size=args.fp_patch_size)

    if args.genes_output_file.is_dir():
        output_file = args.genes_output_file / args.genome_file.name
    else:
        output_file = args.genes_output_file
        open(output_file, "w").close()

    output_file = check_path(args.genes_output_file, force_overwrite=True)
    fp_output_file = check_path(args.fp_output_file, force_overwrite=True)

    if len(genomic_predictions) > 0:
        save_predictions_as_fasta(genomic_predictions, output_file)
        logger.success(f"All potential gene regions saved to {output_file}")
    else:
        logger.success(f"No potential gene regions were found")

    if len(fp_predictions) > 0 and args.fp_output_file is not None:
        save_predictions_as_fasta(fp_predictions, fp_output_file)
        logger.success(f"All false positive regions saved to {fp_output_file}")
    else:
        logger.success(f"No false positive sequences were found")

