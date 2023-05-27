import argparse
from datetime import datetime
from pathlib import Path

from src.core import Controller
from src.io import read_fasta_as_dict, save_predictions
from src.logs import logger
from src.scan import scan_records, ScanScheduler
from src.utils import check_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_path", required=True, type=Path,
                        help="Path to directory with saved models (must contain `weights` and `params` subdirectories)")
    parser.add_argument("--model_name", required=True,
                        help="Model name to load. `HybridModel` and `SimpleCNN` are available. "
                             "E.g. SimpleCNN_v2 or HybridModel")
    parser.add_argument("--proteome_path", required=True, type=Path, help="Path to fasta file with proteome")
    parser.add_argument("--scan_stride", required=False, default=20, type=int,
                        help="Step to scan proteome. Greater -> faster, less accurate, lower -> slower, more accurate. "
                             "Default: 20")
    parser.add_argument("--save_path", required=True, type=Path,
                        help="File to save results into. Output is in tsv format with fields: "
                             "record_id, record_description, pos_count, sequence, prediction_mask, model_name")

    args = parser.parse_args()

    predictor = Controller.make_predictor(args.models_path, args.model_name)

    if args.proteome_path.exists():
        proteome_records = read_fasta_as_dict(args.proteome_path)
        logger.info(f"Loaded proteome file {args.proteome_path} with {len(proteome_records)} sequences")
    else:
        logger.error(f"Proteome file {args.proteome_path} not found")
        raise FileNotFoundError(args.proteome_path)

    start_time = datetime.now()
    # scheduler = ScanScheduler(controller=predictor, patches_limit=45000, scan_stride=args.scan_stride)
    # predictions = scheduler.run(proteome_records.values(), logging_interval=1000)
    predictions = scan_records(predictor, records=proteome_records.values(),
                               stride=args.scan_stride, logging_interval=1000)
    total_time = datetime.now() - start_time
    logger.success(f"Finished scanning proteome. Total time: {total_time}")

    # Print info about proteins with at least 1 prediction
    pos_only_predictions = predictions.query("pos_count > 0")
    for _, row in pos_only_predictions.iterrows():
        print(row.record_id, f"{row.pos_count}/{len(row.sequence)}", sep="\t")

    pos_only_predictions["model_name"] = [args.model_name] * len(pos_only_predictions)

    save_path = check_path(args.save_path)
    save_predictions(pos_only_predictions, save_path)
    logger.success(f"Predictions successfully saved to {save_path}")
