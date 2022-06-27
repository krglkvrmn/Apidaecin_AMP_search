import argparse
import os
from datetime import datetime

from Bio import SeqIO
from loguru import logger

from src.training import Trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_path", required=True,
                        help="Path to directory with saved models (must contain `weights` and `params` subdirectories)")
    parser.add_argument("--model_name", required=True,
                        help="Model name to load. `HybridModel` and `SimpleCNN` are available. E.g. SimpleCNN_v2 or HybridModel")
    parser.add_argument("--proteome_path", required=True, help="Path to fasta file with proteome")
    parser.add_argument("--scan_stride", required=False, default=20, type=int,
                        help="Step to scan proteome. Greater -> faster, less accurate, lower -> slower, more accurate. Default: 20")
    parser.add_argument("--save_path", required=True,
                        help="File to save results into. Output is in tsv format with fields: record_id, record_description, sequence, prediction_mask")

    args = parser.parse_args()

    predictor = Trainer.make_predictor(args.models_path, args.model_name)

    if os.path.exists(args.proteome_path):
        # Load proteome into dict
        proteome_records = {record.id: record for record in SeqIO.parse(args.proteome_path, "fasta")}
        logger.info(f"Loaded proteome file {args.proteome_path} with {len(proteome_records)} sequences")
    else:
        logger.error(f"Proteome file {args.proteome_path} not found")
        raise FileNotFoundError

    start_time = datetime.now()
    predictions_by_id = {}
    for idx, (recid, record) in enumerate(proteome_records.items()):
        predicted_mask = predictor.predict_mask(str(record.seq), stride=args.scan_stride)
        mask_sum = predicted_mask.sum()
        if mask_sum:
            # Refine region of interest
            predicted_mask = predictor.predict_mask(str(record.seq), stride=1).tolist()
            fraction = sum(predicted_mask) / len(record) * 100
            logger.info(f"{recid} {record.description}: {sum(predicted_mask)}/{len(record)}, {fraction:.2f}%")
        else:
            predicted_mask = [0] * len(record)
        predictions_by_id[record.id] = predicted_mask
        if (idx + 1) % 500 == 0:
            logger.info(f"Processed {idx+1}/{len(proteome_records)} sequences")

    total_time = datetime.now() - start_time
    logger.info(f"Finished scanning proteome. Total time: {total_time}")

    # Print info about proteins with at least 1 prediction
    pos_only_predictions = list(filter(lambda x: sum(x[1]) > 0, predictions_by_id.items()))
    sorted_pos_only_predictions = sorted(pos_only_predictions, key=lambda it: -sum(it[1]) / len(it[1]))
    for recid, mask in sorted_pos_only_predictions:
        print(recid, f"{sum(mask)}/{len(mask)}", f"{sum(mask) / len(mask) * 100:.2f}%", sep="\t")

    logger.info(f"Saving results to {args.save_path}")

    with open(args.save_path, "w") as file:
        print("record_id", "record_description", "sequence", "prediction_mask", sep="\t", file=file)
        for recid, mask in sorted_pos_only_predictions:
            record = proteome_records[recid]
            print(recid, record.description, record.seq, "".join(map(str, mask)), sep="\t", file=file)
