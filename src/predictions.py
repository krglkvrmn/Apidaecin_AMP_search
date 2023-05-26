from functools import wraps
from functools import wraps
from typing import Optional

import numpy as np
import pandas as pd
from Bio.SeqRecord import SeqRecord

from src.utils import expand_mask_hits, recover_nucleotide_mask, get_mask_trim_coords, \
    stringify_mask, get_genome_fragment_by_translated_id


def require_columns(columns):
    def decorator(func):
        @wraps(func)
        def wrapper(predictions_data: pd.DataFrame, *args, **kwargs):
            if set(columns) <= set(predictions_data.columns):
                return func(predictions_data, *args, **kwargs)
            raise ValueError(f"Function {func} requires data to contain 'score' column")
        return wrapper
    return decorator


@require_columns({"g_record_id", "score"})
def aggregate_score_by_genomic_seq(predictions_data: pd.DataFrame) -> pd.DataFrame:
    scores_by_genomic_seq = predictions_data.groupby("g_record_id", as_index=False) \
                                            .agg({"score": "sum"})
    predictions_data = predictions_data.drop(columns="score") \
                                       .merge(scores_by_genomic_seq, on="g_record_id")
    return predictions_data.sort_values("score", ascending=False)


@require_columns({"score"})
def split_predictions_by_threshold(predictions_data: pd.DataFrame,
                                   score_threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    true_positive_predictions = predictions_data[predictions_data["score"] > score_threshold]
    false_positive_predictions = predictions_data[predictions_data["score"] <= score_threshold]
    return true_positive_predictions, false_positive_predictions


@require_columns({"sequence", "prediction_mask"})
def extract_fp_aa_predictions(fp_predictions_data: pd.DataFrame, kernel_size: int = 1) -> pd.DataFrame:
    def trim_fp_seq_mask(row: pd.Series) -> tuple[str, str]:
        fp_mask = expand_mask_hits(row.prediction_mask, kernel_size=kernel_size)
        fp_sequence = "".join(np.array(list(row.sequence))[fp_mask])
        fp_mask = "".join(np.array(list(row.prediction_mask))[fp_mask])
        return fp_sequence, fp_mask

    fp_predictions_data.loc[:, ["sequence", "prediction_mask"]] = fp_predictions_data.apply(
        trim_fp_seq_mask, axis=1, result_type="expand"
    ).values
    return fp_predictions_data


@require_columns({"g_record_id", "prediction_mask", "frame"})
def extract_nucl_predictions(
        predictions_data: pd.DataFrame,
        genome_map: dict[str, SeqRecord],
        skip_non_positive: bool = True,
        trimming_offset: tuple[Optional[int], Optional[int]] | np.ndarray = (None, None)
) -> pd.DataFrame:

    nucleotide_predictions_data = []
    for rec_id, genomic_record_pred_data in predictions_data.groupby("g_record_id"):
        genomic_record = get_genome_fragment_by_translated_id(str(rec_id), genome_map)

        translated_masks_by_frames = genomic_record_pred_data.set_index("frame")["prediction_mask"].to_dict()
        nucleotide_mask = recover_nucleotide_mask(translated_masks_by_frames, genomic_record)
        if any(nucleotide_mask) or not skip_non_positive:
            start, end = get_mask_trim_coords(nucleotide_mask, offset=trimming_offset)

            genomic_record_data = {
                "g_record_id": str(rec_id),
                "record_description": genomic_record.description,
                "pos_count": sum(nucleotide_mask),
                "sequence": str(genomic_record[start:end].seq),
                "prediction_mask": stringify_mask(nucleotide_mask)[start:end],
                "selected_fragment_start": start,
                "selected_fragment_end": end,
            }
            if (scores := genomic_record_pred_data.get("score")) is not None:
                genomic_record_data["score"] = scores.sum()
            if (model_names := genomic_record_pred_data.get("model_name")) is not None:
                genomic_record_data["model_name"] = model_names.unique()[0]

            nucleotide_predictions_data.append(pd.DataFrame(genomic_record_data, index=[0]))
    return pd.concat(nucleotide_predictions_data, ignore_index=True)


@require_columns({"sequence", "prediction_mask"})
def extract_aa_predictions(
        predictions_data: pd.DataFrame,
        proteome_map: dict[str, SeqRecord] = None,
        skip_non_positive: bool = True,
        trimming_offset: tuple[Optional[int], Optional[int]] | np.ndarray = (None, None)
) -> pd.DataFrame:

    predictions_data = predictions_data.copy(deep=True)

    def trim_aa_seq_mask(row: pd.Series) -> tuple[str, str, int, int]:
        start, end = get_mask_trim_coords(row.prediction_mask, offset=trimming_offset)
        return row.sequence[start:end], row.prediction_mask[start:end], start, end

    if not skip_non_positive:
        if proteome_map is None:
            raise ValueError(f"'skip_non_positive' option requires 'proteome_map' argument")
        missing_frames_df = get_missing_frames_data(predictions_data, proteome_map=proteome_map)
        predictions_data = pd.concat([predictions_data, missing_frames_df], ignore_index=True).dropna(axis=1)

    predictions_data[["sequence", "prediction_mask", "selected_fragment_start", "selected_fragment_end"]] = \
        predictions_data.apply(trim_aa_seq_mask, axis=1, result_type="expand").values

    return predictions_data


@require_columns({"g_record_id", "frame"})
def get_missing_frames_data(predictions_data: pd.DataFrame,
                            proteome_map: dict[str, SeqRecord]) -> pd.DataFrame:
    missing_frames_data = []
    for rec_id, genomic_record_pred_data in predictions_data.groupby("g_record_id"):
        missing_frames = {'1', '2', '3'} - set(genomic_record_pred_data.frame.values)
        for missing_frame in missing_frames:
            translated_record_id = str(rec_id) + missing_frame
            translated_record = proteome_map[translated_record_id]

            genomic_record_missing_frames = {
                "t_record_id": translated_record_id,
                "g_record_id": str(rec_id),
                "record_description": translated_record.description,
                "pos_count": 0,
                "sequence": str(translated_record.seq),
                "prediction_mask": "0" * len(translated_record),
                "frame": missing_frame
            }
            if (scores := genomic_record_pred_data.get("score")) is not None:
                genomic_record_missing_frames["score"] = 0

            missing_frames_data.append(pd.DataFrame(genomic_record_missing_frames, index=[0]))

    return pd.concat(missing_frames_data, ignore_index=True)
