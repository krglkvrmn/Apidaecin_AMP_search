from typing import Optional

import numpy as np
import pandas as pd
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from src._typing import PredictionMaskAny
from src.logs import logger
from src.predictions import extract_aa_predictions, extract_nucl_predictions
from src.utils import numerize_mask, get_mask_trim_coords


def colorize_by_mask(text: Seq | str, mask: PredictionMaskAny) -> str:
    if len(text) != len(mask):
        raise ValueError(f"Text size ({len(text)}) and mask size ({len(mask)}) do not match")
    mask = numerize_mask(mask)
    colored_sequence = []
    for idx, (s, m) in enumerate(zip(text, mask)):
        p = mask[idx - 1]
        if idx and m != p:
            colored_sequence.append("\033[0m")
        if m and (not idx or not p):
            colored_sequence.append("\033[32m")
        elif not m and (not idx or p):
            colored_sequence.append("\033[31m")

        colored_sequence.append(s)
    return "".join(colored_sequence) + "\033[0m"


def visualize_single_prediction(sequence: Seq | str,
                                prediction_mask: PredictionMaskAny,
                                caption: str = "",
                                offset: tuple[Optional[int], Optional[int]] | np.ndarray = (None, None)):
    start, end = get_mask_trim_coords(prediction_mask, offset=offset)
    sequence = sequence[start:end]
    prediction_mask = prediction_mask[start:end]
    print(caption + f" FragmentStart={start}_FragmentEnd={end}")
    print(colorize_by_mask(sequence, prediction_mask))


def visualize_predictions(predictions_data: pd.DataFrame,
                          genome_map: Optional[dict[str, SeqRecord]] = None,
                          proteome_map: Optional[dict[str, SeqRecord]] = None,
                          show_translated: bool = True,
                          show_nucleotide: bool = True,
                          show_missing_frames: bool = False,
                          nucl_offset: tuple[Optional[int], Optional[int]] | np.ndarray = None,
                          aa_offset: tuple[Optional[int], Optional[int]] | np.ndarray = None):

    # TODO: Fix translated and genomic records drawn separately
    if nucl_offset is not None and aa_offset is None:
        nucl_offset = np.array(nucl_offset)
        aa_offset = nucl_offset // 3
    elif nucl_offset is None and aa_offset is not None:
        aa_offset = np.array(aa_offset)
        nucl_offset = aa_offset * 3
    elif nucl_offset is None and aa_offset is None:
        aa_offset = nucl_offset = (None, None)

    predictions_to_visualize = []

    if show_translated:
        if show_missing_frames and proteome_map is None:
            raise ValueError(f"'show_missing_frame' option requires 'proteome_map'")
        translated_predictions_data = extract_aa_predictions(
            predictions_data, proteome_map=proteome_map,
            skip_non_positive=not show_missing_frames, trimming_offset=aa_offset
        ).assign(sequence_type="translated")
        predictions_to_visualize.append(translated_predictions_data)
        for _, row in translated_predictions_data.iterrows():
            visualize_single_prediction(row.sequence, row.prediction_mask,
                                        caption=f"{row.t_record_id} Score={row.pos_count}", offset=aa_offset)

    if show_nucleotide:
        if genome_map is None:
            raise ValueError(f"'show_nucleotide' option requires 'genome_map'")
        nucleotide_predictions_data = extract_nucl_predictions(
            predictions_data, genome_map=genome_map, skip_non_positive=False, trimming_offset=nucl_offset
        ).assign(sequence_type="nucleotide")
        predictions_to_visualize.append(nucleotide_predictions_data)
        for _, row in nucleotide_predictions_data.iterrows():
            visualize_single_prediction(row.sequence, row.prediction_mask,
                                        caption=str(row.g_record_id),
                                        offset=nucl_offset)
        print("=" * 120)
        print(" " * 120)

    if not predictions_to_visualize:
        logger.warning(f"Neither 'show_nucleotide' nor 'show_translated' options were provided")

    return pd.concat(predictions_to_visualize, ignore_index=True)
