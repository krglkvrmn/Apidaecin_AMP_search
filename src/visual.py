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


def visualize_single_prediction(single_prediction_data: pd.Series,
                                caption_template: str = "",
                                offset: tuple[Optional[int], Optional[int]] | np.ndarray = (None, None)):
    start, end = get_mask_trim_coords(single_prediction_data.prediction_mask, offset=offset)
    sequence = single_prediction_data.sequence[start:end]
    prediction_mask = single_prediction_data.prediction_mask[start:end]
    caption = caption_template.format_map(single_prediction_data)
    print(caption + f" FragmentStart={start}_FragmentEnd={end}")
    print(colorize_by_mask(sequence, prediction_mask))


def visualize_predictions(predictions_data: pd.DataFrame,
                          genome_map: Optional[dict[str, SeqRecord]] = None,
                          proteome_map: Optional[dict[str, SeqRecord]] = None,
                          show_translated: bool = True,
                          show_nucleotide: bool = True,
                          show_missing_frames: bool = False,
                          nucleotide_offset: tuple[Optional[int], Optional[int]] | np.ndarray = None,
                          translated_offset: tuple[Optional[int], Optional[int]] | np.ndarray = None,
                          translated_caption_template: str = "{t_record_id}",
                          nucleotide_caption_template: str = "{g_record_id}"):

    if nucleotide_offset is not None and translated_offset is None:
        nucleotide_offset = np.array(nucleotide_offset)
        translated_offset = nucleotide_offset // 3
    elif nucleotide_offset is None and translated_offset is not None:
        translated_offset = np.array(translated_offset)
        nucleotide_offset = translated_offset * 3
    elif nucleotide_offset is None and translated_offset is None:
        translated_offset = nucleotide_offset = (None, None)

    predictions_to_visualize = []

    if show_translated:
        if show_missing_frames and proteome_map is None:
            raise ValueError(f"'show_missing_frame' option requires 'proteome_map'")
        translated_predictions_data = extract_aa_predictions(
            predictions_data, proteome_map=proteome_map,
            skip_non_positive=not show_missing_frames, trimming_offset=translated_offset
        ).assign(sequence_type="translated")
        predictions_to_visualize.append(translated_predictions_data)

    if show_nucleotide:
        if genome_map is None:
            raise ValueError(f"'show_nucleotide' option requires 'genome_map'")
        nucleotide_predictions_data = extract_nucl_predictions(
            predictions_data, genome_map=genome_map, skip_non_positive=False, trimming_offset=nucleotide_offset
        ).assign(sequence_type="nucleotide")
        predictions_to_visualize.append(nucleotide_predictions_data)

    if not predictions_to_visualize:
        logger.warning(f"Neither 'show_nucleotide' nor 'show_translated' options were provided")

    shown_predictions = set()
    predictions_to_visualize = pd.concat(predictions_to_visualize, ignore_index=True)
    for g_rec_id in predictions_to_visualize["g_record_id"].unique():
        g_rec_subset = predictions_to_visualize.query("g_record_id == @g_rec_id") \
                                               .sort_values("frame")
        for _, row in g_rec_subset.iterrows():
            match row["sequence_type"]:
                case "translated":
                    caption_template = translated_caption_template
                    offset = translated_offset
                case "nucleotide":
                    caption_template = nucleotide_caption_template
                    offset = nucleotide_offset
            visualize_single_prediction(row, caption_template=caption_template, offset=offset)

        print("=" * 120)
        print(" " * 120)
