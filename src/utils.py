import re
import warnings
from copy import deepcopy
from typing import List, Optional, Literal, Collection, NamedTuple, Sized

import numpy as np
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from src._typing import PredictionMaskAny, PredictionMaskBool, PredictionMaskStr

warnings.filterwarnings("ignore")


FragmentRecordId = NamedTuple("FragmentRecordId", [
    ("id", str), ("start", int),
    ("end", int), ("strand", str),
    ("frame", str)
])


def parse_record_id(record_id):
    elements = re.findall(r"(.+)_start=(\d+)_end=(\d+)_frame=([+-])([1-3])?", record_id)[0]
    return FragmentRecordId(id=elements[0], start=int(elements[1]), end=int(elements[2]),
                            strand=elements[3], frame=elements[4])


def get_genome_fragment_by_translated_id(record_id: str, genome_map: dict[str, SeqRecord]) -> SeqRecord:
    rec_id = parse_record_id(record_id)
    genomic_record = genome_map[rec_id.id][rec_id.start:rec_id.end]
    if rec_id.strand == "-":
        genomic_record = genomic_record.reverse_complement()
    return genomic_record


def six_frame_translate(record: SeqRecord) -> List[SeqRecord]:
    """
    Translate a DNA sequence record into protein sequences in all six possible reading frames.

    This function takes a SeqRecord object containing a DNA sequence, and returns a list of SeqRecord objects with the translated protein sequences for each of the six reading frames (three forward and three reverse).

    :param record: A SeqRecord object containing a DNA sequence.
    :type record: SeqRecord
    :return: A list of SeqRecord objects containing the translated protein sequences for each of the six reading frames.
    :rtype: List[SeqRecord]
    """
    translated_records = [
        SeqRecord(record.seq.translate(), id=f"{record.id}_frame=+1", description=record.description),
        SeqRecord(record.seq[1:].translate(), id=f"{record.id}_frame=+2", description=record.description),
        SeqRecord(record.seq[2:].translate(), id=f"{record.id}_frame=+3", description=record.description),
        SeqRecord(record.seq.reverse_complement().translate(),
                  id=f"{record.id}_frame=-1", description=record.description),
        SeqRecord(record.seq.reverse_complement()[1:].translate(),
                  id=f"{record.id}_frame=-2", description=record.description),
        SeqRecord(record.seq.reverse_complement()[2:].translate(),
                  id=f"{record.id}_frame=-3", description=record.description),
    ]
    return translated_records


def cut_record(record: SeqRecord, fragment_size: int, overlap_size: int) -> List[SeqRecord]:
    """
    Cut a sequence record into fragments of a specified size, allowing for optional overlapping regions.

    This function takes a SeqRecord object, a fragment size, and an overlap size, and returns a list of SeqRecord objects where the original sequence has been divided into fragments of the specified size with optional overlaps between them.

    :param record: A SeqRecord object containing a sequence.
    :type record: SeqRecord
    :param fragment_size: The size of the fragments to divide the sequence into.
    :type fragment_size: int
    :param overlap_size: The size of the overlapping region between adjacent fragments.
    :type overlap_size: int
    :return: A list of SeqRecord objects containing the sequence fragments.
    :rtype: List[SeqRecord]
    """
    new_records = []
    for idx in range(0, len(record), fragment_size):
        start_idx = idx - overlap_size * (idx > 0 and overlap_size < fragment_size)
        end_idx = e if (e := start_idx + fragment_size) < len(record) else len(record)
        new_record = record[start_idx:end_idx]
        new_record.id = f"{record.id}_start={start_idx}_end={end_idx}"
        new_records.append(new_record)
    return new_records


def masked_fraction(sequence: Seq | str) -> float:
    return sum(nucl.islower() for nucl in sequence) / len(sequence)


def filter_masked_records(records_list: List[SeqRecord], mask_threshold: float = 0.0) -> List[SeqRecord]:
    return [record for record in records_list if masked_fraction(record.seq) < mask_threshold]


def numerize_mask(mask: PredictionMaskAny) -> PredictionMaskBool:
    if isinstance(mask[0], str):
        return np.array([char == "1" for char in mask], dtype=np.bool8)
    return np.array(mask)


def stringify_mask(mask: PredictionMaskBool) -> PredictionMaskStr:
    return "".join(map(lambda x: chr(x + 48), mask))


def get_mask_trim_coords(mask: PredictionMaskAny,
                         offset: tuple[Optional[int], Optional[int]] | np.ndarray) -> tuple[int, int]:
    mask = numerize_mask(mask)
    if not any(mask):
        return 0, len(mask)
    start = s - offset[0] if offset[0] is not None and (s := np.where(mask)[0][0]) >= offset[0] else 0
    end = len(mask) if offset[1] is None or (e := np.where(mask)[0][-1]) + offset[1] >= len(mask) else e + offset[1]
    return start, end


def recover_nucleotide_mask(protein_masks: dict[Literal[1, 2, 3, "1", "2", "3"], PredictionMaskAny],
                            nucleotide_sequence: Sized) -> PredictionMaskBool:
    mask = np.zeros(len(nucleotide_sequence), dtype=np.bool8)
    for frame, prot_mask in protein_masks.items():
        frame = int(frame)
        nucl_mask = numerize_mask("".join(char * 3 for char in prot_mask))
        if len(nucl_mask) == mask.size:
            mask = mask | nucl_mask
        else:
            diff = mask.size - len(nucl_mask)
            if diff == 3:
                mask[frame-1:frame-4] = mask[frame-1:frame-4] | nucl_mask
            else:
                mask[:-diff] = mask[:-diff] | nucl_mask
    return mask


def get_unique_nucleotide_record_ids(record_ids: Collection["FragmentRecordId"]) -> List["FragmentRecordId"]:
    unique_nucl_record_ids = []   # List to preserve order
    for record_id in record_ids:
        record_id = deepcopy(record_id)
        record_id.frame = ""
        if record_id not in unique_nucl_record_ids:
            unique_nucl_record_ids.append(record_id)
    return unique_nucl_record_ids


def expand_mask_hits(mask: PredictionMaskAny, kernel_size: int) -> PredictionMaskBool:
    mask = numerize_mask(mask)
    expand_kernel = np.ones(kernel_size, dtype=np.bool8)
    mask = np.convolve(mask, expand_kernel)[kernel_size//2:-(kernel_size//2)]
    return mask


def encode_mask(mask: PredictionMaskStr) -> str:
    result = []
    count = 1
    for i in range(1, len(mask)):
        if mask[i] == mask[i-1]:
            count += 1
        else:
            result.append(('t' if mask[i-1] == '1' else 'f') + str(count))
            count = 1
    result.append(('t' if mask[-1] == '1' else 'f') + str(count))
    return ''.join(result)


def decode_mask(encoded_mask: str) -> PredictionMaskStr:
    result = []
    i = 0
    while i < len(encoded_mask):
        char = '1' if encoded_mask[i] == 't' else '0'
        i += 1
        count = ''
        while i < len(encoded_mask) and encoded_mask[i].isdigit():
            count += encoded_mask[i]
            i += 1
        result.append(char * int(count))
    return ''.join(result)


