import itertools
import re
import warnings
from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Literal, Collection, NamedTuple, Sized

import numpy as np
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from src._typing import PredictionMaskAny, PredictionMaskBool, PredictionMaskStr, PredictionMaskStrEncoded, \
    PredictionMaskStrDecoded

warnings.filterwarnings("ignore")


_ENCODING_MAP = {'1': 't', '0': 'f', True: 't', False: 'f'}
_DECODING_MAP = {'t': '1', 'f': '0'}


FragmentRecordId = NamedTuple("FragmentRecordId", [
    ("id", str), ("start", int),
    ("end", int), ("strand", str),
    ("frame", str)
])


def parse_record_id(record_id: str) -> FragmentRecordId:
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


def encode_mask(mask: PredictionMaskStrDecoded | PredictionMaskBool,
                from_boolean: bool = False) -> PredictionMaskStrEncoded:
    # https://stackoverflow.com/questions/18948382/run-length-encoding-in-python
    if from_boolean:
        return encode_mask(stringify_mask(mask), from_boolean=False)
    if mask[0] in _DECODING_MAP:
        raise TypeError("Mask is already encoded")
    return PredictionMaskStrEncoded(
        "".join(f"{_ENCODING_MAP[char]}{sum(1 for _ in char_group)}"
                for char, char_group in itertools.groupby(mask))
    )


def decode_mask(mask: PredictionMaskStrEncoded,
                to_boolean: bool = False) -> PredictionMaskStrDecoded | PredictionMaskBool:
    if to_boolean:
        return numerize_mask(decode_mask(mask, to_boolean=False))

    def _decode_iter(mask):
        mask_iterator = iter(mask)
        try:
            encoded_char = next(mask_iterator)
            while True:
                count_str = ""
                while (next_char := next(mask_iterator)) not in _DECODING_MAP:
                    count_str += next_char
                yield _DECODING_MAP[encoded_char] * int(count_str)
                encoded_char = next_char
        except StopIteration:
            yield _DECODING_MAP[encoded_char] * int(count_str)

    if len(mask) == 0:
        return PredictionMaskStrDecoded("")
    return PredictionMaskStrDecoded("".join(_decode_iter(mask)))


def numerize_mask(mask: PredictionMaskAny) -> PredictionMaskBool:
    match mask[0]:
        case 1 | 0:
            return PredictionMaskBool(np.array(mask, dtype=np.bool8))
        case "1" | "0":
            return PredictionMaskBool(np.array([char == "1" for char in mask], dtype=np.bool8))
        case "t" | "f":
            return decode_mask(mask, to_boolean=True)
        case _:
            raise TypeError(f"Invalid mask type: {type(mask).__name__}")


def stringify_mask(mask: PredictionMaskAny) -> PredictionMaskStr | PredictionMaskStrEncoded:
    match mask[0]:
        case 1 | 0:
            return PredictionMaskStrDecoded("".join(chr(m) for m in mask + 48))   # 48 and 49 are ASCII codes of '1' and '2'
        case "1" | "0":
            return mask
        case "t" | "f":
            return decode_mask(mask)
        case _:
            raise TypeError(f"Invalid mask type: {type(mask).__name__}")


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
        prot_mask = numerize_mask(prot_mask)
        nucl_mask = np.array(list(itertools.chain.from_iterable(itertools.repeat(char, 3) for char in prot_mask)))
        nucl_mask = numerize_mask(nucl_mask)
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


def check_path(path: Optional[Path | str], force_overwrite: bool = False) -> Optional[Path]:
    if path is None:
        return None
    elif isinstance(path, str):
        path = Path(path)
    elif not isinstance(path, Path):
        raise ValueError(f"Invalid path type: {type(path).__name__}")

    if path.suffix == "":
        path.mkdir(parents=True, exist_ok=True)
    elif path.suffix != "":
        path.parent.mkdir(parents=True, exist_ok=True)
        if force_overwrite:
            path.open("w").close()

    return path
