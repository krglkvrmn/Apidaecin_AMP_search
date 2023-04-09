from typing import List

from Bio.SeqRecord import SeqRecord


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
        SeqRecord(record.seq.reverse_complement().translate(), id=f"{record.id}_frame=-1", description=record.description),
        SeqRecord(record.seq.reverse_complement()[1:].translate(), id=f"{record.id}_frame=-2", description=record.description),
        SeqRecord(record.seq.reverse_complement()[2:].translate(), id=f"{record.id}_frame=-3", description=record.description),
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
        end_idx = start_idx + fragment_size
        new_record = record[start_idx:end_idx]
        new_record.id = f"{record.id}_start={start_idx}_end={end_idx}"
        new_records.append(new_record)
    return new_records
