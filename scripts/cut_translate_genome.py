import argparse
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from Bio import SeqIO
from joblib import Parallel, delayed

from src.logs import logger
from src.utils import six_frame_translate, cut_record, filter_masked_records, check_path

warnings.filterwarnings("ignore", module="Bio.Seq")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--fasta', type=Path, help="Path to genomic fasta file")
    parser.add_argument('-o', '--output_file', type=Path, help="Path to resulted file or directory")
    parser.add_argument('-s', '--fragment_size', type=int, default=30000, help="Get fragments of desired length")
    parser.add_argument('-v', '--overlap_size', type=int, default=300, help="Overlap between fragments")
    parser.add_argument('-r', '--repeats_filter_threshold', type=float, default=1.0,
                        help="Drop fragment if fraction of masked nucleotides is bigger than threshold")
    parser.add_argument('-t', '--threads', type=int, default=1, help="Number of parallel units to use")

    args = parser.parse_args()

    logger.info(f"Started processing file {args.fasta} with fragment_size={args.fragment_size} "
                f"and overlap_size={args.overlap_size} utilizing {args.threads} threads")

    translated_records = []
    for idx, genomic_record in enumerate(SeqIO.parse(args.fasta, "fasta"), start=1):
        raw_genomic_record_fragments = cut_record(genomic_record, args.fragment_size, args.overlap_size)
        genomic_record_fragments = filter_masked_records(raw_genomic_record_fragments, args.repeats_filter_threshold)
        logger.info(f"Contig #{idx} {genomic_record.id} was cut into {len(raw_genomic_record_fragments)} fragments. "
                    f"{len(genomic_record_fragments)} ({len(genomic_record_fragments) / len(raw_genomic_record_fragments) * 100:.2f}%) of them passed the repeat filter")
        if len(genomic_record_fragments) == 1:
            translated_record_fragments = six_frame_translate(genomic_record_fragments[0])
        else:
            translated_record_fragments = sum(
                Parallel(n_jobs=args.threads)(delayed(six_frame_translate)(fragment)
                                              for fragment in genomic_record_fragments),
                []
            )
        translated_records.extend(translated_record_fragments)

    output_file = check_path(args.output_file)

    SeqIO.write(translated_records, output_file, "fasta")
    logger.success(f"Successfully saved translated fragments to {output_file}")
