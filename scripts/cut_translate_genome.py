import argparse
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from Bio import SeqIO
from loguru import logger

from src.utils import six_frame_translate, cut_record

warnings.filterwarnings("ignore", module="Bio.Seq")

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--fasta', type=Path, help="Path to genomic fasta file")
parser.add_argument('-o', '--output_path', type=Path, help="Path to resulted file")
parser.add_argument('-s', '--fragment_size', type=int, default=30000, help="Get fragments of desired length")
parser.add_argument('-v', '--overlap_size', type=int, default=300, help="Overlap between fragments")
parser.add_argument('-t', '--threads', type=int, default=1, help="Number of parallel units to use")

args = parser.parse_args()

logger.info(f"Started processing file {args.fasta} with fragment_size={args.fragment_size}"
            f"and overlap_size={args.overlap_size} utilizing {args.threads} threads")

translated_records = []
for genomic_record in SeqIO.parse(args.fasta, "fasta"):
    genomic_record_fragments = cut_record(genomic_record, args.fragment_size, args.overlap_size)
    with ProcessPoolExecutor(args.threads) as pool:
        translated_record_fragments = pool.map(six_frame_translate, genomic_record_fragments)
        translated_record_fragments = sum(translated_record_fragments, [])
    translated_records.extend(translated_record_fragments)

if args.output_path.is_dir():
    output_path = args.output_path / f"{args.fasta.stem}.faa"
else:
    output_path = args.output_path

SeqIO.write(translated_records, output_path, "fasta")
logger.success(f"Successfully saved translated fragments to {output_path}")



