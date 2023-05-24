import argparse
import os.path

import pandas as pd

from src.logs import logger
from src.ncbi import download_file

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=str, required=True, help="Path to data reference table")
    parser.add_argument("--assemblies", type=str, help="Path to directory to download assemblies into")
    parser.add_argument("--annotations", type=str, help="Path to directory to download annotations into")
    parser.add_argument("--force", action="store_true", help="Download all files even if they exist")

    args = parser.parse_args()

    if args.assemblies is None and args.annotations is None:
        raise ValueError("At least target assembly or target annotation dir should be provided")

    if args.assemblies is not None:
        os.makedirs(args.assemblies, exist_ok=True)

    if args.annotations is not None:
        os.makedirs(args.annotations, exist_ok=True)

    data = pd.read_table(args.table, index_col=None)
    for idx, row in data[["Species", "AsmFtpPath", "AnnFtpPath"]].iterrows():
        species = row.Species.replace(" ", "_")
        if args.assemblies is not None:
            asm_target = os.path.join(args.assemblies, f"{species}.fna.gz")
            if not os.path.exists(asm_target) or args.force:
                logger.info(f"Downloading assembly {idx + 1}/{len(data)} for species \"{species}\" from {row.AsmFtpPath}")
                download_file(url=row.AsmFtpPath, destination_file=asm_target)
                logger.success(f"Assembly for species \"{species}\" successfully downloaded to {asm_target}")
            else:
                logger.debug(f"Assembly for species \"{species}\" already exists in {asm_target}")
        if args.annotations is not None:
            if not pd.isna(row.AnnFtpPath):
                ann_target = os.path.join(args.annotations, f"{species}.gff.gz")
                if not os.path.exists(ann_target) or args.force:
                    logger.info(f"Downloading annotation for species \"{species}\" from {row.AnnFtpPath}")
                    download_file(url=row.AnnFtpPath, destination_file=ann_target)
                    logger.success(f"Annotation for species \"{species}\" successfully downloaded to {ann_target}")
                else:
                    logger.debug(f"Annotation for species \"{species}\" already exists in {ann_target}")
            else:
                logger.debug(f"Annotation for \"{species}\" does not exist. Skipping...")
