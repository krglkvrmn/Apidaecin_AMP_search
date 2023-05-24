import argparse
import os.path

import pandas as pd

from src.ncbi import get_assemblies_info, get_taxonomy_info


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--taxon', type=str, required=True, help='Taxon whose assemblies will be used')
    parser.add_argument('--file', type=str, required=True, help='File to save resulting table into')
    parser.add_argument('--retmax', type=int, default=10000, help='Max number of entries returned by NCBI server')

    args = parser.parse_args()

    assembly_data_table = get_assemblies_info(taxon=args.taxon, retmax=args.retmax)
    taxonomy_data_table = get_taxonomy_info(taxid=assembly_data_table["TaxId"], retmax=args.retmax)
    taxonomy_data_table.columns = [f"tax_{col}" if col != "TaxId" else col for col in taxonomy_data_table.columns ]
    complete_data_table = assembly_data_table.merge(taxonomy_data_table, on="TaxId")

    if os.path.exists(args.file):
        original_table = pd.read_table(args.file, index_col=None)
        complete_data_table = pd.concat([original_table, complete_data_table], ignore_index=True)

    complete_data_table.to_csv(args.file, sep='\t', index=False)
