import subprocess
from collections import defaultdict
from typing import List, Union

import numpy as np
import pandas as pd
from Bio import Entrez

from src.logs import logger

NCBI_TAXONOMY_RANKS = [
    "superkingdom", "kingdom", "subkingdom", "superphylum", "phylum",
    "subphylum", "superclass", "class", "subclass", "infraclass", "superorder",
    "order", "suborder", "infraorder", "parvorder", "superfamily", "family",
    "subfamily", "tribe", "subtribe", "genus", "subgenus", "species group",
    "species subgroup", "species", "subspecies", "varietas", "forma"
]

Entrez.email = "opistoconta@gmail.com"


def get_assemblies_info(taxon: str, retmax: int = 10000) -> pd.DataFrame:
    filter_query = f'latest[filter] AND "representative genome"[filter] AND all[filter] NOT anomalous[filter]'
    all_handle = Entrez.esearch(db="assembly", term=f"{taxon}[Organism] AND ({filter_query})", retmax=retmax)
    annotated_handle = Entrez.esearch(db="assembly",
                                      term=f'{taxon}[Organism] AND ({filter_query} AND "has annotation"[Properties])',
                                      retmax=retmax)
    all_search_result = Entrez.read(all_handle)
    annotated_search_result = Entrez.read(annotated_handle)
    all_handle.close()
    annotated_handle.close()

    logger.info(f"Found a total of {len(all_search_result['IdList'])} assemblies for taxon \"{taxon}\", "
                f"{len(annotated_search_result['IdList'])} of which are annotated")

    all_handle = Entrez.esummary(db="assembly", id=all_search_result["IdList"])
    annotated_handle = Entrez.esummary(db="assembly", id=annotated_search_result["IdList"])
    all_summary_result = Entrez.read(all_handle)
    annotated_summary_result = Entrez.read(annotated_handle)
    all_handle.close()
    annotated_handle.close()

    annotated_accessions = {entry["AssemblyAccession"]
                            for entry in annotated_summary_result["DocumentSummarySet"]["DocumentSummary"]}

    data = defaultdict(list)
    for result_entry in all_summary_result["DocumentSummarySet"]['DocumentSummary']:
        accession = result_entry["AssemblyAccession"]
        data["AssemblyAccession"].append(accession)
        data["Species"].append(result_entry["SpeciesName"])
        data["LastUpdate"].append(result_entry["LastUpdateDate"])
        ftp_path = path if (path := result_entry["FtpPath_RefSeq"]) != "" else result_entry["FtpPath_GenBank"]
        ftp_path_spl = ftp_path.split("/")
        asm_ftp_path = f'https:/{"/".join(ftp_path_spl[1:])}/{ftp_path_spl[-1]}_genomic.fna.gz'
        ann_ftp_path = asm_ftp_path.replace(".fna.gz", ".gff.gz")
        data["AsmFtpPath"].append(asm_ftp_path)
        if accession in annotated_accessions:
            data["AnnFtpPath"].append(ann_ftp_path)
        else:
            data["AnnFtpPath"].append(np.nan)
        data["TaxId"].append(result_entry["Taxid"])

    return pd.DataFrame(data)


def get_taxonomy_info(taxid: Union[str, List[str]], retmax: int = 10000) -> pd.DataFrame:
    handle = Entrez.efetch(db="taxonomy", id=taxid, retmax=retmax, retmode="xml")
    fetch_result = Entrez.read(handle)
    handle.close()

    taxonomy_table = defaultdict(list)

    for result_entry in fetch_result:
        taxonomy_table["TaxId"].append(result_entry["TaxId"])
        lineage_dict = {lin["Rank"]: lin["ScientificName"] for lin in result_entry["LineageEx"]}
        for rank in NCBI_TAXONOMY_RANKS:
            taxonomy_table[rank].append(lineage_dict.get(rank, np.nan))

    taxonomy_table = pd.DataFrame(taxonomy_table).drop_duplicates("TaxId")
    return taxonomy_table


def download_file(url: str, destination_file: str):
    logger.debug(f"Downloading {url}\tto\t{destination_file}")
    subprocess.run(["wget", "-c", url, "-O", destination_file, "-nv", "--show-progress"])


