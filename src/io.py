import re

import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from src.logs import logger
from src.utils import encode_mask, parse_record_id


def read_fasta_as_dict(fasta_file_path: str) -> dict[str, SeqRecord]:
    return {record.id: record for record in SeqIO.parse(fasta_file_path, "fasta")}


def read_predictions(prediction_file_path: str) -> pd.DataFrame:
    predictions_data = pd.read_table(prediction_file_path, index_col=None)
    predictions_data = predictions_data[~predictions_data["record_id"].str.startswith("# ")]
    frames = predictions_data["record_id"].apply(lambda rec_id: parse_record_id(rec_id).frame)
    is_trans_record_id = frames.str.isdigit()
    predictions_data.loc[:, "g_record_id"] = predictions_data.loc[~is_trans_record_id, "record_id"]
    predictions_data.loc[is_trans_record_id, "g_record_id"] = \
        predictions_data.loc[is_trans_record_id, "record_id"].str.rstrip("123")
    predictions_data.loc[:, "t_record_id"] = predictions_data.loc[is_trans_record_id, "record_id"]
    predictions_data["frame"] = frames
    predictions_data = predictions_data.drop(columns="record_id") \
                                       .dropna(axis=1, how="all")
    return predictions_data


def save_predictions_as_fasta(predictions_data: pd.DataFrame, file_path: str):
    records = []
    for idx, row in predictions_data.iterrows():
        row_data = dict(row.astype(str))
        row_data["prediction_mask"] = encode_mask(row_data["prediction_mask"])
        g_rec_id = row_data.pop("g_record_id", np.nan)
        t_rec_id = row_data.pop("t_record_id", np.nan)
        if not pd.isna(t_rec_id):
            rec_id = t_rec_id
        elif not pd.isna(g_rec_id):
            rec_id = g_rec_id
        else:
            logger.warning(f"Neither of 't_record_id' or 'g_record_id' were found for prediction {idx}. Skipping")
            continue

        record = SeqRecord(
            Seq(row_data.pop("sequence")),
            id=rec_id,
            description="{} meta[{}]".format(
                row_data.pop('record_description'),
                ";".join([f"{key}={val}" for key, val in row_data.items()])
            )
        )
        records.append(record)
    SeqIO.write(records, file_path, "fasta")


def get_augustus_proteins(augustus_gff_file):
    with open(augustus_gff_file) as gff:
        content = gff.read()

    annotation = pd.read_table(augustus_gff_file, comment="#", header=None,
                               names=["SeqName", "Source", "FeatureType", "Start",
                                      "End", "Score", "Strand", "Phase", "Attributes"])

    sequence_names = re.findall(r"name = (.+)\)", content)
    raw_protein_sequences = re.findall(r"protein sequence = \[([A-Z# \n]+?)\]", content, flags=re.MULTILINE)
    protein_sequences = [seq.replace("\n# ", "") for seq in raw_protein_sequences]

    sequence_names_filtered = [seq_name for seq_name in sequence_names if seq_name in annotation.SeqName.values]
    assert len(sequence_names_filtered) == len(protein_sequences)

    records = []
    for protein_sequence, sequence_name in zip(protein_sequences, sequence_names_filtered):
        sequence_annotation = annotation.query("SeqName == @sequence_name")
        gene_start = sequence_annotation["Start"].sort_values().iloc[0]
        gene_end = sequence_annotation["End"].sort_values().iloc[-1]
        record = SeqRecord(
            Seq(protein_sequence),
            id=sequence_name,
            description=f"Gene_start={gene_start} Gene_end={gene_end} Auto"
        )
        records.append(record)
    return records
