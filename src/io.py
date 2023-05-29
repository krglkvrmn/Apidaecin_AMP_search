import re
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from src.logs import logger
from src.predictions import require_columns
from src.utils import encode_mask, parse_record_id, decode_mask


_COLUMN_DATA_TYPES = {"score": float, "pos_count": int}


def read_fasta_as_dict(fasta_file_path: str) -> dict[str, SeqRecord]:
    return {record.id: record for record in SeqIO.parse(fasta_file_path, "fasta")}


def read_predictions(prediction_file_path: str) -> pd.DataFrame:
    predictions_data = pd.read_table(prediction_file_path, index_col=None)
    return _preprocess_predictions(predictions_data)


def read_predictions_from_fasta(fasta_file_path: str) -> pd.DataFrame:
    predictions_data = []
    for record in SeqIO.parse(fasta_file_path, "fasta"):
        metadata_match = re.search(r'meta\[(.+)\]', record.description)
        single_prediction_data = {
            "record_id": record.id,
            "record_description": record.description.replace(metadata_match.group(0), ""),
            "sequence": str(record.seq)
        }
        metadata_keyvalues = metadata_match.group(1).split(";")
        single_prediction_data |= dict([keyval.split("=") for keyval in metadata_keyvalues])
        predictions_data.append(pd.Series(single_prediction_data))

    predictions_data = pd.concat(predictions_data, axis=1).T
    predictions_data = _preprocess_predictions(predictions_data)
    return predictions_data


@require_columns({"record_id", "prediction_mask"})
def _preprocess_predictions(predictions_data: pd.DataFrame) -> pd.DataFrame:
    predictions_data = predictions_data[~predictions_data["record_id"].str.startswith("# ")]
    predictions_data = split_tg_record_ids(predictions_data)
    predictions_data["prediction_mask"] = predictions_data["prediction_mask"].apply(
        lambda mask: decode_mask(mask, to_boolean=True)
    )
    dtypes_mapper = {column: type_ for column, type_ in _COLUMN_DATA_TYPES.items()
                     if column in predictions_data.columns}
    predictions_data = predictions_data.astype(dtypes_mapper)
    predictions_data = reorder_columns(predictions_data)
    return predictions_data


@require_columns({"record_id"})
def split_tg_record_ids(predictions_data: pd.DataFrame) -> pd.DataFrame:
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


def combine_tg_record_ids(predictions_data: pd.DataFrame) -> pd.DataFrame:
    predictions_data = predictions_data.copy(deep=True)
    g_record_ids = predictions_data.get("g_record_id")
    t_record_ids = predictions_data.get("t_record_id")
    if g_record_ids is not None:
        g_rec_id_mask = ~g_record_ids.isna()
        predictions_data.loc[g_rec_id_mask, "record_id"] = predictions_data.loc[g_rec_id_mask, "g_record_id"]
    if t_record_ids is not None:
        t_rec_id_mask = ~t_record_ids.isna()
        predictions_data.loc[t_rec_id_mask, "record_id"] = predictions_data.loc[t_rec_id_mask, "t_record_id"]
    predictions_data.drop(columns=["g_record_id", "t_record_id"], inplace=True, errors="ignore")
    return predictions_data


def reorder_columns(predictions_data: pd.DataFrame) -> pd.DataFrame:
    columns_order_start = ["t_record_id", "g_record_id", "record_id", "pos_count", "score", "model_name"]
    columns_order_end = ["record_description", "prediction_mask", "sequence"]
    columns_order_start = list(filter(lambda col: col in predictions_data.columns, columns_order_start))
    columns_order_end = list(filter(lambda col: col in predictions_data.columns, columns_order_end))
    columns_middle = [col for col in predictions_data.columns if col not in columns_order_start + columns_order_end]
    new_columns = columns_order_start + columns_middle + columns_order_end
    return predictions_data.reindex(columns=new_columns)


def save_predictions(predictions_data: pd.DataFrame, file_path: Path | str):
    predictions_data = combine_tg_record_ids(predictions_data)
    predictions_data = reorder_columns(predictions_data)
    predictions_data.to_csv(file_path, sep="\t", index=False)


def save_predictions_as_fasta(predictions_data: pd.DataFrame, file_path: str):
    predictions_data = combine_tg_record_ids(predictions_data)

    records = []
    for idx, row in predictions_data.iterrows():
        row_data = dict(row.astype(str))
        row_data["prediction_mask"] = encode_mask(row["prediction_mask"], from_boolean=True)
        rec_id = row_data.pop("record_id", np.nan)
        if pd.isna(rec_id):
            logger.warning(f"'record_id' was not found for prediction {idx}. Skipping")
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
