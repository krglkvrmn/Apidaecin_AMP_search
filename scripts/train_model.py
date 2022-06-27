import argparse
import os.path
from importlib import import_module

import numpy as np
from Bio import SeqIO
from loguru import logger

from src.models import SimpleCNN, HybridModel
from src.parameters import parse_config_as_dict, ModelParameters, Hyperparameters
from src.training import Trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--apidaecins_file", required=True, help="Fasta file containing preprocessed apidaecins sequences")
    parser.add_argument("--database_file", required=False, default=None, help="Fasta file containing other AMP sequences")
    parser.add_argument("--proteins_file", required=True, help="Fasta file containing non-AMP sequences")
    parser.add_argument("--model_name", required=True, choices=["SimpleCNN", "HybridModel"],
                        help="Model name, which is also prefix for saved parameters file")
    parser.add_argument("--config", required=False, help="Config containing adjustable model and training parameters")
    parser.add_argument("--epochs", required=False, default=10, type=int, help="Number of epochs to train on. Default: 10")
    parser.add_argument("--save_dir", required=False, default="models", help="Directory to save trained model into. Default: 'models'")

    args = parser.parse_args()

    apidaecins_sequences = list(set(map(lambda rec: str(rec.seq), SeqIO.parse(args.apidaecins_file, "fasta"))))
    logger.info(f"Loaded {len(apidaecins_sequences)} apidaecins sequences")

    proteins_sequences = list(set(map(lambda rec: str(rec.seq), SeqIO.parse(args.proteins_file, "fasta"))))
    logger.info(f"Loaded {len(proteins_sequences)} negative class sequences")

    if args.database_file:
        database_sequences = set()
        for record in SeqIO.parse(args.database_file, "fasta"):
            if "Apidaecin" in record.description or "apidaecin" in record.description:
                continue
            database_sequences.add(str(record.seq))
        database_sequences = list(database_sequences)
    else:
        database_sequences = []
    logger.info(f"Loaded {len(database_sequences)} database sequences")

    if args.config:
        try:
            config_dict = parse_config_as_dict(args.config)
            mp = ModelParameters(**config_dict["mp"])
            hp = Hyperparameters(model_parameters=mp, **config_dict["hp"])
            device = config_dict["gp"]["device"]
            logger.info(f"Config was successfully loaded")
        except Exception:
            logger.error(f"Config was loaded with errors. Falling back to default parameters")
            hp = Hyperparameters()
            device = 'cpu'
    else:
        logger.info(f"Config was not loaded. Falling back to default parameters")
        hp = Hyperparameters()
        device = "cpu"

    x_data = np.array(apidaecins_sequences + proteins_sequences + database_sequences)
    y_labels = np.array([1] * len(apidaecins_sequences) + [0] * len(proteins_sequences) + [2] * len(database_sequences))

    model_class = HybridModel if args.model_name == "HybridModel" else SimpleCNN
    trainer = Trainer(model_class, X_train=x_data, X_val=None, y_train=y_labels, y_val=None,
                      hyperparameters=hp, setup=True, device=device)

    trainer.train(n_epochs=args.epochs, writer=None, valid=False, vis_seqs=None, cache_embeddings=False)
    trainer.save_model(args.save_dir, args.model_name)
