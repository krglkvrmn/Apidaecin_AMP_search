import argparse

import numpy as np
from Bio import SeqIO
from sklearn.model_selection import train_test_split

from src.core import Controller
from src.logs import logger
from src.models import SimpleCNN, HybridModel
from src.parameters import parse_config_as_dict, ModelParameters, Hyperparameters

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--apidaecins_file", required=True,
                        help="Fasta file containing preprocessed apidaecins sequences")
    parser.add_argument("--not_apidaecins_file", required=False, default=None,
                        help="Fasta file containing high priority non-apidaecin sequences")
    parser.add_argument("--other_proteins_file", required=True,
                        help="Fasta file containing low priority non-AMP protein sequences")
    parser.add_argument("--model_name", required=True,
                        help="Model name, which is also prefix for saved parameters file")
    parser.add_argument("--config", required=False,
                        help="Config containing adjustable model and training parameters")
    parser.add_argument("--epochs", required=False, default=10, type=int,
                        help="Number of epochs to train on. Default: 10")
    parser.add_argument("--valid_fraction", required=False, default=0.0, type=float,
                        help="Fraction of validation set. Default: 0.0 - no validation")
    parser.add_argument("--save_dir", required=False, default="models",
                        help="Directory to save trained model into. Default: 'models'")

    args = parser.parse_args()

    apidaecins_sequences = list(set(map(lambda rec: str(rec.seq), SeqIO.parse(args.apidaecins_file, "fasta"))))
    logger.info(f"Loaded {len(apidaecins_sequences)} apidaecin sequences")

    other_sequences = list(set(map(lambda rec: str(rec.seq), SeqIO.parse(args.other_proteins_file, "fasta"))))
    logger.info(f"Loaded {len(other_sequences)} negative class sequences")

    if args.not_apidaecins_file:
        not_api_sequences = set()
        for record in SeqIO.parse(args.not_apidaecins_file, "fasta"):
            if "Apidaecin" in record.description or "apidaecin" in record.description:
                continue
            not_api_sequences.add(str(record.seq))
        not_api_sequences = list(not_api_sequences)
    else:
        not_api_sequences = []
    logger.info(f"Loaded {len(not_api_sequences)} non-apidaecin sequences")

    if args.config:
        try:
            config_dict = parse_config_as_dict(args.config)
            mp = ModelParameters(**config_dict["mp"])
            hp = Hyperparameters(model_parameters=mp, **config_dict["hp"])
            logger.debug(f"Training configuration was successfully loaded")
        except Exception as exc:
            logger.warning(f"Config was loaded with errors: {exc}. Falling back to default parameters")
            hp = Hyperparameters()
    else:
        logger.info(f"Config was not loaded. Falling back to default parameters")
        hp = Hyperparameters()

    x_data = np.array(apidaecins_sequences + other_sequences + not_api_sequences)
    y_labels = np.array([1] * len(apidaecins_sequences) + [0] * len(other_sequences) + [2] * len(not_api_sequences))
    if args.valid_fraction != 0:
        X_train, X_val, y_train, y_val = train_test_split(x_data, y_labels, test_size=args.valid_fraction, stratify=y_labels)
    else:
        X_train, X_val, y_train, y_val = x_data, None, y_labels, None

    if args.model_name.startswith("HybridModel"):
        model_class = HybridModel
    elif args.model_name.startswith("SimpleCNN"):
        model_class = SimpleCNN
    trainer = Controller(model_class, X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val,
                         hyperparameters=hp, setup=True)

    trainer.train(n_epochs=args.epochs, writer=None, valid=args.valid_fraction != 0, vis_seqs=None, cache_embeddings=False)
    trainer.save_model(args.save_dir, args.model_name)
