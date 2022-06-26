from dataclasses import dataclass
from functools import partial
from typing import Callable, List, Optional

import optuna
from loguru import logger
from sklearn.model_selection import StratifiedKFold
from torch.utils.tensorboard import SummaryWriter

from train_funcs.logs import tb_run, reinit_tensorboard_local
from train_funcs.parameters import Hyperparameters, ModelParameters


@dataclass
class NestedCVresults:
    study: optuna.Study
    best_params: dict
    loss: float
    metrics: dict
    hp: Hyperparameters


def nested_cv(trainer_preset: Callable, objective: Callable, X: List[str], y: List[int],
              hp_preset: Callable, mp_preset: Callable, n_trials: int = 10, outer_k: int = 3,
              inner_k: int = 3, use_tensorboard: bool = False,
              random_state: Optional[int] = None) -> List[NestedCVresults]:
    """
    Run nested cross-validation
    :param trainer_preset: Trainer class with preset arguments
    :type trainer_preset: Callable
    :param objective: Optuna objective function
    :type objective: Callable
    :param X: A list of sequences of any length
    :type X: List[str]
    :param y: A list of labels corresponding to each sequence in X
    :type y: List[int]
    :param hp_preset: Hyperparameters class with preset parameters
    :type hp_preset: Callable
    :param mp_preset: ModelParameters class with preset parameters
    :type mp_preset: Callable
    :param n_trials: A number of trials to perform on every outer split
    :type n_trials: int
    :param outer_k: Number of outer splits
    :type outer_k: int
    :param inner_k: Number of inner splits
    :type inner_k: int
    :param use_tensorboard: Whether to use Tensorboard for logging
    :type use_tensorboard: bool
    :param random_state: Random state seed
    :type random_state: Optional[int]
    :return: A list of NestedCVresults objects, which contain information about each outer split results
    :rtype:  List[NestedCVresults]
    """

    outer_cv_results = []
    outer_kfold = StratifiedKFold(n_splits=outer_k, shuffle=True, random_state=random_state)
    logger.info(f"Started nested {outer_k}-fold CV")
    for current_split, (outer_train_indices, test_indices) in enumerate(outer_kfold.split(X, y)):
        logger.info(f"Running nested CV, training outer split {current_split + 1}/{outer_k}")
        outer_X_train, X_test = X[outer_train_indices], X[test_indices]
        outer_y_train, y_test = y[outer_train_indices], y[test_indices]

        inner_kfold = StratifiedKFold(n_splits=inner_k, shuffle=True, random_state=random_state + current_split)

        objective_preset = partial(
            objective,
            trainer_preset=trainer_preset,
            X=outer_X_train,
            y=outer_y_train,
            kfold=inner_kfold,
            mp_preset=mp_preset,
            hp_preset=hp_preset,
            outer_split=(current_split + 1, outer_k),
            use_tensorboard=use_tensorboard
        )

        study = optuna.create_study()
        study.optimize(objective_preset, n_trials=n_trials)

        best_params = study.best_params

        logger.info(f"Running nested CV, validating outer split {current_split + 1}/{outer_k}")

        mp = mp_preset(**{param: value for param, value in best_params.items() if param in vars(ModelParameters)})
        hp = hp_preset(model_parameters=mp,
                       **{param: value for param, value in best_params.items() if param in vars(Hyperparameters)})
        trainer = trainer_preset(X_train=outer_X_train, X_val=X_test,
                                 y_train=outer_y_train, y_val=y_test,
                                 hyperparameters=hp)
        if use_tensorboard:
            outer_split_tb_run = tb_run(f"nested_CV/outer_split_{current_split + 1}_validation")
            reinit_tensorboard_local(outer_split_tb_run, clear_log=True)
            writer = SummaryWriter(log_dir=outer_split_tb_run)
        else:
            writer = None
        loss, metrics = trainer.train(n_epochs=60, valid=6, writer=writer)
        results = NestedCVresults(
            study=study,
            best_params=best_params,
            loss=loss,
            metrics=metrics,
            hp=hp
        )
        outer_cv_results.append(results)
    return outer_cv_results
