# Deep learning approach for search of novel antimicrobial peptides

## Purpose and objectives of the project

### Purpose:

+ Develop an algorithm, which can perform search of antimicrobial peptides (AMP) coding genes in genomes and proteomes of various organisms.

### Objectives:

+ Prepare training data
+ Develop and train neural network model that can classify whether protein sequence belong to apidaecin-like AMP
+ Select optimal model architecture and data processing methods
+ Validate obtained model on real-world data

## Results

In the course of the project two models for sequence classification were designed:

1. Inaccurate, but fast convolutional neural network (*Conv* or *SimpleCNN*)
2. Slow, but very precise hybrid model possessing convolutional and recurrent layers  (*Conv + LSTM* or *HybridModel*)

On the figure below there are architecture of *HybridModel*. Convolutional layer effectively summarizes local information and LSTM layers are good at summarizing distant data. Combined approach yields very promising results.

![img](https://lh4.googleusercontent.com/ONP2y55cfRniKnJ4EJ1WeDa-SJzTc-FLhWNIbZoBFhe0fTeBn7VSa6wiOIDL3Viq8304eEFRvMGnbZ4RkFF_NqOVT_v_j9WjiD6WXe-r8BQ0zNrMq2j9lhOPzbvyuFWaiwN1C264y6NHnvs7ow)

Convolutional model shows great recall values, but lacks precision, which it crucial for this task. Hybrid model significantly improves precision, while slightly lowers recall, which is pretty acceptable.

![img](https://lh4.googleusercontent.com/tVF2MeQTLTJObEoDVvaeI7Ofv4cKmAC640N33UdKyqWt0IiWjos4hxv7yWjC7iDi85hJcQxv9v-IoRyXwJoH-6CgbKtMAAmp4XupG61ZLbYkXcocVp9X2xfo5WDmf4pNmX4oWIyXVjwIcAozHQ)

On the contrast with convolutional model, hybrid model generates features that allow to easily distinguish positive and negative class sequences, even if negative class possess proline-rich AMP, which convolutional model is struggling to distinhuish from apidaecins.

![img](https://lh4.googleusercontent.com/PLOUFVqcknH09dazoplnDqe5QH-bm8o7NpJ6zFUgSBv0FGaUxM2iX2fYByTuomGlUlmxrydqpiWY2RN3oiZx8a3zNC-K4vjzRIF1-7Hl9Zo52-reZllCZGNKe8MpSTTSU6HYWa_2anoOQurA5w)

As a result, *HybridModel* is a very promising model for classification of apidaecin-like AMP, that might make search of novel antimicrobial peptides much easier.

## Data description

### Training data

All training data used in this project is stored in **data** directory. This includes:

+ **data/apidaecins.fasta**  &mdash; unfiltered apidaecins sequences, obtained from [NCBI database](https://www.ncbi.nlm.nih.gov/protein/)
+ **data/pro-apidaecins.fasta** &mdash; filtered and processed apidaecins sequences. Processing included removal of signal peptides
+ **data/not_apidaecins.fasta** &mdash; all AMP sequences downloaded from [APD3 database](https://aps.unmc.edu/) and sequences that yielded false positive signal on test data. These sequences get higher priority when training a model
+ **data/other_proteins.fasta** &mdash; proteins, which are not related to AMP or any other antibiotics, downloaded from Uniprot database

### Validation data

These files are used for validation purpose

+ **data/proteomes/*** &mdash; proteomes of organisms known to have apidaecins
+ **cv_results** &mdash; nested cross validation results

### Models

**models** directory contain pretrained models. 

* **models/params/*.pk** &mdash; hyperparameters needed to construct model. 
* **models/weights/*.pt** &mdash; saved model parameters.

The newest and the best performing model called **HybridModel_v3**.

### Predictions

**predictions** directory contain predicted labels for validation proteomes from **data/proteomes** and can be used for new predictions

## Workflow overview

### Functions and classes

All reusable and non-interactive code is stored in **src** package.

* **src/dataset.py** &mdash; dataset class that allows to set probability of classes occurrence in batch 
* **src/logs.py**&mdash; functions to manage tensorboard logs
* **src/metrics.py** &mdash; sklearn metrics with predefined *zero_division* parameter
* **src/models.py**&mdash; classes of neural networks
* **src/parameters.py**&mdash; config parsing functions and dataclasses for convenient parameters storage
* **src/processing.py** &mdash; sequence processing, encoding and augmentation
* **src/training.py** &mdash; *Trainer* class that encapsulates all training-related functionality
* **src/validation.py**&mdash; functions for nested cross-validation
* **src/utils.py**&mdash; utility functions

### Notebook

Detailed step-by-step and easy-to-reproduce workflow is available in notebook **AMP_search_with_deep_learning.ipynb**. This includes: 

+ Data processing
+ Models training
+ Hyperparameters optimization and validation via nested cross-validation
+ Cross-validation on proteomic data.

### Scripts

Core functionality such as __*model training*__ and __*prediction on proteomic data*__ were encapsulated into standalone scripts that allow you easily configure and train models and use pretrained models for predictions.

+ **scripts/train_model.py** &mdash; train model on given data (fasta files)
+ **scripts/cut_translate_genome.py**&mdash; translate genome in 6 reading frames and cut sequences with overlaps
+ **scripts/scan_proteome.py** &mdash; make predictions for single proteome using pretrained models
+ **parameters.properties** &mdash; configuration file for **train_model.py** launch. It allows you to easily adjust model hyperparameters.

## Usage

### Download source code and install dependencies (local)

```
git clone https://github.com/krglkvrmn/Apidaecin_AMP_search.git
cd Apidaecin_AMP_search
pip install -r requirements.txt
```

All code was run on **python 3.7.9**. Correct work on other versions is not guaranteed.

### Open notebook (local)

```
jupyter notebook AMP_search_with_deep_learning.ipynb
# or
jupyter lab   # then select this notebook
```
### Launch tensorboard (local)

Working on local machine allows you to launch tensorboard as separate instance

```
mkdir runs
tensorboard --logdir runs
```

Then open address `127.0.0.1:6006` in web browser.

### Launch notebook copy in Google Colab (web)

https://colab.research.google.com/github/krglkvrmn/Apidaecin_AMP_search/blob/main/AMP_search_with_deep_learning_colab.ipynb

### Run scripts (web & local)

#### train_model.py

Train model on custom fasta files

```
usage: train_model.py [-h] --apidaecins_file APIDAECINS_FILE
                           [--not_apidaecins_file NOT_APIDAECINS_FILE]
                           --other_proteins_file OTHER_PROTEINS_FILE
                           --model_name MODEL_NAME
                           [--config CONFIG]
                           [--epochs EPOCHS]
                           [--save_dir SAVE_DIR]

options:
  -h, --help            show this help message and exit
  --apidaecins_file APIDAECINS_FILE
                        Fasta file containing preprocessed apidaecins sequences
  --not_apidaecins_file NOT_APIDAECINS_FILE
                        Fasta file containing high priority non-apidaecin sequences
  --other_proteins_file OTHER_PROTEINS_FILE
                        Fasta file containing low priority non-AMP protein sequences
  --model_name MODEL_NAME
                        Model name, which is also prefix for saved parameters file
  --config CONFIG       Config containing adjustable model and training parameters
  --epochs EPOCHS       Number of epochs to train on. Default: 10
  --save_dir SAVE_DIR   Directory to save trained model into. Default: 'models'
```

**Example:**

In this run *HybridModel* will be trained on 100 epochs using hyperparameters from **parameters.properties** file. ❗You may struggle configuring **parameters.properties** on Google Colab❗

```
python -m scripts.train_model --apidaecins_file data/pro-apidaecins.fasta \
                              --not_apidaecins_file data/not_apidaecins.fasta \
                              --other_proteins_file data/other_proteins.fasta \
                              --config parameters.properties \
                              --epochs 100 \
                              --model_name HybridModel \
                              --save_dir models
```

Script will create two files: **models/weights/HybridModel_vX.pt** containing model weights and **models/params/HybridModel_vX.pk** containing model hyperparameters.

#### scan_proteome.py

Predict labels for each amino acid in proteome

```
usage: scan_proteome.py [-h] --models_path MODELS_PATH --model_name MODEL_NAME
                        --proteome_path PROTEOME_PATH
                        [--scan_stride SCAN_STRIDE] --save_path SAVE_PATH

optional arguments:
  -h, --help            show this help message and exit
  --models_path MODELS_PATH
                        Path to directory with saved models (must contain
                        `weights` and `params` subdirectories)
  --model_name MODEL_NAME
                        Model name to load. `HybridModel` and `SimpleCNN` are
                        available. E.g. SimpleCNN_v2 or HybridModel
  --proteome_path PROTEOME_PATH
                        Path to fasta file with proteome
  --scan_stride SCAN_STRIDE
                        Step to scan proteome. Greater -> faster, less
                        accurate, lower -> slower, more accurate. Default: 20
  --save_path SAVE_PATH
                        File to save results into. Output is in tsv format
                        with fields: record_id, record_description, sequence,
                        prediction_mask
```

**Example_1:**

Command below takes the newest pretrained *HybridModel* found in **models** directory and use it to predict labels for proteome **data/proteomes/Megalopta_genalis.faa** with stride 20. By default, models scans proteome with given stride in order to lower execution time. If model predicts at least one fragment in protein, it will refine whole sequence with stride=1 to produce very accurate prediction.

```
python -m scripts.scan_proteome --models_path models \
                                --model_name HybridModel \
                                --proteome_path data/proteomes/Megalopta_genalis.faa \
                                --scan_stride 20 \
                                --save_path predictions/megalopta_genalis_results.tsv
```

The script produces summary logs about potentially found apidaecin-like AMP

```
# Sequence_id  positive_predictions/sequence_len  positive_pred_percent
XP_033332544.1                  399/423                 94.33%
XP_033332545.1                  371/395                 93.92%
XP_033332546.1                  371/395                 93.92%
XP_033332547.1                  343/367                 93.46%
XP_033341762.1                  45/491                  9.16%
XP_033335395.1                  195/2363                8.25%
XP_033336333.1                  162/1973                8.21%
XP_033335392.1                  195/2489                7.83%
XP_033335393.1                  188/2428                7.74%
XP_033335394.1                  182/2372                7.67%
XP_033335398.1                  171/2285                7.48%
XP_033340670.1                  95/1297                 7.32%
XP_033335397.1                  164/2324                7.06%
XP_033336331.1                  131/1863                7.03%
XP_033335396.1                  161/2333                6.90%
```

Saved file **predictions/megalopta_genalis_results.tsv** contain detailed information about predictions (only for proteins with at least 1 positive prediction). This is TSV file with fields: *record_id*, *record_description*, *sequence*, *prediction_mask*. *prediction_mask* is predicted labels for each amino acid of protein.

**Example_2:**

It is also possible to use specific version of model

```
python -m scripts.scan_proteome --models_path models \
                                --model_name HybridModel_v2 \
                                --proteome_path data/proteomes/Megalopta_genalis.faa \
                                --scan_stride 20 \
                                --save_path predictions/megalopta_genalis_results.tsv
```

## Contacts

If you have any questions, please contact @krglkvrmn - Telegram or kruglikov1911@mail.ru - Email
