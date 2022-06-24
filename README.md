# Deep learning approach for search of novel antimicrobial peptides

## Purpose and objectives of the project

### Purpose:

+ Develop an algorithm, which can perform search of antimicrobial peptides (AMP) coding genes in genomes and proteomes of various organisms.

### Objectives:

+ Prepare training data
+ Develop and train neural network model that can classify whether protein sequence belong to apidaecin-like AMP
+ Select optimal model architecture and data processing methods
+ Validate obtained model on real-world data

## Data description

All data used in this project is stored in **data** directory. This includes:

+ **data/apidaecins.fasta**  &mdash; unfiltered apidaecins sequences, obtained from NCBI database
+ **data/pro-apidaecins.fasta** &mdash; filtered and processed apidaecins sequences. Processing included removal of signal peptides
+ **data/APD_DB.fasta** &mdash; all AMP sequences downloaded from [APD3 database](https://aps.unmc.edu/)
+ **data/not_api_proteins.fasta** &mdash; proteins, which are not related to AMP or any other antibiotics

Nested cross validation results are stored as python pickles in **pickles** directory.

Models directory contain pretrained models. Files **models/classes/*.py** contain all variables, that you may need to load the model. Files **models/weights/*.pt** contain saved model parameters.

The newest and the best performing model called **HybridModel_3a**.

## Workflow overview

Detailed step-by-step and easy-to-reproduce workflow is available in notebook **AMP_search_with_deep_learning.ipynb**. This includes: 

+ Definitions of all functions, classes and models
+ Data processing
+ Models training
+ Hyperparameters optimization and validation via nested cross-validation
+ Cross-validation on proteomic data.

## Usage

### Download source code and install dependencies

```
git clone https://github.com/krglkvrmn/Apidaecin_AMP_search.git
cd Apidaecin_AMP_search
pip install -r requirements.txt
```

All code was run on **python 3.7.9**. Correct work on other versions is not guaranteed.

### Open notebook

```
jupyter notebook AMP_search_with_deep_learning.ipynb
# or
jupyter lab   # then select this notebook
```
### Or open in Google Colab

https://colab.research.google.com/github/krglkvrmn/Apidaecin_AMP_search/blob/main/AMP_search_with_deep_learning_colab.ipynb

