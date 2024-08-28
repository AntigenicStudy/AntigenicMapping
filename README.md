# Using pathogen genomics to predict antigenic changes of influenza H3N2 virus
This repository contains the framework code, data files, links to large data, and trained models for the paper "Using pathogen genomics to predict antigenic changes of influenza H3N2 virus" by Lin Wang, Sidan Yao et al.(2024).

### System requirements
The major Python package requirements and their tested versions are in [requirements.txt](requirements.txt)    
Our experiments were run with Python version 3.9 on Ubuntu 20.04.1.

### Installation guide
[pip](https://packaging.python.org/en/latest/key_projects/#pip) is the recommended installer for this environment.  

### Demo

The main results in our paper can be found in the [results/](results) directory and can be reproduced using the commands below.  
Trained models are in the [models/](models) directory, they are used for our analyses.  

** Embedding HA protein **
To calculate the embedding of a protein, please run this script given the example sequence, input fasta file and the trained BiLSTM model.
```
python code/flu-fasta.py examples/example_flu.fa aligned/h3-train.fa --checkpoint models/flu-h3.hdf5
```

### Calculate embedding distance
This step returns the embedding distance between a pair of proteins.
```
python code/embeddingDistance.py 'data/matrix' 'embed_dis/h1-pairwise.txt'
```

### Translate embedding distance into antigenic distance 
```
python code/regression.py 'data/distance' 'h1-dis.txt' 'data/embedding' 'h1_pairwise.txt' 'pred-output' 'losses-h1.txt' 'h1.mdl' 'y-pred-test.txt' 'y-gt-test.txt' 'y-pred-train.txt' 'y-gt-train.txt'
```

### Visualize results
```
python code/visualization.py 'pred-output' 'y-pred-test.txt' 'y-gt-test.txt' 'y-pred-train.txt' 'y-gt-train.txt' 'losses-h1.txt' 'visual-output' 'test-h1n1.png' 'train-h1n1.png'
```

## Data

### Amino acid sequences of influenza HA protein
To train the sequence processing model for each type of influenza, we retrieve more than 80,000 amino acid sequences of HA protein from H3 influenza from GISAID.

### Representative viruses for training ResNet
To train MLP for regression, we collect 279 representative H3 viruses with both HI measurements and full-length HA sequences

### HA sequence-based embedding distance
Please download HA sequence-based embedding distance between each pair of representative viruses from https://drive.google.com/drive/folders/1vQhz0SYSHiWYtGX0lGxcBH-btTNGB8rm?usp=sharing

### Antigenic distance 
Please find the antigenic distance between each pair of representative H3 viruses in this [folder](data/distance/)

## Pretrained models

### Sequence processing model built with BiLSTM
Please find pre-trained BiLSTM in this [folder](/models/trained_sequence_models) for each influenza type.

### MLP regressor
Please find our trained MLP regressor for H3 influenza in this [folder](/models/trained_regression_models) 


### Schematic of the model components for predicting antigenic change of H3N2 virus in each season.

![Image text](https://github.com/AntigenicStudy/AntigenicMapping/blob/main/img/Fig1_ABCD.jpg)

(A) BiLSTM-based protein language model was trained with 83,145 HA sequences of H3N2 viruses for high-dimensional encoding of genetic sequences. (B) Antigenic map provides a robust summary of the raw titre data, with antigenic drift of influenza virus interpreted as the movements of new strains in antigenic space. In the map, antigenic distance between two viruses represents their antigenic dissimilarity. (C) Multilayer perceptron (MLP) was trained using antigenic distance and sequence data of the mapped viruses from the past seasons to translate sequence-based embedding distance to antigenic distance. (D) Prediction of the antigenic distance between each new virus and each other virus was done by applying the trained MLP to their HA sequences. The viruses in each new season were then placed onto the antigenic map using multidimensional scaling (MDS) with input data of predicted and pre-existing antigenic distances.



