# Antigenic Variation Prediction
We develop a two-stage deep learning based framework to predict the antigenic distance between pairwise influenza viruses using their hemagglutinin (HA) protein sequences. In the first stage, for each type of influenza virus, we encode HA sequences in a high-dimensional continuous space by training a sequence processing model using BiLSTM with large-volume HA sequences retrieved from GISAID, the global influenza data initiative. In the second stage, for each influenza subtype/lineage, we train a ResNet to predict the antigenic distance between pairwise viruses using their HA sequence-based embedding distance.

## Data

### Amino acid sequences of influenza HA protein
To train the sequence processing model for each type of influenza, we retrieve large-volume amino acid sequences of HA protein from that type of influenza from GISAID.

### Representative viruses for training ResNet
To train ResNet-18 for each influenza subtype/lineage, we collect representative viruses of the same subtype/lineage with both HI measurements and full-length HA sequences

### HA sequence-based embedding distance
Please download HA sequence-based embedding distance between each pair of representative viruses from https://drive.google.com/drive/folders/1vQhz0SYSHiWYtGX0lGxcBH-btTNGB8rm?usp=sharing

### Antigenic distance 
Please find the antigenic distance between each pair of representative viruses for each influenza subtype/lineage in this [folder](data/distance/)

## Pretrained models

### Sequence processing model built with BiLSTM
Please find pre-trained BiLSTM in this [folder](/models/trained_sequence_models) for each influenza type.

### ResNet-18
Please find our trained ResNet-18 for each influenza subtype/lineage in this [folder](/models/trained_regression_models) 

## Usage
### Train BiLSTM
```
python code/trainBiLSTM.py 
```
### Embedding HA protein
```
python code/calculateProtEmbedding.py 
```

### Calculate embedding distance
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

![Image text](https://github.com/AntigenicStudy/AntigenicMapping/blob/main/img/pipeline.PNG)
![Image text](https://github.com/AntigenicStudy/AntigenicMapping/blob/main/img/train-h1.png)
![Image text](https://github.com/AntigenicStudy/AntigenicMapping/blob/main/img/test-h1.png)