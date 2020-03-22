# pet_project
Hands on machine translation with PyTorch

# Setup

You'll need to install the required python packages:

```
python -m pip install -r requirements.txt
```

# Dataset

In this repo an eng-fra part of GloVe data for word embeddings training is used

- [Download whole GloVe](http://nlp.stanford.edu/data/glove.6B.zip)

# Train model:

You can train the model with this terminal command

```
python train.py
```

This script will save models in ```./checkpoints/``` . It already contains some pretrained models.

# Results:

Run and see how it works:

```
python demo.py
```

Some sample results:

```
Input:  vous etes celui qui m a entrainee .
GT translation:  you re the one who trained me .
Model translation:  you re the one who trained me .

Input: vous etes fort timide .
GT translation: you re very timid .
Model translation:  you re very timid .

Input:  tu trouves toujours des reproches a me faire .
GT translation: you re always finding fault with me .
Model translation:  you re always finding fault with me .

```

# To do

Train on some more datasets in order to increase word capacity

# Source:

https://github.com/joosthub/PyTorchNLPBook
