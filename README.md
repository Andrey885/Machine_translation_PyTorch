# Machine translation with PyTorch
Basic hands-on experience in machine translation with PyTorch

In this repo the translator from German to English is trained and demonstrated

# Setup

You may use setup script:
```
sed -i 's/\r//g' setup.sh # edit line endings

bash setup.sh
```

or complete the installation manually in four steps:

 1. You'll need to install the required python packages:

```
python -m pip install -r requirements.txt
```

 2. Download and install pretrained Spacy language models:
```
$ sudo python -m spacy download en

$ sudo python -m spacy download de_core_news_sm
```

 3. Clone submodules
 '''
 git submodule init
 git submodule update
 '''
 
 4. Download a [pretrained model](https://drive.google.com/drive/u/0/folders/13qQTz_GgBbwokAsphghqryVmjV2SE4KI) and kindly put it in *checkpoints* folder.
 
# Dataset

Use [Multi30k](https://github.com/multi30k/dataset) translation dataset available from [PyTorch](https://torchtext.readthedocs.io/en/latest/datasets.html) - a
 small dataset from 2016 year challenge. The training is done on de-en part of it.
# Train model

You can train the model with this terminal command:

```
python train.py
```

This script will be saving models in ```./checkpoints/``` . It already contains some pretrained models.

# Results:

The model is capable of producing decent results on samples from test set, achieving 0.35 Bleu score on Multi30k dataset.
 This indicates nice level of perfomance (however, not as nice as state-of-the-art pretrained Bert models).

Run and see how it works:

```
python demo.py
```

Some sample results:

```
Input: eine straße neben einem interessanten ort mit vielen säulen .
GT translation: a road next to an interesting place with lots of pillars .
Model output: a street next to a plaza with many interesting pillars .

Input: ein skateboarder in einem schwarzen t-shirt und jeans fährt durch die stadt .
GT translation:  a skateboarder in a black t - shirt and jeans skating threw the city .
Model output:  a skateboarder in a black t - shirt and jeans is riding through the city .

```

# Source:

Tutorial with awesome model architectures:

https://github.com/bentrevett/pytorch-seq2seq

Paper with the description of **attention** model:

https://arxiv.org/abs/1706.03762

**Also useful tutorials**:

Nice short book to understand NLP basics (awful for production and demo, however):

https://github.com/joosthub/PyTorchNLPBook

Tutorials from good PyTorch folks, also nice and simple to get started:

https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
