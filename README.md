#### What is this?

This is a partial rewrite of [hugochan/Graph2Seq-for-KGQG](https://github.com/hugochan/Graph2Seq-for-KGQG)
using a version of PyTorch that's modern for 2024.

> [!WARNING]
> The code in this repository is for educational purposes only and is not guaranteed to run without error or produce correct results. All credit is due to the original authors.

#### Config
This is very work-in-progress, but the project is managed with
Conda and comes with a Conda `environment.yml` file. Make a
reproducible Conda environment with:
```shell
conda env create -n <name> -f environment.yml
```
Before running `main.py`, download and unzip [GloVe embeddings](http://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip)
into `data/`. Also download training and metrics data from [here](https://1drv.ms/u/s!AjiSpuwVTt09gVsFilSx0NpJlid-?e=1TKqfG)
and copy all files into `data/` as well.

Runs on Python 3.11 and requires GPU support.