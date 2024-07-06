# minimodel

Code to train and test models from paper:

Fengtong Du, Miguel Angel Núñez-Ochoa, Marius Pachitariu, Carsen Stringer. [Towards a simplified model of primary visual cortex](https://www.biorxiv.org/content/10.1101/2024.06.30.601394v1). bioRxiv.

## Setup

Before running the code, please install the minimodel module by running the following command in your terminal under the  root directory:

```sh
python setup.py install
```

If you want to use GPU to train the models, you need to uninstall the current CPU version of PyTorch and install the GPU version. First, run:
```sh
pip uninstall torch
```

Then, install the GPU version of PyTorch using conda:
```sh
conda install pytorch==1.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```

## Project Structure

The project structure is organized as follows:

- `data/`: Directory for storing image and recording files.
- `minimodel/`: Directory containing the main minimodel module code.
- `notebooks/`: Directory for Jupyter notebooks of training and testing models.
- `setup.py`: Setup script for installing the minimodel module.

## Monkey dataset
The data we used to train the monkey models is from the [paper](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006897) by Cadena, et al. For loading the monkey data, please check this github page ([link](https://github.com/sacadena/Cadena2019PlosCB)).

Use `data/monkey_data.ipynb` to convert the monkey data to the format required by our models.


## Sensorium Model

To test the performance of the CNN Baseline model from the [paper](https://arxiv.org/abs/2206.08666) by Willeke, et al. on our dataset, follow these steps:

1. Use the notebook `data/sensorium_format.ipynb` to convert our data to the format required by the Sensorium competition.
2. Use the model training notebook from the Sensorium competition available at [this link](https://github.com/sinzlab/sensorium/blob/main/notebooks/model_tutorial/1a_model_training_sensorium.ipynb) to train the model.




