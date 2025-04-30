# 🧠 minimodel

Code to train and test models from the paper:  **Fengtong Du, Miguel Angel Núñez-Ochoa, Marius Pachitariu, Carsen Stringer.** *[A simplified minimodel of visual cortical neurons](https://www.biorxiv.org/content/10.1101/2024.06.30.601394v1)*, bioRxiv


## Installation

We recommend using a **clean Python environment** to avoid dependency conflicts.

Create and activate a new environment:
```bash
conda create -n minimodel-env python=3.10 -y
conda activate minimodel-env
```

Clone the repository:
```bash
git clone https://github.com/MouseLand/minimodel.git
cd minimodel
```

Install the required dependencies and minimodel module:
```bash
pip install -e .
```
## Example notebook
[mouse_pipeline.ipynb](https://github.com/MouseLand/minimodel/blob/main/notebooks/mouse_pipeline.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/dufengtong/minimodel/blob/master/notebooks/mouse_pipeline.ipynb)
: This notebook demonstrates how to use the minimodel module to train and test models on the mouse dataset. It includes data loading, fullmodel training and evaluation, and minimodel training and evaluation.



## Project Structure

The project structure is organized as follows:


- `minimodel/`: Directory containing the main minimodel module code.
- `notebooks/`: Directory for Jupyter notebooks of training and testing models.
    - `data/`: Directory for storing image and recording files.
    - `checkpoints/`: Directory for storing model checkpoints.
- `setup.py`: Setup script for installing the minimodel module.

## Datasets

### Mouse dataset
You can download the mouse dataset from [Janelia's Figshare](https://janelia.figshare.com/articles/dataset/Towards_a_simplified_model_of_primary_visual_cortex/28797638) or use the following command:
```bash
wget https://janelia.figshare.com/ndownloader/articles/28797638/versions/2
```
This will download a zip file containing the mouse dataset. Unzip the file and place it in the `notebooks/data/` directory.

### Monkey dataset
The data we used to train the monkey models is from the [paper](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006897) by Cadena, et al. For loading the monkey data, please check this github page ([link](https://github.com/sacadena/Cadena2019PlosCB)).

Use `notebook/data/monkey_data.ipynb` to convert the monkey data to the format required by our models.


### Sensorium Model

To test the performance of the CNN Baseline model from the [paper](https://arxiv.org/abs/2206.08666) by Willeke, et al. on our dataset, follow these steps:

1. Use the notebook `notebook/data/sensorium_format.ipynb` to convert our data to the format required by the Sensorium competition.
2. Use the model training notebook from the Sensorium competition available at [this link](https://github.com/sinzlab/sensorium/blob/main/notebooks/model_tutorial/1a_model_training_sensorium.ipynb) to train the model.

