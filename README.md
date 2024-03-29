# Dynamics of a neuronal pacemaker in the weakly electric fish *Apteronotus*

This repository is the codebase for [Dynamics of a neuronal pacemaker in the weakly electric fish *Apteronotus*](https://www.nature.com/articles/s41598-020-73566-3) By Shifman *et. al.* (sci. rep.)

Please cite this as **Shifman, A.R., Sun, Y., Benoit, C.M. et al. Dynamics of a neuronal pacemaker in the weakly electric fish *Apteronotus*. Sci Rep 10, 16707 (2020)**

## Model details
The model can be viewed in its abstract form (parametric) in data/MODEL_PARAMETRIC

## Requirements

This work was done on Ubuntu 18.04 under WSL with python3.6.9.

Because we use *brian2* as our backend we recommend using a linux machine with a conda environment

To install requirements:

```
conda install -c conda-forge brian2==2.3.0.1
pip install -r requirements.txt
```

To compile the supplemental tables and equations you require a functional LaTeX install. 

## Run Analysis and Create Figures

This implementation ships with the model fits presented in the paper. The analysis will use this, to re-run the fitting see the subsequent section
```
python run_analysis.py
``` 

## Run Model Fitting

This will run 10 independent iterations of the model fitting for each of the four neuron recordings (warning: this will could take a very long time)
```
python run_fitting.py
```

## Creating Supplemental Tables

To create the supplemental tables, equations and figures please compile the LaTeX files `S_Equations.tex`, `S_Tables.tex`, and `S_Figure.tex`.
