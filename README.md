ADVI in Pymc3
===
This repository houses a final project for the Fall 2018 Harvard APMTH207 course. We selected two papers related to Bayesian artificial neural networks and made an exploratory and didactic tutorial, in the jupyter notebook `BBB_tutorial.ipynb`. More techically, we test the performance of ADVI in `pymc3` using the Fashion-MINIST dataset and an active learning task on this dataset. Note that in the `/staging` folder we have older versions of the notebooks that may also provide useful API references. We include instructions below to help get the code running:

To make the environment here use the following (assumes anaconda installation):

```
conda env create -f environment.yml
````

And, if you're worried about system pollution, it can just as easily be removed with:

```
conda env remove -n bbb_models
```

To install additional packages to the environment simply `conda activate bbb_models` to use the environment and use conda or pip to install as usual.  **NOTE:** If using jupyter notebook or jupyterlab use the following to add the kernel to your options:

```
conda install -n bbb_models nb_conda_kernels
```

This doesn't need to be done in the `bbb_models` environment, just the one from which you launch jupyterlab or jupyter notebook.

