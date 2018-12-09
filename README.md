# ADVI in Pymc3
We test the performance of ADVI in Pymc3 using the Fashion-MINIST

To make the environment here use the following (assumes anaconda installation):

```
conda env create -f environment.yml
````

And, if you're worried about system pollution, it can just as easily be removed with:

```
conda env remove -n bbb_models
```

To install additional packages to the environment simply activate the environment and use conda or pip to install as usual.

```
conda activate bbb_models
```

**NOTE:** If using jupyter notebook or jupyter lab use the following to add the kernel to your options:

```
conda install -n bbb_models nb_conda_kernels
```

