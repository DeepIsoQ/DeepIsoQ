# Welcome to DeepIsoQ 👋
The content of this repository was the result of the hard work of all the team members to develop Project nr. 16 from DTU course 02456 Deep Learning: 

<h1 align="center">
Predicting Isoform Expression from Gene-Level Profiles using Representation Learning
</h1>

### Abstract

Despite the prevalence of the one-gene-one-protein model,
the process of alternative splicing can yield multiple isoforms
from the same gene, often with different functions and biological
relevance in different processes. Therefore, it is of
great interest to develop computational methods to measure
isoform expression. In this work, different architectures are
explored to predict isoform expression from gene expression
data, and the effect of different feature representation
methods in the model performance is analysed. The results
showed that, even if dimensionality reduction reveals useful
in reducing the computational workload, further hyperparameter
tuning and verification in new datasets is still necessary
to achieve better performance.

### Instructions on how to use this repository

This Git repository was designed to be cloned into the DTU HPC cluster. 
```bash
$ git clone https://github.com/DeepIsoQ/DeepIsoQ.git
```

Recommended Python and CUDA versions:
```bash
$ module load python3/3.10.18
$ module load cuda/12.8.1
```
(Either write this into the if-statement of the `.bashrc` file, or run it everytime when logging into the HPC). 

Alternatively, check other versions using:
```bash
module avail
```

Create a virtual environment:
```bash
python3 -m venv .venv
```

Then, activate the virtual environment with: 
```bash
source venv/bin/activate
```

Subsequently pip install relevant Python libraries (**note**: it might potentially also require the installation of a few others packages):
```bash
pip install torch
pip install anndata
pip install scanpy
```

**Important:** 
As a first step it is vital that you run the `data_preprocessing.py` script. 
If memory becomes an issue, be sure to use the job script, `data_preprocessing.lsf` using this command and after having validated and updated the content of the job script first:

```bash
bsub < data_preprocessing.lsf
```

(!) ***Warning***: The code in this repository was created by users with slightly different Python and venv configurations. <br>
Therefore, be sure to always double-check the location of the venv directory, etc. when running one of the lsf job scripts. 
<br>
<br>
<br>
----<br>
Enjoy, <br>
Group 39
