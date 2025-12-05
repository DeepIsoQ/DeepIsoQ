# DeepIsoQ
Deep Learning Project 16: Predicting Isoform Expression from Gene-Level Profiles using Representation Learning

The central dogma of molecular biology states that DNA is transcribed into RNA, which is then translated into proteins. While a one-gene–one-protein model is often assumed, the process of alternative splicing at the RNA level can result in multiple protein isoforms with distinct biological functions from the same gene. Hence gene expression data only provides a coarse-grained view of transcriptional activity and in many cases disease and cell-type-specific behaviour is determined by which isoforms are expressed. However, while gene expression data is abundant, measuring isoform expression is technically challenging and costly, especially in single-cell experiments. Therefore, the goal of this project is to develop a computational method for accurately predicting isoform expression from gene expression alone. If successful, this would unlock isoform-level insights for millions of existing RNA-seq samples.

-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

This code Git repository was designed to be cloned into the DTU HPC cluster. <br>
`git clone https://github.com/DeepIsoQ/DeepIsoQ.git`
<br>
<br>

**Initialization**
<br>

Recommended Python and CUDA versions: <br>
`module load python3/3.10.18` <br>
`module load cuda/12.8.1` <br>
(Either write this into the if-statement of the `.bashrc` file, or run it everytime when logging into the HPC). <br>
<br>
Or check other versions using: <br>
`module avail` <br>
<br>
Create a virtual environment: <br>
`python3 -m venv .venv` <br>
<br>
Subsequently pip install relevant Python libraries: <br>

`pip install torch` <br>
`pip install anndata` <br>
`pip install scanpy` <br>
(Might potentially also require the installation of a few others packages). <br>
<br>
<br>

**Important:** <br>
As a first step it is vital that you run the `data_preprocessing.py` script. <br>
If memory becomes an issue, be sure to use the job script, `data_preprocessing.lsf` <br>
Using this command: <br>
`bsub < data_preprocessing.lsf`
<br>
(After having validated and updated the content of the job script first).
<br>

(!) Warning: The code in this repository was created by users with slightly different Python and venv configurations. <br>
Therefore, be sure to always double-check the location of the venv directory, etc. when running one of the lsf job scripts. 
<br>
<br>
Enjoy, <br>
Group 39