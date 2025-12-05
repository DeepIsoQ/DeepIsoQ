
---------------------------
Location of pre-processed data:

/work3/s193518/scIsoPred/data

---------------------------
Initialization - transfer the data to the BLACKHOLE cluster (assuming access to DTU HPC).
Run these commands:

cd $BLACKHOLE
mkdir s215065 (replace s215065 with your student number or user)


Now you need to transfer the data to your BLACKHOLE directory. 
Run these commands:

cp /work3/s193518/scIsoPred/data/bulk_processed_genes.h5ad $BLACKHOLE/s215065  (replace s215065 with your student number or user)
cp /work3/s193518/scIsoPred/data/bulk_processed_transcripts.h5ad $BLACKHOLE/s215065 (replace s215065 with your student number or user)

---------------------------

Preprocessing data:
(!) Adjust your email and your environment path in the `data_preprocessing.lsf`

Run `data_preprocessing.py` once to create data.pt under $BLACKHOLE/$USER/.
bsub < DeepIsoQ/data_preprocessing.lsf

---------------------------

Running the FFNN: 

(!) Adjust your email and your environment path in the `run_ffnn_gpu.lsf`

Submit the GPU job:
bsub < ffnn_search/run_ffnn.lsf


The script will automatically use $BLACKHOLE/$USER/data.pt unless you override DATA_PT.

Please, beware that it should be run from the DeepIsoQ directory, otherwise it won't be able to find the directory.