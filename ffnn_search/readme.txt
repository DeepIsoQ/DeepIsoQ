Run data_preprocessing.py once to create data.pt under $BLACKHOLE/$USER/.

Adjust your email and your environment path in the run_ffnn_gpu.lsf

Submit the GPU job:
bsub < ffnn_search/run_ffnn_gpu.lsf

The script will automatically use $BLACKHOLE/$USER/data.pt unless you override DATA_PT.

Please, beware that it should be run from the DeepIsoQ directory, otherwise it won't be able to find the directory.