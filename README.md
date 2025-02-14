#Biological data compression experiment
1. Modify the data path and run sampling.py to get hdf5 files for the mrc files.
2. Modify the datapath of z vector in train.py and data_dir in main.py.
3. Example code for training: python main.py --vae  --train --epoch 100.
4. You will get a z vector file as well as checkpoints for reconstruction
5. Modify the path and run reconstruction.py. The results are in output folder.
