import os
import numpy as np
import argparse
from train import IM_AE
from Binary_train import OccupancyTrainer

os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
parser = argparse.ArgumentParser()
parser.add_argument("--epoch", action="store", dest="epoch", default=0, type=int, help="Epoch to train [0]")
parser.add_argument("--learning_rate", action="store", dest="learning_rate", default=0.00005, type=float, help="Learning rate for adam [0.00005]")
parser.add_argument("--beta1", action="store", dest="beta1", default=0.5, type=float, help="Momentum term of adam [0.5]")
parser.add_argument("--dataset", action="store", dest="dataset", default="density", help="The name of dataset")
parser.add_argument("--checkpoint_dir", action="store", dest="checkpoint_dir", default="checkpoint", help="Directory name to save the checkpoints [checkpoint]")
parser.add_argument("--data_dir", action="store", dest="data_dir", default="/home/zcy/seperate_VAE", help="Root directory of dataset [data]")

# parser.add_argument("--load_checkpoint", action="store_true", dest="load_checkpoint", default=False, help="True to load existing checkpoint [False]")
parser.add_argument("--train", action="store_true", dest="train", default=False, help="True for training, False for testing [False]")
parser.add_argument("--ae", action="store_true", dest="ae", default=False, help="True for ae [False]")
parser.add_argument("--binary", action="store_true", dest="binary", default=False, help="True for binary occupancy model [False]")
parser.add_argument("--load_checkpoint", action="store_true", dest="load_checkpoint", default=False, help="True to load existing checkpoint [False]")

FLAGS = parser.parse_args()

if FLAGS.train:
    # if FLAGS.binary:
    #     # Train binary occupancy model
    #     trainer = OccupancyTrainer(FLAGS)
    #     trainer.train(FLAGS)
    if FLAGS.ae:
        # Train density prediction model
        im_svae = IM_AE(FLAGS)
        im_svae.train(FLAGS)
    else:
        print("Please specify an operation: --ae or --binary")
else:
    print("Please specify --train to train a model")