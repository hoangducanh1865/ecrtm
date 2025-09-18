import argparse
import scipy
import numpy as np
import os
from src.evaluator.cluster import *
from src.evaluator.TD import *

class Evaluator:
    def __init__(self, args):
        self.args = args
    
    def evaluate(self):        

        test_theta = scipy.io.loadmat(self.args.path)['test_theta']
        pred = np.argmax(test_theta, axis=1)

        test_labels = np.loadtxt(self.args.label_path + '/test_labels.txt')

        clustering_metric(test_labels, pred)
        compute_TD(self.args.data_path)