'''
Perceptron.py
Contains all methods for a perceptron algorithm/model.
Author: Zhanchong Deng
Date: 2/27/2020
'''
import numpy as np
import pandas as pd

def loadData(fp):
    newfile = open(fp, 'r')
    newfile.seek(0)
    raw_strings = newfile.read().split("\n")[:-1]
    return np.array([np.array(entry.split(" "), dtype="int") for entry in raw_strings])

def loadDictionary(fp):
    newfile = open(fp, 'r')
    newfile.seek(0)
    all_words = newfile.read().split("\n")
    return np.array(all_words, dtype='str')

class Perceptron():
    # The Core
    w = 0
    label_map = {}
    w_passes = {}
    w_voted = []
    
    def fit(self, training_data, num_passes):
        self.set_label_map(training_data)
        self.w = np.array([0] * (len(training_data[0])-1))
        # For voted
        cur_w_weight = 1
        # How many epochs
        for cur_pass in range(num_passes):
            for data in training_data:
                # Transform label to 1/-1
                y = self.label_map[data[-1]]
                # Update case:
                if y * np.dot(data[:-1], self.w) <= 0:
                    self.w_voted.append([self.w, cur_w_weight])
                    cur_w_weight = 1
                    self.w += y * data[:-1]
                else:
                    cur_w_weight += 1
            # Record w for later usage
            self.w_passes[cur_pass+1] = self.w
                    
    def predict_pass_one(self, a_test_data):
        if np.dot(a_test_data, self.w) > 0:
            return 1
        else:
            return -1
    
    def predict_pass(self, testing_data, version):
        self.w = self.w_passes[version]
        return np.apply_along_axis(self.predict_pass_one, 1, testing_data[:,:-1])
    
    def error(self, testing_data, version):
        predictions = self.predict_pass(testing_data, version)
        # Transform test label to 1/-1
        actual = []
        for original_label in testing_data[:,-1]:
            actual.append(self.label_map[original_label])
        return np.mean(predictions != np.array(actual))
    
    def set_label_map(self, training_data): 
        unique_labels = np.unique(training_data[:,-1])
        if len(unique_labels) != 2:
            print("More than two labels")
            return;
        else:
            self.label_map[unique_labels[0]] = 1
            self.label_map[unique_labels[1]] = -1