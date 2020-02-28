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
    return np.array(all_words, dtype='str')[:-1]

class Perceptron():
    # The perceptron for:
    w = 0         # Single
    w_voted = []    # Voted
    w_average = []   # Average
    label_as_one = 0
    
    def fit(self, training_data, num_passes, label_as_one):
        # Set up label mapping and initialize w
        self.label_as_one = label_as_one
        self.w = np.array([0] * (len(training_data[0])-1))
        # For voted
        cur_w_weight = 1
        self.w_voted = []
        self.w_average = []
        # How many epochs
        for cur_pass in range(num_passes):
            for data in training_data:
                # Transform label to 1/-1
                y = self.transform_label(data[-1])
                # Update case:
                if y * np.dot(data[:-1], self.w) <= 0:
                    self.w_voted.append([np.copy(self.w), cur_w_weight])
                    cur_w_weight = 1
                    self.w += y * data[:-1]
                else:
                    cur_w_weight += 1
        # Record w for single, voted, as well as average
        self.w_voted.append([np.copy(self.w), cur_w_weight])
        self.w_average = self.set_average_w()
            
    
    def transform_label(self, original_label): 
        if original_label == self.label_as_one:
            return 1
        else:
            return -1
    
    '''
    Functions for Prediction/Testing
    '''
    # Calculate Error
    def error(self, testing_data, method):
        # Depending on what method is, calculate predictions
        predictions = []
        if method == "single":
            predictions = self.predict_pass(testing_data)
        elif method == "voted":
            predictions = self.predict_voted(testing_data)
        elif method == "average":
            predictions = self.predict_average(testing_data)
        # Transform test label to 1/-1
        actual = []
        for original_label in testing_data[:,-1]:
            actual.append(self.transform_label(original_label))
        return np.mean(predictions != np.array(actual))
    
    
    # Single Perceptron
    def predict_pass_one(self, a_test_data):
        if np.dot(a_test_data, self.w) >= 0:
            return 1
        else:
            return -1
    
    def predict_pass(self, testing_data):
        return np.apply_along_axis(self.predict_pass_one, 1, testing_data[:,:-1])
    
    
    # Voted Perceptron
    def predict_voted_one(self, a_test_data):
        output = 0
        for pair in self.w_voted:
            prediction = np.dot(np.array(pair[0]), a_test_data) >= 0
            if prediction:
                output += pair[1]
            else:
                output -= pair[1]
        if output >=0:
            return 1
        else:
            return -1
    
    def predict_voted(self, testing_data):
        return np.apply_along_axis(self.predict_voted_one, 1, testing_data[:,:-1])
    
    # Average Perceptron
    def set_average_w(self):
        w_sum = np.array([0] * len(self.w))
        for pair in self.w_voted:
            w_sum += (pair[0] * pair[1])
        return w_sum
    
    def predict_average_one(self, a_test_data):
        if np.dot(a_test_data, self.w_average) >= 0:
            return 1
        else:
            return -1
        
    def predict_average(self, testing_data):
        return np.apply_along_axis(self.predict_average_one, 1, testing_data[:,:-1])