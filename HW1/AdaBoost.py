from typing import List
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime
import math
import random
import bisect
import os

def gini_index(classffier_arr: np.ndarray):
    # determine which unique values the classffier_arry have and count how many from each class
    values, counts = np.unique(classffier_arr, return_counts=True)
    # calculate how many values all and all in the classfier
    S = len(classffier_arr)
    # calculate gini index equation
    gini_idx = 1 - ((counts / S) ** 2).sum()

    return gini_idx


def total_gini(attr_array: np.ndarray, splitter: float, classffier_arr: np.ndarray):
    # store all values above the splitter value
    above_splitter = classffier_arr[attr_array >= splitter]
    # store all values below the splitter value
    below_splitter = classffier_arr[attr_array < splitter]

    # calculate gini index for both groups
    above_splitter_gini = gini_index(classffier_arr=above_splitter)
    below_splitter_gini = gini_index(classffier_arr=below_splitter)
    # calculate how many values in the column
    S = len(classffier_arr)
    # calculate the totla gini of the column, based on the current splitter
    total = (len(above_splitter) / S) * above_splitter_gini + (len(below_splitter) / S) * below_splitter_gini

    return total


def best_gini_for_attr(attr_array, classfier_arr):
    # store the unique values of the data in a sorted array
    unique_attr_array = np.unique(attr_array)
    min_gini = 1
    min_splitter = 0

    # for every mean between 2 adjacent values, calculate total gini
    for i in range((len(unique_attr_array) - 1)):
        splitter = float(np.mean(unique_attr_array[i:i + 2]))
        temp_gini = total_gini(attr_array, splitter, classfier_arr)

        # if its the best total gini store it as min_gini
        if temp_gini < min_gini:
            min_gini = temp_gini
            min_splitter = splitter

    return min_gini, min_splitter


def choose_column_to_split_by(data_df: pd.DataFrame, classffier_arr: np.ndarray):
    best_col = ''
    splitter = 0
    min_gini = 1.1
    # for every attribute in the data frame
    for column in data_df.columns.values:
        # calculate the best gini for the current attribute
        temp_gini, temp_splitter = best_gini_for_attr(data_df[column].values, classfier_arr=classffier_arr)
        # if it is the best total gini-store it as min gini,
        # the best splitter for this attribute and the attribute's name
        if temp_gini < min_gini:
            min_gini = temp_gini
            splitter = temp_splitter
            best_col = column

    return min_gini, splitter, best_col


class Node:
    def __init__(
            self,
            data_df: pd.DataFrame,
            classffier_arr: np.ndarray,
            depth
    ):
        self.data_df = data_df
        self.classffier_arr = classffier_arr
        self.depth = depth
        self.column = None
        self.gini = None
        self.splitter = None
        # array for left and right children
        self.children = []
        # determine which unique values the classffier_arry have and count how many from each class
        values, counts = np.unique(self.classffier_arr, return_counts=True)
        # store and array of the unique values in the classfier
        self.values_names = values
        # store the count of each class
        self.values = counts
        # store the class with the most values
        self.cls = values[np.argmax(counts)]

    # if all the data is form 1 class
    def perfectly_classified(self):
        return len(self.values) == 1

    # gets only one row from the data
    def classify(self, data_row):
        # if the node is a leaf than classify
        # the current row as the majority class of the current node
        if len(self.children) == 0:
            return self.cls
        # store the current value of the column we are currently splitting by
        current_value = data_row[self.column]

        # if the current value is smaller than the node's splitter
        # go to the left child to classify
        # return the result of the left child
        if current_value < self.splitter and self.children[0] is not None:
            return self.children[0].classify(data_row)
        # if the current value is bigger or equals than the node's splitter
        # go to the right child to classify
        # return the result of the right child
        elif current_value >= self.splitter and self.children[1] is not None:
            return self.children[1].classify(data_row)
        else:
            # return the class of this node
            return self.cls


class Stump:
    def __init__(
            self, data_df: pd.DataFrame,
            classffier_arr: np.ndarray,
    ):
        self.data_df = data_df
        self.classffier_arr = classffier_arr
        self.childern = []
        self.root = None
        # stores the weight for each observation in the data frame
        self.weight_array = np.array([])
        # stores the stump's weight
        self.weight = None

    # function to fill the stumps with it's root and 2 leafs
    # and his properties
    def stump_trainer(self, data_df: pd.DataFrame, classffier_arr: np.ndarray):
        # create root node
        root_node = Node(data_df=data_df, classffier_arr=classffier_arr, depth=0)
        # calculate which attribute to split by-the one with the best gini will be chosen
        min_gini, splitter, best_col = choose_column_to_split_by(data_df=root_node.data_df,
                                                                 classffier_arr=root_node.classffier_arr)
        # store the values of the attribute to split data by
        root_node.column = best_col
        root_node.gini = min_gini
        root_node.splitter = splitter

        # The start weight for every observation will be
        # 1/the number of the observations in the data frame
        start_weight = 1 / len(data_df)
        # store in weight_array the start weight for every observation
        self.weight_array = np.full(len(data_df), start_weight)

        # Store the data that has smaller value from the splitter
        # and drop the current attribute from the data
        left_node_df = root_node.data_df[root_node.data_df[best_col] < splitter]
        left_node = None
        # if exists data for the left node(bigger than the splitter)
        if len(left_node_df) > 0:
            # create left node and store the data in it
            left_node = Node(
                data_df=left_node_df,
                classffier_arr=root_node.classffier_arr[root_node.data_df[best_col] < splitter],
                depth=root_node.depth + 1
            )

        # Store the  data that has bigger or equal value from the splitter
        right_node_df = root_node.data_df[root_node.data_df[best_col] >= splitter]
        right_node = None
        # if exists data for the right node(smaller than the splitter)
        if len(right_node_df) > 0:
            # create right node and store the data in it
            right_node = Node(
                data_df=root_node.data_df[root_node.data_df[best_col] >= splitter],
                classffier_arr=root_node.classffier_arr[root_node.data_df[best_col] >= splitter],
                depth=root_node.depth + 1
            )
        # store the left and right new nodes to the current node's children array
        root_node.children = [left_node, right_node]
        return root_node

    # function to calculate the total error of the stump
    def total_error(self):
        # initialize the number of errors to 0
        error = 0
        # store the stump's data frame in data_df
        data_df = self.data_df
        # store the stump's classffier_arr in classfier_arr
        classifier_arr = self.classffier_arr
        # as long as we have rows in the data frame
        for i in range(len(data_df)):
            row = data_df.iloc[i]
            # predict the class of the current row according to the stump's decision
            predicted = self.root.classify(row)
            # store the real class of the current row
            real = classifier_arr[i]

            # if the predition was wrong- count it
            if real != predicted:
                error += 1
        # return the total error for this stump
        return error / len(data_df)

    # function to update the weight of the data frame's observations
    def change_weight(self):
        # store the stump's data frame in data_df
        data_df = self.data_df
        # store the stump's classffier_arr in classfier_arr
        classifier_arr = self.classffier_arr
        # store the new weight for an observation correctly classified
        # according to the equation: new_weight=sample_weight*e**(-stump_weight)
        right_weight = self.weight_array[0] * math.e ** (-self.weight)
        # store the new weight for an observation incorrectly classified
        # according to the equation: new_weight=sample_weight*e**stump_weight
        wrong_weight = self.weight_array[0] * math.e ** self.weight
        # as long as we have rows in the data frame
        for i in range(len(data_df)):
            row = data_df.iloc[i]
            # predict the class of the current row according to the stump's decision
            predicted = self.root.classify(row)
            # store the real class of the current row
            real = classifier_arr[i]

            # if the prediction was wrong- increase weight
            if real != predicted:
                self.weight_array[i] = wrong_weight
            # if the prediction was right-decrease the weight
            else:
                self.weight_array[i] = right_weight
            # calculate the sum of the weights of all observations
        sum_weight = self.weight_array.sum()
        # normalize the weight of every observation by
        # divding the current observation's weight by the sum of the weights of all observations
        for i in range(len(self.weight_array)):
            self.weight_array[i] = self.weight_array[i] / sum_weight

    # function to create a new dataframe and his classifer to create from them a new stump
    def create_weighted_df(self):
        # Intialize an array to store the weights ranges
        ranges_array = np.array([])
        weight_sum = 0
        # for every weight of an observation
        for curr_weight in self.weight_array:
            # add the weight of the current observation to weight_sum
            weight_sum = weight_sum + curr_weight
            # append the new sum to the weight ranges array
            ranges_array = np.append(ranges_array, weight_sum)
        # create a new data frame with the same columns as the given data frame
        new_df = pd.DataFrame(columns=self.data_df.columns)
        # create a new empty classfier
        new_classfier = np.array([])
        # for every observation in the data frame
        for i in range(len(self.data_df)):
            # use a random number between 0 and 1
            # and save the index that should be for the random number if it was in the
            # weights array
            index = bisect.bisect_right(ranges_array, random.random())
            # add the index of this row to the new data frame
            # this step will create a data frame with the observations that have more weight
            # more times
            new_df = new_df.append(self.data_df.iloc[index], ignore_index=True)
            # add the class of the selected observation to the classifier array
            new_classfier = np.append(new_classfier, self.classffier_arr[index])
        # return the new data frame and his classifier to create a new stump from them
        return new_df, new_classfier


# define a class of adaboost
class Adaboost:
    def __init__(
            self,
            max_stumps
    ):
        # hyper parameter for the number of stumps
        self.max_stumps = max_stumps
        # an array to store the stumps of the booster
        self.stumps_array = []

    # function to create the stumps of the booster and store them in the booster's stumps array
    def adaboost_trainer(self, data_df: pd.DataFrame, classffier_arr: np.ndarray):
        # create the selected number of stumps
        for i in range(1, self.max_stumps):
            # create a new empty stump
            temp_stump = Stump(data_df, classffier_arr)
            # Create the stump's root and nodes
            temp_stump.root = temp_stump.stump_trainer(data_df, classffier_arr)
            # calculate the total error of the stump and store it in the stump
            temp_error = temp_stump.total_error()
            # calculate the weight of the sutmp according to totla error and store it in the stump
            temp_stump.weight = 0.5 * math.log((1 - temp_error) / temp_error)
            # update the weight of every observation in the stump
            temp_stump.change_weight()
            # add the new stump to the booster's array of stumps
            self.stumps_array.append(temp_stump)
            # create a new data frame and classffier to create a new stump
            # the new data frame will contain the observations with bigger weight
            # more times
            data_df, classffier_arr = temp_stump.create_weighted_df()

    # function to predict the class of the row according to the weights of the stumps
    def predict(self, data_row):
        # the sum of weights of stumps that chose 'B' class
        btype_sum = 0
        # the sum of weights of stumps that chose 'P' class
        ptype_sum = 0
        # f or every stump in the booster
        for stump in self.stumps_array:
            # predict the class with the current stump
            predicted = stump.root.classify(data_row)
            # add the weight of the stump to the sum of the result class
            if predicted == 'B':
                btype_sum += stump.weight
            else:
                ptype_sum += stump.weight
        # predict the class of the observation according to the class that has more weight
        # from all of the stumps
        if btype_sum <= ptype_sum:
            return 'P'
        return 'B'

    # function to predict the class of every observation and calculate the accuracy of the booster
    def inference(self, data_df: pd.DataFrame, calssifier_arr):
        correct = 0
        # as long as we have rows in the data frame
        for i in range(len(data_df)):
            row = data_df.iloc[i]
            # predict the class of the current row according to the tree's decision
            predicted = self.predict(row)
            # store the real class of the current row
            real = calssifier_arr[i]

            # if the prediction was correct- count it
            if real == predicted:
                correct += 1

        # calculate how many true predictions where made from all the data
        accuracy = correct / len(data_df)
        return accuracy

    # function to return an array of the predictions of every observation
    def results(self, data_df: pd.DataFrame):
        results_arr = []
        # as long as we have rows in the data frame
        for i in range(len(data_df)):
            row = data_df.iloc[i]
            # predict the class of the current row according to the booster's decision
            predicted = self.predict(row)
            results_arr.append(predicted)

        return results_arr


def best_booster(booster_A, booster_B, booster_C, accA, accB, accC):
    if accA >= accB and accA >= accC:
        return booster_A, accA
    if accB >= accA and accB >= accC:
        return booster_B, accB
    else:
        return booster_C, accC



def main():
    # get the path to the folder of the script
    folder_path = os.path.dirname(__file__)
    # store csv file name as a string
    csv_name = "data.csv"
    # create the path and store it
    csv_path = os.path.join(folder_path, csv_name)
    # Read data from csv file and store in real_estate
    real_estate = pd.read_csv(csv_path, index_col=[0])

    # store area_type column as classfier
    classfier_arr = np.array(real_estate['area_type'])
    # store data without the classfier
    real_estate_features = real_estate.drop('area_type', axis=1)
    # store train data
    train_df = real_estate_features.iloc[:8040]
    # store the classfier for train data
    train_cls_arr = classfier_arr[:8040]
    # store the validation data
    validation_df = real_estate_features.iloc[8041:10050]
    # store the validation classifer
    validation_cls_arr = classfier_arr[8041:10050]
    print("Validation:")

    ##############trail 1 with 5 stumps ##########
    start_time_A = datetime.now()
    # build Booster with 5 stumps
    booster_A = Adaboost(5)
    booster_A.adaboost_trainer(train_df, train_cls_arr)
    print("Booster with", booster_A.max_stumps, "stumps was built successfully!")
    accA = booster_A.inference(validation_df, validation_cls_arr)
    print("The accuracy of the Booster is:", accA)
    end_time_A = datetime.now()
    ##########################################################

    ##############trail 2 with 8 stumps ##########
    start_time_B = datetime.now()
    # build Booster with 8 stumps
    booster_B = Adaboost(8)
    booster_B.adaboost_trainer(train_df, train_cls_arr)
    print("Booster with", booster_B.max_stumps, "stumps was built successfully!")
    accB = booster_B.inference(validation_df, validation_cls_arr)
    print("The accuracy of the Booster is:", accA)
    end_time_B = datetime.now()
    ##########################################################

    ##############trail 1 with 10 stumps ##########
    start_time_C = datetime.now()
    # build Booster with 10 stumps
    booster_C = Adaboost(5)
    booster_C.adaboost_trainer(train_df, train_cls_arr)
    print("Booster with", booster_C.max_stumps, "stumps was built successfully!")
    accC = booster_C.inference(validation_df, validation_cls_arr)
    print("The accuracy of the Booster is:", accC)
    end_time_C = datetime.now()
    ##########################################################

    best_adaboost, best_accuracy = best_booster(booster_A, booster_B, booster_C, accA, accB, accC)
    best_accuracy = f"{best_accuracy:.3%}"
    print("The best Booster has", best_adaboost.max_stumps, "stumps, with accuracy:", best_accuracy)
    if best_adaboost == booster_A:
        print('run time: {}'.format(end_time_A - start_time_A))
    if best_adaboost == booster_B:
        print('run time: {}'.format(end_time_B - start_time_B))
    if best_adaboost == booster_C:
        print('run time: {}'.format(end_time_C - start_time_C))

    validation_results = best_adaboost.results(validation_df)
    # store the test data
    test_df = real_estate_features.iloc[10051:]
    # store the test classifer
    test_cls_arr = classfier_arr[10051:]
    # calculate accuracy of the test data based on the booster
    acc = best_adaboost.inference(test_df, test_cls_arr)
    acc = f"{acc:.3%}"
    print("Test:")
    print("The accuracy of the booster is:", acc)
    # store test results
    test_results = best_adaboost.results(test_df)




if __name__ == "__main__":
    main()
