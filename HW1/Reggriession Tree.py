from typing import List
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime
import os

def ssr_score(attr_array: np.ndarray, splitter: float, classfier_arr: np.ndarray):
    # if the data frame is empty, then the sum is 0
    if len(classfier_arr[attr_array >= splitter]) == 0:
        sum_above_splitter = 0
    else:
        # calculate the mean of all values above the splitter value
        mean_above_splitter = classfier_arr[attr_array >= splitter].mean()
        # sum the square residuals(real attribute value-mean of above split values) of all elements in the array
        # that are bigger than the splitter
        sum_above_splitter = ((classfier_arr[attr_array >= splitter] - mean_above_splitter) ** 2).sum()
    # if the data frame is empty, then the sum is 0
    if len(classfier_arr[attr_array < splitter]) == 0:
        sum_below_splitter = 0
    else:
        # calculate the mean of below the splitter value
        mean_below_splitter = classfier_arr[attr_array < splitter].mean()
        # sum the square residuals(real attribute value-mean of below split values) of all elements in the array
        # that are smaller than the splitter
        sum_below_splitter = ((classfier_arr[attr_array < splitter] - mean_below_splitter) ** 2).sum()

    # return SSR score for this split
    return sum_above_splitter + sum_below_splitter


def best_ssr_for_attr(attr_array, classfier_arr):
    # store the unique values of the data in a sorted array
    unique_attr_array = np.unique(attr_array)
    min_ssr = None
    min_splitter = None

    if len(unique_attr_array) == 1:
        return ssr_score(attr_array, unique_attr_array[0], classfier_arr), unique_attr_array[0]

    # for every mean between 2 adjacent values, calculate ssr
    for i in range((len(unique_attr_array) - 1)):
        temp_splitter = float(np.mean(unique_attr_array[i:i + 2]))
        temp_ssr = ssr_score(attr_array, temp_splitter, classfier_arr)

        # if it is the first ssr calculated-store it as min ssr
        if min_ssr is None:
            min_ssr = temp_ssr
            min_splitter = temp_splitter
        # if its ssr store it as min_ssr
        if temp_ssr < min_ssr:
            min_ssr = temp_ssr
            min_splitter = temp_splitter

    return min_ssr, min_splitter


def choose_column_to_split_by(data_df: pd.DataFrame, classffier_arr: np.ndarray):
    best_col = ''
    splitter = 0
    min_ssr = None
    # for every attribute in the data frame
    for column in data_df.columns.values:
        # calculate the best ssr for the current attribute
        temp_ssr, temp_splitter = best_ssr_for_attr(data_df[column].values, classfier_arr=classffier_arr)
        # if it is the best ssr-store it as min ssr,
        # best splitter for this attribute and the attribute's name
        if min_ssr is None:
            min_ssr = temp_ssr
            splitter = temp_splitter
            best_col = column

        if temp_ssr < min_ssr:
            min_ssr = temp_ssr
            splitter = temp_splitter
            best_col = column
    return min_ssr, splitter, best_col


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
        self.ssr = None
        self.splitter = None
        # array for left and right children
        self.children = []
        self.mean_price = self.classffier_arr.mean()

    # gets only one row from the data
    def classify(self, data_row):
        # if the node is a leaf than classfiy
        # the current row as the majority class of the current node
        if len(self.children) == 0:
            return self.mean_price
        # store the current value of the column we are currently splitting by
        current_value = data_row[self.column]

        # if the current value is smaller than the node's splitter
        # go to the left child to classify
        # return the result of the left child
        if current_value < self.splitter and self.children[0] is not None:
            return self.children[0].classify(data_row)
        # if the current value is bigger or equal than the node's splitter
        # go to the right child to classify
        # return the result of the right child
        elif current_value >= self.splitter and self.children[1] is not None:
            return self.children[1].classify(data_row)
        else:
            # return the class of this node
            return self.mean_price


class Regression_tree:
    def __init__(self,
                 min_split_size, depth):
        self.depth = depth
        self.min_split_size = min_split_size
        self.root = None

    def dt_trainer(self, data_df: pd.DataFrame, classffier_arr: np.ndarray):
        # Initialize a queue for the nodes of the tree
        q_nodes: List[Node] = []
        # create root node
        self.root = Node(data_df=data_df, classffier_arr=classffier_arr, depth=0)
        # add root node to the queue
        q_nodes.append(self.root)
        # as long as we have nodes in the queue
        while len(q_nodes) > 0:
            # get first node in the queue
            node = q_nodes.pop(0)

            # check if the node is a leaf or data frame is empty
            if len(node.data_df) < self.min_split_size or node.depth == self.depth:
                continue
            # calculate which attribute to split by-the one with the best ssr will be chosen
            min_ssr, splitter, best_col = choose_column_to_split_by(data_df=node.data_df,
                                                                    classffier_arr=node.classffier_arr)

            # store the values of the attribute to split data by
            node.column = best_col
            node.ssr = min_ssr
            node.splitter = splitter

            # Store the data that has bigger value from the splitter
            # and drop the current attribute from the data

            left_node_df = node.data_df[node.data_df[best_col] < splitter]
            left_node = None
            # if exists data for the left node(bigger than the splitter)
            if len(left_node_df) > 0:
                # create left node and store the data in it
                left_node = Node(
                    data_df=left_node_df,
                    classffier_arr=node.classffier_arr[node.data_df[best_col] < splitter],
                    depth=node.depth + 1
                )
                # add the new node to the queue
                q_nodes.append(left_node)

            # Store the  data that has smaller value from the splitter
            # and drop the current attribute from the data
            right_node_df = node.data_df[node.data_df[best_col] >= splitter]
            right_node = None
            # if exists data for the right node(smaller than the splitter)
            if len(right_node_df) > 0:
                # create right node and store the data in it
                right_node = Node(
                    data_df=node.data_df[node.data_df[best_col] >= splitter],
                    classffier_arr=node.classffier_arr[node.data_df[best_col] >= splitter],
                    depth=node.depth + 1

                )
                # add the new node to the queue
                q_nodes.append(right_node)
                # store the left and right new nodes to the current node's children array
            node.children = [left_node, right_node]

    def dt_inference(self, dt: Node, data_df: pd.DataFrame, calssifier_arr):
        sum = 0
        # as long as we have rows in the data frame
        for i in range(len(data_df)):
            row = data_df.iloc[i]
            # predict the price of the current row according to the tree's decision
            predicted = dt.classify(row)
            # calculate the difference between the real price and the predicted price
            # and square it
            sqr_residual = (calssifier_arr[i] - predicted) ** 2
            sum += sqr_residual
        # calculate mse by divide the total differences sum in the number of observations
        mse = sum / len(data_df)
        # calculate how many true predictions where made from all the data

        return mse

    def dt_results(self, dt: Node, data_df: pd.DataFrame):
        results_arr = []

        # as long as we have rows in the data frame
        for i in range(len(data_df)):
            row = data_df.iloc[i]
            # predict the price of the current row according to the tree's decision
            predicted = dt.classify(row)
            results_arr.append(predicted)

        return results_arr


def best_reg_tree(reg_tree_A, reg_tree_B, reg_tree_C, mseA, mseB, mseC):
    if mseA <= mseB and mseA <= mseC:
        return reg_tree_A, mseA
    if mseB <= mseA and mseB <= mseC:
        return reg_tree_B, mseB
    else:
        return reg_tree_C, mseC


def main():
    # get the path to the folder of the script
    folder_path = os.path.dirname(__file__)
    # store csv file name as a string
    csv_name = "data.csv"
    # create the path and store it
    csv_path = os.path.join(folder_path, csv_name)
    # Read data from csv file and store in real_estate
    real_estate = pd.read_csv(csv_path, index_col=[0])

    # store area_type column as classifier
    classfier_arr = np.array(real_estate['price in rupees'])
    # store data without the classifier
    real_estate_features = real_estate.drop('price in rupees', axis=1)
    real_estate_features.loc[(real_estate_features.area_type == 'B'), "area_type"] = 0
    real_estate_features.loc[(real_estate_features.area_type == 'P'), "area_type"] = 1
    real_estate_features['area_type'] = pd.to_numeric(real_estate_features['area_type'])
    # store train data
    train_df = real_estate_features.iloc[:8040]
    # store the classifier for train data
    train_price_arr = classfier_arr[:8040]
    # store the validation data
    validation_df = real_estate_features.iloc[8041:10050]
    # store the validation classifer
    validation_cls_arr = classfier_arr[8041:10050]
    print("Validation:")
    ###########trail 1 with minimum 20 observations and maximu depth=4
    # build decision tree based on training data
    start_time_A = datetime.now()
    reg_tree_A = Regression_tree(20, 4)
    reg_tree_A.dt_trainer(train_df, train_price_arr)
    print("Tree was built successfully!")
    mseA = reg_tree_A.dt_inference(reg_tree_A.root, validation_df, validation_cls_arr)
    print("The MSE of the tree is:", mseA)
    end_time_A = datetime.now()
    ##########################################################

    ###########trail 2 with minimum 20 observations and maximu depth=8
    # build decision tree based on training data
    start_time_B = datetime.now()
    reg_tree_B = Regression_tree(20, 8)
    # store the validation data
    reg_tree_B.dt_trainer(train_df, train_price_arr)
    print("Tree was built successfully!")
    mseB = reg_tree_B.dt_inference(reg_tree_B.root, validation_df, validation_cls_arr)
    print("The MSE of the tree is:", mseB)
    end_time_B = datetime.now()
    ##########################################################

    ###########trail 3 with minimum 20 observations and maximu depth=8
    # build decision tree based on training data
    start_time_C = datetime.now()
    reg_tree_C = Regression_tree(20, 8)
    # store the validation data
    reg_tree_C.dt_trainer(train_df, train_price_arr)
    print("Tree was built successfully!")
    mseC = reg_tree_C.dt_inference(reg_tree_C.root, validation_df, validation_cls_arr)
    print("The MSE of the tree is:", mseC)
    end_time_C = datetime.now()
    ##########################################################

    best_tree, best_mse = best_reg_tree(reg_tree_A, reg_tree_B, reg_tree_C, mseA, mseB, mseC)
    print("The best regression Tree is has", best_tree.depth, "depth, with mse:", best_mse)
    if best_tree == reg_tree_A:
        print('run time: {}'.format(end_time_A - start_time_A))
    if best_tree == reg_tree_B:
        print('run time: {}'.format(end_time_B - start_time_B))
    if best_tree == reg_tree_C:
        print('run time: {}'.format(end_time_C - start_time_C))

    print("Test:")
    validation_results = best_tree.dt_results(best_tree.root, validation_df)
    # store the test data
    test_df = real_estate_features.iloc[10051:]
    # store the test classifer
    test_price_arr = classfier_arr[10051:]
    test_mse = best_tree.dt_inference(best_tree.root, test_df, test_price_arr)
    test_results = best_tree.dt_results(best_tree.root, test_df)


if __name__ == "__main__":
    main()
