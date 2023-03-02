from typing import List
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime
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
        # best splitter for this attribute and the attribute's name
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

    # Function to predict the class of an observation in the data
    # gets only one row from the data
    def classify(self, data_row):
        # if the node is a leaf than classfiy
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
        # if the current value is bigger or equals to the node's splitter
        # go to the right child to classify
        # return the result of the right child
        elif current_value >= self.splitter and self.children[1] is not None:
            return self.children[1].classify(data_row)
        else:
            # return the class of this node
            return self.cls


class dt_tree:
    def __init__(self, depth
                 ):
        self.root = None
        self.depth = depth

    def dt_trainer(self, data_df: pd.DataFrame, classffier_arr: np.ndarray):
        # Initialize a queue for the nodes of the tree
        q_nodes: List[Node] = []
        # create root node
        root_node = Node(data_df=data_df, classffier_arr=classffier_arr, depth=0)
        # add root node to the queue
        q_nodes.append(root_node)
        # as long as we have nodes in the queue
        while len(q_nodes) > 0:
            # get first node in the queue
            node = q_nodes.pop(0)
            # check if the node is a leaf(contains only one class) or data frame is empty
            if node.perfectly_classified() or len(node.data_df.columns) == 0 or node.depth == self.depth:
                # calculate gini score for the current data
                node.gini = gini_index(node.classffier_arr)
                # continue to the next node
                continue
            # calculate which attribute to split by-the one with the best gini will be chosen
            min_gini, splitter, best_col = choose_column_to_split_by(data_df=node.data_df,
                                                                     classffier_arr=node.classffier_arr)

            # store the values of the attribute to split data by
            node.column = best_col
            node.gini = min_gini
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

        return root_node

    def dt_inference(self, dt: Node, data_df: pd.DataFrame, calssifier_arr):
        correct = 0
        # as long as we have rows in the data frame
        for i in range(len(data_df)):
            row = data_df.iloc[i]
            # predict the class of the current row according to the tree's decision
            predicted = dt.classify(row)
            # store the real class of the current row
            real = calssifier_arr[i]

            # if the predition was corret- count it
            if real == predicted:
                correct += 1

        # calculate how many true predictions where made from all the data
        accuracy = correct / len(data_df)
        return accuracy

    def dt_results(self, dt: Node, data_df: pd.DataFrame):
        results_arr = []

        # as long as we have rows in the data frame
        for i in range(len(data_df)):
            row = data_df.iloc[i]
            # predict the class of the current row according to the tree's decision
            predicted = dt.classify(row)
            results_arr.append(predicted)

        return results_arr


def best_class_tree(class_tree_A, class_tree_B, class_tree_C, accA, accB, accC):
    if accA >= accB and accA >= accC:
        return class_tree_A, accA
    if accB >= accA and accB >= accC:
        return class_tree_B, accB
    else:
        return class_tree_C, accC


def main():
    # get the path to the folder of the script
    folder_path = os.path.dirname(__file__)
    # store csv file name as a string
    csv_name = "data.csv"
    # create the path and store it
    csv_path = os.path.join(folder_path, csv_name)
    # Read data from the csv file and store in real_estate
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
    ##############trail 1 with depth 5 ##########
    start_time_A = datetime.now()
    # build decision tree with depth 5
    class_tree_A = dt_tree(5)
    class_tree_A.root = class_tree_A.dt_trainer(train_df, train_cls_arr)
    print("Tree with", class_tree_A.depth, "was built successfully!")
    accA = class_tree_A.dt_inference(class_tree_A.root, validation_df, validation_cls_arr)
    print("The accuracy of the tree is:", accA)
    end_time_A = datetime.now()
    ##########################################################

    ##############trail 2 with depth 10 ##########
    start_time_B = datetime.now()
    # build decision tree with depth 10
    class_tree_B = dt_tree(10)
    class_tree_B.root = class_tree_B.dt_trainer(train_df, train_cls_arr)
    print("Tree with", class_tree_B.depth, "was built successfully!")
    accB = class_tree_B.dt_inference(class_tree_B.root, validation_df, validation_cls_arr)
    print("The accuracy of the tree is:", accB)
    end_time_B = datetime.now()
    ##########################################################

    ##############trail 3 with depth 15 ##########
    start_time_C = datetime.now()
    # build decision tree with depth 15
    class_tree_C = dt_tree(15)
    class_tree_C.root = class_tree_C.dt_trainer(train_df, train_cls_arr)
    print("Tree with", class_tree_C.depth, "was built successfully!")
    accC = class_tree_C.dt_inference(class_tree_C.root, validation_df, validation_cls_arr)
    print("The accuracy of the tree is:", accC)
    end_time_C = datetime.now()
    ##########################################################

    best_tree, best_accuracy = best_class_tree(class_tree_A, class_tree_B, class_tree_C, accA, accB, accC)
    best_accuracy = f"{best_accuracy:.3%}"
    print("The best regression Tree has", best_tree.depth, "depth, with accuracy:", best_accuracy)
    if best_tree == class_tree_A:
        print('run time: {}'.format(end_time_A - start_time_A))
    if best_tree == class_tree_B:
        print('run time: {}'.format(end_time_B - start_time_B))
    if best_tree == class_tree_C:
        print('run time: {}'.format(end_time_C - start_time_C))

    # store the test data
    test_df = real_estate_features.iloc[10051:]
    # store the test classifer
    test_cls_arr = classfier_arr[10051:]
    # calculate accuracy of the test data based on the decision tree
    acc = best_tree.dt_inference(best_tree.root, test_df, test_cls_arr)
    acc = f"{acc:.3%}"
    print("Test:")
    print("The accuracy of the tree is:", acc)
    # store test results
    test_results = best_tree.dt_results(best_tree.root, test_df)


if __name__ == "__main__":
    main()
