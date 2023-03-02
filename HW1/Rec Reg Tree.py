from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


def ssr_score(attr_array: np.ndarray, splitter: float, classfier_arr: np.ndarray):
    # calculate the mean of all values above the splitter value
    mean_above_splitter = classfier_arr[attr_array >= splitter].mean()
    # scalculate the mean of below the splitter value
    mean_below_splitter = classfier_arr[attr_array < splitter].mean()

    # sum the square residuals(real attribute value-mean of above split values) of all elements in the array
    # that are bigger than the splitter
    sum_above_splitter = ((classfier_arr[attr_array >= splitter] - mean_above_splitter) ** 2).sum()
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
    ):
        self.data_df = data_df
        self.classffier_arr = classffier_arr
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

        # if the current value is bigger than the node's splitter
        # go to the left child to classify
        # return the result of the left child
        if current_value >= self.splitter and self.children[0] is not None:
            return self.children[0].classify(data_row)
        # if the current value is smaller than the node's splitter
        # go to the right child to classify
        # return the result of the right child
        elif current_value < self.splitter and self.children[1] is not None:
            return self.children[1].classify(data_row)
        else:
            # return the class of this node
            return self.mean_price


class Regression_tree:
    def __init__(self,
                 min_split_size):
        self.min_split_size = min_split_size
        self.root = None

    def dt_trainer(self, data_df: pd.DataFrame, classffier_arr: np.ndarray):
        def split(data_df: pd.DataFrame, classffier_arr: np.ndarray):
            if len(classffier_arr) < self.min_split_size:
                return None

            # create  node
            node = Node(data_df=data_df, classffier_arr=classffier_arr)

            # calculate which attribute to split by-the one with the best ssr will be chosen
            min_ssr, splitter, best_col = choose_column_to_split_by(data_df=node.data_df,
                                                                    classffier_arr=node.classffier_arr)

            # store the values of the attribute to split data by
            node.column = best_col
            node.ssr = min_ssr
            node.splitter = splitter

            # Store the data that has bigger value from the splitter
            # and drop the current attribute from the data

            left_node_df = node.data_df[node.data_df[best_col] >= splitter]
            right_node_df = node.data_df[node.data_df[best_col] < splitter]
            left_clasffier=classffier_arr[node.data_df[best_col] >= splitter]
            right_clasffier=classffier_arr[node.data_df[best_col] < splitter]
            node.children = [split(left_node_df, left_clasffier), split(right_node_df, right_clasffier)]
            return node

        self.root = split(data_df, classffier_arr)


    def dt_inference(self, dt: Node, data_df: pd.DataFrame, calssifier_arr):
        correct = 0
        # as long as we have rows in the data frame
        for i in range(len(data_df)):
            row = data_df.iloc[i]
            # predict the price of the current row according to the tree's decision
            predicted = dt.classify(row)
            # store the real price of the current row
            real = calssifier_arr[i]

            # if the predition was corret- count it
            if real >= predicted * 1.05 or real <= predicted * 0.95:
                correct += 1

        # calculate how many true predictions where made from all the data
        accuracy = correct / len(data_df)
        return accuracy

    def dt_results(self, dt: Node, data_df: pd.DataFrame):
        results_arr = []

        # as long as we have rows in the data frame
        for i in range(len(data_df)):
            row = data_df.iloc[i]
            # predict the price of the current row according to the tree's decision
            predicted = dt.classify(row)
            results_arr.append(predicted)

        return results_arr


def main():
    real_estate = pd.read_csv(r'C:\Users\taldu\OneDrive\Desktop\לימודים\מערכות מידע\למידת מכונה\HW1\data.csv',
                              index_col=[0])
    # store area_type column as classifier
    classfier_arr = np.array(real_estate['price in rupees'])

    # store data without the classifier
    real_estate_features = real_estate.drop('price in rupees', axis=1)
    real_estate_features.loc[(real_estate_features.area_type == 'B'), "area_type"] = 0
    real_estate_features.loc[(real_estate_features.area_type == 'P'), "area_type"] = 1
    real_estate_features['area_type']=pd.to_numeric(real_estate_features['area_type'])
    # store train data
    train_df = real_estate_features.iloc[:8040]
    # store the classifier for train data
    train_price_arr = classfier_arr[:8040]
    # build decision tree based on training data
    reg_tree = Regression_tree(20)
    reg_tree.dt_trainer(train_df, train_price_arr)
    print("Tree was built successfully!")


if __name__ == "__main__":
    main()
