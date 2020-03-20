#!/usr/bin/env python
# coding: utf-8

# # Importing necessary libraries

# In[612]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sb

from pprint import pprint

import random


# In[613]:


get_ipython().run_line_magic('matplotlib', 'inline')
sb.set_style("darkgrid")


# # Loading Data 

# In[614]:


org_data = pd.read_csv("./iris.data")
#print(org_data)

#assigning names to each column (features and label)
org_data.columns=['SepalLength','SepalWidth','PetalLength','PetalWidth','class']
#print(org_data)

#printing first 5 rows in beautiful tabular form 
org_data.head()


# # Splitting data in to training and testing

# In[615]:


#splitting data where split_size is in % (i.e. 70%) ==> it will then divide data in to 70% train_data and 30% test_data
def split_data (org_data, split_size):
    
    org_data_size = len(org_data)
    #print (org_data_size)
    test_data_size = round((split_size/100)*org_data_size)
    train_data_size = round(org_data_size-((split_size/100)*org_data_size))
    #print (train_data_size, test_data_size)

    #creating list of indexes which will help to randomly select datapoints from the whole data to split dataset
    indexes_list = org_data.index.tolist()
    #print(indices_list)

    #randomly selecting test datapoint's indexes
    test_data_indexes = random.sample (population=indexes_list, k= test_data_size)

    #print(test_data_indexes)

    #using loc to access rows of data frames for specific indexes and storing it in testing data    
    testing_data = org_data.loc[test_data_indexes]
    
    #testing data is being dropped and rest data is getting stored in training data
    training_data = org_data.drop(test_data_indexes)

    #print (type(testing_data),type(training_data))
    
    return training_data, testing_data
    
    
    


# In[616]:


# to generate same random numbers every time
random.seed(0)

#splitting data into 70% training and 30% testing data
training_data, testing_data = split_data(org_data, split_size=30)
#print (training_data)
#print (testing_data)


# In[617]:


#print(training_data.values)

# getting training data as 2-d list
data = training_data.values

# printing first 10 rows
data[:5]


# # Checking singularity of the data

# In[618]:


#checking if the data has only one class, so it automatically belongs to that classes and reaches leaf.
def check_singularity(data):
    
    # extracting only class column of the dataset
    classes = data[:,-1]
    
    #extracting unique classes
    unique_classes = np.unique(classes)
    #print(unique_classes)
    
    #if there is only one class return true else false
    if (len(unique_classes) == 1):
        return True
    else:
        return False


# In[619]:


#check_singularity(training_data.values)

def classify_data (data): 
    # extracting only class column of the dataset
    classes = data[:,-1]
    
    #extracting unique classes and its count
    unique_classes, unique_classes_count = np.unique(classes,return_counts=True)
    
    # storing max unique count's index
    class_max_count_index = unique_classes_count.argmax()
    
    # storing class with max count
    class_classified = unique_classes[class_max_count_index]
    
    return class_classified


# # Splitting Data

# In[620]:



#function to get all the splits between the data points
def get_splits(data):
    #making dictionary for splits
    splits = {}
    # no of columns to extract features for splitting the data
    no_of_columns = len(data[0])
    
    #column-1 because we are not interested in class/label
    for column in range (0,no_of_columns-1):
        #for every column storing splits
        splits [column] = []
        
        #extracting data column wise
        specific_column_data = data[:,column]
        
        #getting unique values in the columns
        unique_specific_column_data = np.unique(specific_column_data)
        
        #starting loop from 1 because we will get previous class also and index 0 has no previous element
        for index in range (1,len(unique_specific_column_data)):
            current_column_value = unique_specific_column_data [index]
            previous_column_value = unique_specific_column_data[index-1]
            
            #split between two data points
            split = (current_column_value+previous_column_value)/2
            
            #storing split in respective column's array in dictionary
            splits[column].append(split)
            
            
    return splits


# In[621]:


#function to split the data given split column and specific value in that column
def split_data(data, split_column, split_value):
    
    #extracting split column data values
    split_column_data = data[:, split_column]

    #data below and above that split
    data_below = data[split_column_data <= split_value]
    data_above = data[split_column_data >  split_value]
    
    return data_below, data_above


# # Entropy Measurement to Determine the best split

# In[622]:


# measure entropy of different splits 
def measure_entropy(data):
    
    #extracting last column
    class_column = data[:, -1]
    #extracting unique column's counts
    _, unique_class_counts = np.unique(class_column, return_counts=True)

    #calculating probabilities of each class
    probabilities = unique_class_counts / unique_class_counts.sum()
    
    # weighted sum of product of probalility and uncertainity
    entropy = sum(probabilities * -np.log2(probabilities))
     
    #print (entropy)    
    return entropy

#measure_entropy(training_data.values)


# In[623]:


#calculating overall entropy (data below + data above)
def measure_overall_entropy(data_below, data_above):
    
    n = len(data_below) + len(data_above)
    p_data_below = len(data_below) / n
    p_data_above = len(data_above) / n

    #first calculating entropy for data below then data above and then summing to get overall entropy which will be used to 
    # determine the best split
    overall_entropy =  (p_data_below * measure_entropy(data_below) 
                      + p_data_above * measure_entropy(data_above))
    

    return overall_entropy


# In[624]:


#function to get the best split
def find_best_split(data, splits):
    
    #first setting entropy to max but considering it as minimum entropy
    min_entropy = 99999
    
    #looping the dataframe having lists
    for column in splits:
        #loop for the list inside dataframe
        for value in splits[column]:
            #extracting data below the split and data above
            data_below, data_above = split_data (data, split_column=column, split_value=value)
            #calculating overall entropy for that split
            entropy = measure_overall_entropy(data_below, data_above)
            
            #check to determine minimum entropy which will determine the best split
            if entropy <= min_entropy:
                min_entropy = entropy
                best_split_feature = column
                best_split_value = value
    
    return best_split_feature, best_split_value


# In[625]:



def build_decision_tree(training_data, counter=0, min_samples=2, max_depth=5):
    
    # converting data which is at first dataframe into 2-d array but after first time it will 
    # be array, so else case is for that.
    if counter == 0:
        global COLUMN_HEADERS
        COLUMN_HEADERS = training_data.columns
        data = training_data.values
    else:
        data = training_data           
    
    
    # base cases
    # 1. if the data has only one label it means it is the base case because there is no need to further split the data
    # 2. if counter which is incremented on every recursive call reaches the max_depth
    # 3. if datapoints become less than min_samples
    if (check_singularity(data)) or (counter == max_depth) or (len(data) < min_samples):
        label = classify_data(data)
        
        return label

    
    # recursive part
    else:    
        counter = counter + 1

        # getting all splits
        splits = get_splits(data)
        # finding best split_column and split value
        split_column, split_value = find_best_split(data, splits)
        #getting data below and data above that split
        data_below, data_above = split_data(data, split_column, split_value)
        
        # instantiate sub-tree
        feature_name = COLUMN_HEADERS[split_column]
        question = "{} <= {}".format(feature_name, split_value)
        sub_tree = {question: []}
        
        # find answers (recursion)
        is_classified = build_decision_tree(data_below, counter, min_samples, max_depth)
        is_not_classified = build_decision_tree(data_above, counter, min_samples, max_depth)
        
        # If both is_classified and is_not_classified are same there is no need to further split and break tree
        # the data is classified
        # yet (min_samples or max_depth base cases).
        if is_classified == is_not_classified:
            sub_tree = is_classified
        else:
            sub_tree[question].append(is_classified)
            sub_tree[question].append(is_not_classified)
        
        return sub_tree


# In[626]:


tree = build_decision_tree(training_data, max_depth=5)
pprint(tree)


# In[627]:


list(tree.keys())[0]


# In[628]:


def predict(datapoint, tree):
    #getting left side of the tree
    question = list(tree.keys())[0]
    #spliting data into three parts
    feature_name, comparison_operator, value = question.split()

    # if datapoint's feature name is less than equal to the value then assign label
    if datapoint[feature_name] <= float(value):
        answer = tree[question][0]
    #else assign right part of tree
    else:
        answer = tree[question][1]

    # base case
    if not isinstance(answer, dict):
        return answer
    
    #recusrsively going through the tree
    else:
        residual_tree = answer
        return predict(datapoint, residual_tree)


# In[629]:


print(testing_data.iloc[1])
predict(testing_data.iloc[1],tree)


# In[630]:


#calculating accuracy
def calculate_accuracy(testing_data, tree):

    testing_data["classification"] = testing_data.apply(predict, axis=1, args=(tree,))
    testing_data["classification_correct"] = testing_data["classification"] == testing_data["class"]
    
    accuracy = testing_data["classification_correct"].mean()
    
    return accuracy


# In[631]:


print (calculate_accuracy(testing_data,tree))

