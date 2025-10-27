import numpy as np
from numpy.random import default_rng
from read_data import read_dataset, split_dataset

#read from files
(clean_x, clean_y, clean_classes) = read_dataset("clean_dataset.txt")
(noisy_x, noisy_y, noisy_classes) = read_dataset("noisy_dataset.txt")

seed = 60012
rg = default_rng(seed)
#split datasets
x_train_clean, x_test_clean, y_train_clean, y_test_clean = split_dataset(
                                                 clean_x, clean_y,
                                                 test_proportion=0.2,
                                                 random_generator=rg)

x_train_noisy, x_test_noisy, y_train_noisy, y_test_noisy = split_dataset(
                                                 noisy_x, noisy_y,
                                                 test_proportion=0.2,
                                                 random_generator=rg)

#node class for decision tree
class Node:
  def __init__(self, attribute=None, value=None, l_tree=None, r_tree=None, leaf_class=None):
    self.attribute = attribute #which column to split on
    self.value = value #value to split by
    self.l_tree = l_tree #subtree for < split value
    self.r_tree = r_tree #subtree for > split value
    self.leaf_class = leaf_class #class label if leaf node

#calculate entropy
def h(label_counts):
  if np.sum(label_counts) == 0:
    return 0
  pk = label_counts / np.sum(label_counts)
  pk = pk[pk > 0]
  return - np.sum(pk * np.log2(pk))

#calculate information gain for splitting into given subsets
def calc_info_gain(l_sizes, r_sizes):
  h_all = h(r_sizes + l_sizes)
  h_left = h(l_sizes)
  h_right = h(r_sizes)
  left_size = np.sum(l_sizes)
  right_size = np.sum(r_sizes)
  remainder = ((left_size * h_left)/(left_size + right_size)) + ((right_size * h_right)/(left_size + right_size))
  return h_all - remainder

#find the best split for a given attribute (column)
def find_col_split(dataset_x, dataset_y, attribute):
  sorted_indices = np.argsort(dataset_x[:,attribute])
  best_split_val = None
  best_info_gain = 0
  unique_labels, r_sizes = np.unique(dataset_y, return_counts = True)
  l_sizes = np.zeros_like(r_sizes)
  # tracking number of each label in the subsplits
  label_to_i = {label: idx for idx, label in enumerate(unique_labels)}
  # labels may not be 0 indexed and consecutive
  for i in range(dataset_x.shape[0] - 1):
      if dataset_x[sorted_indices[i], attribute] == dataset_x[sorted_indices[i+1], attribute]:
        continue
      label = label_to_i[dataset_y[sorted_indices[i]]]
      l_sizes[label] += 1
      r_sizes[label] -= 1
      info_gain = calc_info_gain(l_sizes, r_sizes)
      if info_gain > best_info_gain:
          best_info_gain = info_gain
          best_split_val = (dataset_x[sorted_indices[i], attribute] + dataset_x[sorted_indices[i+1], attribute]) / 2
  return best_split_val, best_info_gain

#find the best split across all attributes
def find_split(dataset_x, dataset_y):
  splits = np.zeros((dataset_x.shape[1], 2))
  for attribute in range(dataset_x.shape[1]):
    splits[attribute] = find_col_split(dataset_x, dataset_y, attribute)
  max_index = np.argmax(splits[:, 1])
  return splits[max_index, 0], max_index

#recursively build decision tree
def decision_tree_learning(dataset_x, dataset_y, depth):
  classes = np.unique(dataset_y)
  if classes.shape[0] <= 1:
    return Node(leaf_class=classes[0]), depth
  else:
    split_val, split_attr = find_split(dataset_x, dataset_y)
    split_attr = int(split_attr)
    # no info gain
    if split_val == None:
      most_common = np.bincount(dataset_y).argmax()
      return Node(leaf_class=most_common), depth
    pred = dataset_x[:, split_attr] < split_val
    l_branch, l_depth = decision_tree_learning(dataset_x[pred], dataset_y[pred], depth + 1)
    r_branch, r_depth = decision_tree_learning(dataset_x[~pred], dataset_y[~pred], depth + 1)
    node = Node(attribute=split_attr, value=split_val, l_tree=l_branch, r_tree=r_branch)
    return node, max(l_depth, r_depth)

#decision tree class
class DecisionTree:
  #fit the decision tree to the dataset
  def fit(self, dataset_x, dataset_y):
    self.tree, _ = decision_tree_learning(dataset_x, dataset_y, 0)
  
  #predict class labels for given inputs
  def predict(self, xs):
    y = np.zeros(xs.shape[0], dtype=int)
    for i, x in enumerate(xs):
      cur = self.tree
      while cur.leaf_class is None:
        if x[cur.attribute] <= cur.value:
          cur = cur.l_tree
        else:
          cur = cur.r_tree
      y[i] = cur.leaf_class
    return y