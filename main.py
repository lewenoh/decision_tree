from numpy.random import default_rng
from plot_tree import plot_tree
from evaluation import cross_validate
from decision_tree import DecisionTree, read_dataset
from read_data import split_dataset


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


#tree trained on entire clean dataset
clean_tree = DecisionTree()
clean_tree.fit(clean_x, clean_y)
plot_tree(clean_tree.tree)


(clean_x, clean_y, _) = read_dataset("clean_dataset.txt")
(noisy_x, noisy_y, _) = read_dataset("noisy_dataset.txt")

#c
print("\nClean dataset:")
cross_validate(clean_x, clean_y, k=10, seed=42)

print("\nNoisy dataset:")
cross_validate(noisy_x, noisy_y, k=10, seed=42)
