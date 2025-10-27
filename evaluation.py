import numpy as np
from matplotlib.pylab import default_rng
from decision_tree import DecisionTree


def accuracy(y_test, predictions):
  assert len(y_test) == len(predictions)

  try:
    accuracy = np.sum(y_test == predictions) / len(y_test)
  except ZeroDivisionError:
    accuracy = 0.

  return accuracy

def confusion_matrix(y_gold, y_prediction, class_labels=None):
        # if no class_labels are given, we obtain the set of unique class labels from
    # the union of the ground truth annotation and the prediction
    if not class_labels:
        class_labels = np.unique(np.concatenate((y_gold, y_prediction)))

    confusion = np.zeros((len(class_labels), len(class_labels)), dtype=int)

    # for each correct class (row),
    # compute how many instances are predicted for each class (columns)

    for i, true_label in enumerate(class_labels):
      for j, pred_label in enumerate(class_labels):
        confusion[i, j] = np.sum((y_gold == true_label) & (y_prediction == pred_label))

    return confusion

def precision(confusion):
    class_labels = confusion.shape[0]

    col_sums = np.sum(confusion, axis=0)

    p = np.zeros(class_labels, dtype=float)

    for i in range(class_labels):
      if col_sums[i] == 0:
        p[i] = 0
      else:
        p[i] = confusion[i, i] / col_sums[i]

    # Compute the macro-averaged precision
    macro_p = np.mean(p)

    return (p, macro_p)


def recall(confusion):
    class_labels = confusion.shape[0]
    row_sums = np.sum(confusion, axis=1)
    r = np.zeros(class_labels, dtype=float)
    for i in range(class_labels):
      if row_sums[i] == 0:
        r[i] = 0
      else:
        r[i] = confusion[i, i] / row_sums[i]

    # Compute the macro-averaged recall
    macro_r = np.mean(r)
    return (r, macro_r)

def f1_score(confusion):

    (precisions, macro_p) = precision(confusion)
    (recalls, macro_r) = recall(confusion)

    # just to make sure they are of the same length
    assert len(precisions) == len(recalls)
    f = np.zeros((len(precisions), ))
    f = (2 * precisions * recalls) / (precisions + recalls)
    macro_f = np.mean(f)
    return (f, macro_f)

def accuracy_from_confusion(confusion):
    if np.sum(confusion) > 0:
        return np.trace(confusion) / np.sum(confusion)
    else:
        return 0.

def evaluate(test_db, trained_tree):
    """
    test_db: tuple (x_test, y_test)
    trained_tree: DecisionTree object already trained
    returns: accuracy (float)
    """
    x_test, y_test = test_db
    predictions = trained_tree.predict(x_test)
    acc = np.sum(predictions == y_test) / len(y_test)
    return acc

def cross_validate(dataset_x, dataset_y, k=10, seed=None):
    rng = default_rng(seed)
    n = len(dataset_y)
    indices = rng.permutation(n)
    folds = np.array_split(indices, k)

    accuracies = []
    confusion_total = None

    for i in range(k):
        test_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])

        x_train = dataset_x[train_idx]
        y_train = dataset_y[train_idx]
        x_test = dataset_x[test_idx]
        y_test = dataset_y[test_idx]

        tree = DecisionTree()
        tree.fit(x_train, y_train)

        # Use evaluate() as per spec
        acc = evaluate((x_test, y_test), tree)
        accuracies.append(acc)

        # build confusion matrix
        preds = tree.predict(x_test)
        cm = confusion_matrix(y_test, preds)
        if confusion_total is None:
            confusion_total = cm
        else:
            confusion_total += cm

    # average metrics
    avg_acc = np.mean(accuracies)
    per_class_p, macro_p = precision(confusion_total)
    per_class_r, macro_r = recall(confusion_total)
    per_class_f1, macro_f1 = f1_score(confusion_total)

    print("Cross-validation results:")
    print(f"Accuracy: {avg_acc:.4f}")
    print("Confusion matrix:\n", confusion_total)
    print("Precision per class:", per_class_p)
    print("Recall per class:", per_class_r)
    print("F1 per class:", per_class_f1)
    print(f"Macro Precision: {macro_p:.4f}, Macro Recall: {macro_r:.4f}, Macro F1: {macro_f1:.4f}")