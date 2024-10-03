import os
import numpy as np

def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)


def calculate_scalar(x):
    if x.ndim == 2:
        axis = 0
    elif x.ndim == 3:
        axis = (0, 1)

    mean = np.mean(x, axis=axis)
    std = np.std(x, axis=axis)

    return mean, std


def scale(x, mean, std):
    return (x - mean) / std


# ------------------ demo ---------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------

from sklearn import metrics
def cal_acc_auc(predictions, targets):
    tagging_truth_label_matrix = targets
    pre_tagging_label_matrix = predictions

    # overall
    tp = np.sum(pre_tagging_label_matrix + tagging_truth_label_matrix > 1.5)
    fn = np.sum(tagging_truth_label_matrix - pre_tagging_label_matrix > 0.5)
    fp = np.sum(pre_tagging_label_matrix - tagging_truth_label_matrix > 0.5)
    tn = np.sum(pre_tagging_label_matrix + tagging_truth_label_matrix < 0.5)

    Acc = (tp + tn) / (tp + tn + fp + fn)

    aucs = []
    for i in range(targets.shape[0]):
        test_y_auc, pred_auc = targets[i, :], predictions[i, :]
        if np.sum(test_y_auc):
            test_auc = metrics.roc_auc_score(test_y_auc, pred_auc)
            aucs.append(test_auc)
    final_auc = sum(aucs) / len(aucs)
    return Acc, final_auc




def calculate_accuracy(target, predict, classes_num, average=None):
    """Calculate accuracy.

    Inputs:
      target: integer array, (audios_num,)
      predict: integer array, (audios_num,)

    Outputs:
      accuracy: float
    """

    samples_num = len(target)

    correctness = np.zeros(classes_num)
    total = np.zeros(classes_num)

    for n in range(samples_num):

        total[target[n]] += 1

        if target[n] == predict[n]:
            correctness[target[n]] += 1

    accuracy = correctness / total

    if average is None:
        return accuracy

    elif average == 'macro':
        return np.mean(accuracy)

    else:
        raise Exception('Incorrect average!')



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


