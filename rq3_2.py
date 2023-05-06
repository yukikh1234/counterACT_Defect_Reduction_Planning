import pandas as pd

from planner import *
from othertools import *

import matplotlib.pyplot as plt


def main():
    scores_t = readfile('results/rq3_matrix_Timeline.csv')
    scores_t = [[ast.literal_eval(s) for s in sublist] for sublist in scores_t]
    scores_CF = readfile('results/rq3_matrix_CF.csv')
    scores_CF = [[ast.literal_eval(s) for s in sublist] for sublist in scores_CF]

    list1 = [scores_t, scores_CF]
    names = ['TimeLIME', 'CF']
    # projects = ['jedit', 'camel1', 'camel2', 'log4j', 'xalan', 'ant', 'velocity', 'poi', 'synapse']
    projects = ['xalan-ant','log4j-ant','camel-log4j']


    results_recalls = [projects]
    results_precisions = [projects]
    for i in range(len(names)):
        scores = list1[i]
        precisions = []
        recalls = []
        #     dummy = []
        N = len(projects)
        for k in range(0, N):
            # tp, tn, fp, fn
            TP = np.mean([lst[0] for lst in scores[k]])
            TN = np.mean([lst[1] for lst in scores[k]])
            FP = np.mean([lst[2] for lst in scores[k]])
            FN = np.mean([lst[3] for lst in scores[k]])

            recalls.append(np.round(TP / (TP + FN), 2) * 100)
            precisions.append(np.round(TP / (TP + FP), 2) * 100)
        results_recalls.append(recalls)
        results_precisions.append(precisions)

    return results_precisions, results_recalls


if __name__ == "__main__":
    names = np.array(['TimeLIME', 'CF'])

    precision, recall = main()

    results_recalls = pd.DataFrame(recall)
    results_recalls.columns = results_recalls.iloc[0]
    results_recalls = results_recalls[1:].set_index(names).T
    results_recalls.loc['AVG'] = results_recalls.mean()
    results_recalls.loc['STD'] = results_recalls.std()
    results_recalls.loc['Rank'] = [2, 1]
    print(results_recalls)

    results_precision = pd.DataFrame(precision)
    results_precision.columns = results_precision.iloc[0]
    results_precision = results_precision[1:].set_index(names).T

    results_precision.loc['AVG'] = results_precision.mean()
    results_precision.loc['STD'] = results_precision.std()
    results_precision.loc['Rank'] = [1, 1]
    print(results_precision)

    results_recalls.to_excel('results/RQs/RQ3_recall_cross_projects.xlsx')
    results_precision.to_excel('results/RQs/RQ3_precision_cross_projects.xlsx')
