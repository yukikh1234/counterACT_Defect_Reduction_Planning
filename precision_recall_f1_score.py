import pandas as pd

from planner import *
from othertools import *
import seaborn as sns
import matplotlib.pyplot as plt

import matplotlib.patches as patches

commit = True


def f1_score_from_list(precision_list, recall_list):
    """
    Calculates the F1 score for each pair of precision and recall values in the input lists.
    Returns a list of F1 scores.
    """
    f1_scores = []
    for precision, recall in zip(precision_list, recall_list):
        numerator = 2 * (precision * recall)
        denominator = precision + recall
        f1 = numerator / denominator
        f1_scores.append(f1)
    return f1_scores


def main():
    if commit:
        scores_t = readfile('results/rq3_Timelime_commit.csv')
        scores_t = [[ast.literal_eval(s) for s in sublist] for sublist in scores_t]
        scores_x = readfile('results/rq3_matrix_XTREE.csv')
        scores_x = [[ast.literal_eval(s) for s in sublist] for sublist in scores_x]
        scores_f = readfile('results/rq3_LIME_commit.csv')
        scores_f = [[ast.literal_eval(s) for s in sublist] for sublist in scores_f]
        scores_alve = readfile('results/rq3_Alves_commit.csv')
        scores_alve = [[ast.literal_eval(s) for s in sublist] for sublist in scores_alve]
        scores_oliv = readfile('results/rq3_Oliv_commit.csv')
        scores_oliv = [[ast.literal_eval(s) for s in sublist] for sublist in scores_oliv]
        scores_Shat = readfile('results/rq3_Shat_commit.csv')
        scores_Shat = [[ast.literal_eval(s) for s in sublist] for sublist in scores_Shat]
        scores_rw = readfile('results/rq3_matrix_Random.csv')
        scores_rw = [[ast.literal_eval(s) for s in sublist] for sublist in scores_rw]
        list1 = [scores_t, scores_f, scores_x, scores_alve, scores_Shat, scores_oliv,scores_rw]
        
        names = np.array(['TimeLIME', 'LIME', 'XTREE', 'Alves', 'Shatnawi', 'Oliveira','Random'])
        projects = ['kafka', 'activecluster', 'nifi', 'zookeper', 'phoenix']

    else:

        scores_t = readfile('results/rq3_matrix_Timeline.csv')
        scores_t = [[ast.literal_eval(s) for s in sublist] for sublist in scores_t]
        scores_CF = readfile('results/rq3_matrix_CF.csv')
        scores_CF = [[ast.literal_eval(s) for s in sublist] for sublist in scores_CF]
        scores_x = readfile('results/rq3_matrix_XTREE.csv')
        scores_x = [[ast.literal_eval(s) for s in sublist] for sublist in scores_x]
        scores_f = readfile('results/rq3_matrix_LIME.csv')
        scores_f = [[ast.literal_eval(s) for s in sublist] for sublist in scores_f]
        scores_alve = readfile('results/rq3_matrix_Alves.csv')
        scores_alve = [[ast.literal_eval(s) for s in sublist] for sublist in scores_alve]
        scores_oliv = readfile('results/rq3_matrix_Oliv.csv')
        scores_oliv = [[ast.literal_eval(s) for s in sublist] for sublist in scores_oliv]
        scores_Shat = readfile('results/rq3_matrix_Shat.csv')
        scores_Shat = [[ast.literal_eval(s) for s in sublist] for sublist in scores_Shat]
        scores_rw = readfile('results/rq3_matrix_Random.csv')
        scores_rw = [[ast.literal_eval(s) for s in sublist] for sublist in scores_rw]
        list1 = [scores_t, scores_f, scores_x, scores_alve, scores_Shat, scores_oliv, scores_rw, scores_CF]
        names = np.array(['TimeLIME', 'LIME', 'XTREE', 'Alves', 'Shatnawi', 'Oliveira', 'Random', 'CounterACT'])
        projects = ['jedit', 'camel1', 'camel2', 'log4j', 'xalan', 'ant',
                    'velocity', 'poi', 'synapse']


    results_recalls = [projects]
    results_precisions = [projects]
    results_f1_scores = [projects]
    for i in range(len(names)):
        scores = list1[i]
        precisions = []
        recalls = []
        f1_scores = []
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
            f1_score = np.round((2 * TP) / ((2 * TP) + FP + FN), 2) * 100
            f1_scores.append(np.around(f1_score))

        results_recalls.append(recalls)
        results_precisions.append(precisions)
        results_f1_scores.append(f1_scores)


    return results_precisions, results_recalls, results_f1_scores


if __name__ == "__main__":
    if commit:
        names = np.array(['TimeLIME', 'LIME', 'XTREE', 'Alves', 'Shatnawi', 'Oliveira','Random'])
        precision, recall, f1_score = main()

        print(precision)

        results_f1_score = pd.DataFrame(f1_score)
        results_f1_score.columns = results_f1_score.iloc[0]
        results_f1_score = results_f1_score[1:].set_index(names).T



        # results_f1_score = results_f1_score.drop(results_f1_score.index[-6:], axis=0)
        print(results_f1_score)

        results_recalls = pd.DataFrame(recall)
        results_recalls.columns = results_recalls.iloc[0]
        results_recalls = results_recalls[1:].set_index(names).T

        # results_recalls = results_recalls.drop(results_recalls.index[-6:], axis=0)

        results_recalls = pd.DataFrame(recall)
        results_recalls.columns = results_recalls.iloc[0]
        results_recalls = results_recalls[1:].set_index(names).T

        results_precision = pd.DataFrame(precision)
        results_precision.columns = results_precision.iloc[0]
        results_precision = results_precision[1:].set_index(names).T
        # results_precision = results_precision.drop(results_precision.index[-6:], axis=0)
        CounterACT_recall = [74, 78, 70, 78, 67]
        results_recalls['CounterACT'] = [74, 78, 70, 78, 67]

        
        ARM_recall = [45, 37, 38, 37, 34]
        results_recalls['ARM'] = [45, 37, 38, 37, 34]

        CounterACT_precision = [74, 78, 70, 78, 67]
        results_precision['CounterACT'] = [70, 50, 82, 83, 55]

        ARM_precision = [11, 46, 46, 34, 40] 
        results_precision['ARM'] = [11, 46, 46, 34, 40] 

        f1_scores = f1_score_from_list(CounterACT_recall, CounterACT_precision)
        results_f1_score['CounterACT'] = f1_scores

        f1_scores = f1_score_from_list(ARM_recall, ARM_precision)
        results_f1_score['ARM'] = f1_scores

    else:

        names = np.array(['TimeLIME', 'LIME', 'XTREE', 'Alves', 'Shatnawi', 'Oliveira', 'Random','CounterACT'])

        precision, recall, f1_score = main()
        results_f1_score = pd.DataFrame(f1_score)
        results_f1_score.columns = results_f1_score.iloc[0]

        results_f1_score = results_f1_score[1:].set_index(names).T

        results_f1_score = results_f1_score.drop(results_f1_score.index[-1:], axis=0)

        results_recalls = pd.DataFrame(recall)
        results_recalls.columns = results_recalls.iloc[0]
        results_recalls = results_recalls[1:].set_index(names).T

        results_recalls = results_recalls.drop(results_recalls.index[-1:], axis=0)

        results_precision = pd.DataFrame(precision)
        results_precision.columns = results_precision.iloc[0]
        results_precision = results_precision[1:].set_index(names).T
        results_precision = results_precision.drop(results_precision.index[-1:], axis=0)

        ARM_recall = [37, 1, 10, 11, 46, 16, 33, 31.5]
        results_recalls['ARM'] = ARM_recall

        ARM_precision = [39, 16, 13, 1,37, 12, 34.27, 31.55] 
        results_precision['ARM'] = ARM_precision
        
        f1_scores = f1_score_from_list(ARM_recall, ARM_precision)
        results_f1_score['ARM'] = f1_scores


    names = np.array(['TimeLIME', 'LIME', 'XTREE', 'Alves', 'Shatnawi', 'Oliveira', 'Random',"ARM", 'CounterACT'])

    results_precision['Type'] = 'Precision'
    results_recalls['Type'] = 'Recall'
    results_f1_score['Type'] = 'F1_score'

    # combine dataframes into a single dataframe
    df_combined = pd.concat([results_precision, results_recalls, results_f1_score])
    fig, axs = plt.subplots(nrows=1, ncols=9, figsize=(24, 3), sharey=True)

    # Add a red box to the 6th subplot
    for spine in axs[8].spines.values():
        spine.set_edgecolor('navy')
        spine.set_linewidth(2)
    for i in range(9):
        df_combined.fillna(0.0,inplace=True)
        # create boxplot using seaborn
        sns.boxplot(y=names[i], data=df_combined, x='Type', ax=axs[i], palette='Set2')
        axs[i].set_title(names[i])
    for ax in axs.flat:
        ax.set_xlabel('')
        ax.set_ylabel('')
    # enable tick labels on both sides of each subplot

    fig.subplots_adjust(top=0.8, hspace=1.1)


    # show the plot
    plt.savefig('precision_recall_score_commit.pdf', format='pdf')

    plt.show()

