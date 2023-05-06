from planner import *
from othertools import *
import matplotlib.pyplot as plt
import scipy as sp


def main():
    fnames = [['jedit-4.0.csv', 'jedit-4.1.csv', 'jedit-4.2.csv'],
              ['camel-1.0.csv', 'camel-1.2.csv', 'camel-1.4.csv'],
              ['camel-1.2.csv', 'camel-1.4.csv', 'camel-1.6.csv'],
              ['log4j-1.0.csv', 'log4j-1.1.csv', 'log4j-1.2.csv'],
              ['xalan-2.4.csv', 'xalan-2.5.csv', 'xalan-2.6.csv'],
              ['ant-1.5.csv', 'ant-1.6.csv', 'ant-1.7.csv'],
              ['velocity-1.4.csv', 'velocity-1.5.csv', 'velocity-1.6.csv'],
              ['poi-1.5.csv', 'poi-2.5.csv', 'poi-3.0.csv'],
              ['synapse-1.0.csv', 'synapse-1.1.csv', 'synapse-1.2.csv']
              ]
    scores_t = readfile('results/rq2_TimeLIME.csv')
    scores_f = readfile('results/rq2_LIME.csv')
    scores_x = readfile('results/rq2_XTREE.csv')
    scores_alve = readfile('results/rq2_Alves.csv')
    scores_shat = readfile('results/rq2_Shat.csv')
    scores_oliv = readfile('results/rq2_Oliv.csv')
    scores_rw = readfile('results/rq2_Random.csv')
    score_CF = readfile('results/rq2_CF.csv')

    results_IQR = []
    results_overlap = []

    N = len(scores_t)
    for i in range(N):
        IQR = []
        overlap = []
        print()
        print(fnames[i][0])

        IQR.append(sp.stats.iqr(scores_t[i]) * 100)
        overlap.append(np.median(scores_t[i]) * 100)
        IQR.append(sp.stats.iqr(scores_f[i]) * 100)
        overlap.append(np.median(scores_f[i]) * 100)
        IQR.append(sp.stats.iqr(scores_x[i]) * 100)
        overlap.append(np.median(scores_x[i]) * 100)
        IQR.append(sp.stats.iqr(scores_alve[i]) * 100)
        overlap.append(np.median(scores_alve[i]) * 100)
        IQR.append(sp.stats.iqr(scores_shat[i]) * 100)
        overlap.append(np.median(scores_shat[i]) * 100)
        IQR.append(sp.stats.iqr(scores_oliv[i]) * 100)
        overlap.append(np.median(scores_oliv[i]) * 100)
        IQR.append(sp.stats.iqr(scores_rw[i]) * 100)
        overlap.append(np.median(scores_rw[i]) * 100)
        IQR.append(sp.stats.iqr(score_CF[i]) * 100)
        overlap.append(np.median(score_CF[i]) * 100)

        results_IQR.append(IQR)
        results_overlap.append(overlap)

    return results_overlap, results_IQR


if __name__ == "__main__":
    results_overlap, results_IQR = main()

    names = np.array(['TimeLIME', 'LIME', 'XTREE', 'Alves', 'Shatnawi', 'Oliveira', 'Random', 'CF'])
    # projects = ['jedit', 'camel1', 'camel2', 'log4j', 'xalan', 'ant', 'velocity', 'poi', 'synapse']
    projects = ['xalan-ant','log4j-ant','camel-log4j']


    results_overlap = pd.DataFrame(results_overlap)
    results_overlap.columns = names
    results_overlap = results_overlap.T
    results_overlap.columns = projects
    results_overlap.T.to_excel('results/RQs/RQ2_overlap_cross_project.xlsx')

    results_IQR = pd.DataFrame(results_IQR)
    results_IQR.columns = names
    results_IQR = results_IQR.T
    results_IQR.columns = projects
    results_IQR.T.to_excel('results/RQs/RQ2_IQR_cross_project.xlsx')


    print(results_overlap.T)
