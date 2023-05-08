import pandas as pd
from othertools import *
import scipy as sp



# def main():
#     scores_t = readfile('results/rq2_TimeLIME.csv')
#     scores_f = readfile('results/rq2_LIME.csv')
#     scores_x = readfile('results/rq2_XTREE.csv')
#     scores_alve = readfile('results/rq2_Alves.csv')
#     scores_shat = readfile('results/rq2_Shat.csv')
#     scores_oliv = readfile('results/rq2_Oliv.csv')
#     scores_rw = readfile('results/rq2_Random.csv')
#     scores_CF= readfile('results/rq2_CF.csv')
#
#     bcs_t = readfile('results/rq3_TimeLIME.csv')
#     bcs_f = readfile('results/rq3_LIME.csv')
#     bcs_x = readfile('results/rq3_XTREE.csv')
#     bcs_alve = readfile('results/rq3_Alves.csv')
#     bcs_shat = readfile('results/rq3_Shat.csv')
#     bcs_oliv = readfile('results/rq3_Oliv.csv')
#     bcs_rw = readfile('results/rq3_Random.csv')
#     bcs_CF = readfile('results/rq3_CF.csv')
#
#     list1 = [scores_t,scores_f,scores_x,scores_alve,scores_shat,scores_oliv,scores_rw,scores_CF]
#     list2 = [bcs_t,bcs_f,bcs_x,bcs_alve,bcs_shat,bcs_oliv,bcs_rw,bcs_CF]
#     names = ['TimeLIME','LIME','XTREE','Alves','Shatnawi','Oliveira','Random','CF']
#     projects = ['jedit', 'camel1', 'camel2', 'log4j', 'xalan', 'ant',
#                 'velocity', 'poi', 'synapse','xlan\nant','log4j\nant','camel\nlog4j',
#                 'velocity\nsynapse', 'jedit\npoi', 'synapse\nxalan']
#     # projects = ['xalan-ant','log4j-ant','camel-log4j']
#
#
#     results=[projects]
#     print(projects)
#     print()
#     for i in range(len(names)):
#         scores = list1[i]
#         bcs = list2[i]
#         dummy = []
#         N = len(scores)
#         for k in range(0, len(scores)):
#             temp = 0
#             for j in range(0, len(scores[k])):
#                 temp -= (bcs[k][j] * scores[k][j])
#             total = -np.sum(bcs[k])
#             dummy.append(np.round(temp / total, 2)*100)
#         results.append(dummy)
#     return results

def main():

    scores_f = readfile('results/rq2_LIME_commit.csv')
    scores_x = readfile('results/rq2_XTREE.csv')
    scores_alve = readfile('results/rq2_Alves_commit.csv')
    scores_shat = readfile('results/rq2_Shat_commit.csv')
    scores_oliv = readfile('results/rq2_Oliv_commit.csv')
    scores_t = readfile('results/rq2_TimeLIME_commit.csv')

    results_IQR = []
    results_overlap = []

    N = len(scores_t)
    for i in range(N):
        IQR = []
        overlap = []
        print()
        # print(fnames[i][0])

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



        results_IQR.append(IQR)
        results_overlap.append(overlap)
    return results_IQR, results_overlap

if __name__ == "__main__":
    names = np.array(['TimeLIME', 'LIME', 'XTREE','Alves','Shatnawi', 'Oliveira'])
    projects = ['kafka', 'activecluster', 'nifi', 'zookeper', 'phoenix']

    results_overlap, results_IQR = main()


    results_overlap = pd.DataFrame(results_overlap)
    results_overlap.columns = names
    results_overlap = results_overlap.T
    results_overlap.columns = projects
    # results_overlap.T.to_excel('results/RQs/RQ2_overlap_cross_project.xlsx')

    results_IQR = pd.DataFrame(results_IQR)
    results_IQR.columns = names
    results_IQR = results_IQR.T
    results_IQR.columns = projects

    print(results_overlap.T)

    print(results_IQR.T)
