import pandas as pd

from planner import *
from othertools import *
import matplotlib.pyplot as plt


def main():
    scores_t = readfile('rq2_TimeLIME.csv')
    scores_f = readfile('rq2_LIME.csv')
    scores_x = readfile('rq2_XTREE.csv')
    scores_alve = readfile('rq2_Alves.csv')
    scores_shat = readfile('rq2_Shat.csv')
    scores_oliv = readfile('rq2_Oliv.csv')
    scores_rw = readfile('rq2_Random.csv')
    scores_CF= readfile('rq2_CF.csv')

    bcs_t = readfile('rq3_TimeLIME.csv')
    bcs_f = readfile('rq3_LIME.csv')
    bcs_x = readfile('rq3_XTREE.csv')
    bcs_alve = readfile('rq3_Alves.csv')
    bcs_shat = readfile('rq3_Shat.csv')
    bcs_oliv = readfile('rq3_Oliv.csv')
    bcs_rw = readfile('rq3_Random.csv')
    bcs_CF = readfile('rq3_CF.csv')

    list1 = [scores_t,scores_f,scores_x,scores_alve,scores_shat,scores_oliv,scores_rw,scores_CF]
    list2 = [bcs_t,bcs_f,bcs_x,bcs_alve,bcs_shat,bcs_oliv,bcs_rw,bcs_CF]
    names = ['TimeLIME','LIME','XTREE','Alves','Shatnawi','Oliveira','Random','CF']
    projects = ['jedit','camel1','camel2', 'log4j', 'xalan','ant','velocity','poi','synapse']

    results=[projects]
    print(projects)
    print()
    for i in range(len(names)):
        scores = list1[i]
        bcs = list2[i]
        dummy = []
        N = len(scores)
        for k in range(0, len(scores)):
            temp = 0
            for j in range(0, len(scores[k])):
                temp -= (bcs[k][j] * scores[k][j])
            total = -np.sum(bcs[k])
            dummy.append(np.round(temp / total, 2)*100)
        results.append(dummy)
    return results

def main():
    scores_t = readfile('results/rq2_TimeLIME.csv')
    scores_f = readfile('results/rq2_LIME.csv')
    scores_x = readfile('results/rq2_XTREE.csv')
    scores_alve = readfile('results/rq2_Alves.csv')
    scores_shat = readfile('results/rq2_Shat.csv')
    scores_oliv = readfile('results/rq2_Oliv.csv')
    scores_rw = readfile('results/rq2_Random.csv')
    scores_CF= readfile('results/rq2_CF.csv')

    bcs_t = readfile('results/rq3_TimeLIME.csv')
    bcs_f = readfile('results/rq3_LIME.csv')
    bcs_x = readfile('results/rq3_XTREE.csv')
    bcs_alve = readfile('results/rq3_Alves.csv')
    bcs_shat = readfile('results/rq3_Shat.csv')
    bcs_oliv = readfile('results/rq3_Oliv.csv')
    bcs_rw = readfile('results/rq3_Random.csv')
    bcs_CF = readfile('results/rq3_CF.csv')

    list1 = [scores_t,scores_f,scores_x,scores_alve,scores_shat,scores_oliv,scores_rw,scores_CF]
    list2 = [bcs_t,bcs_f,bcs_x,bcs_alve,bcs_shat,bcs_oliv,bcs_rw,bcs_CF]
    names = ['TimeLIME','LIME','XTREE','Alves','Shatnawi','Oliveira','Random','CF']
    # projects = ['jedit','camel1','camel2', 'log4j', 'xalan','ant','velocity','poi','synapse']
    projects = ['xalan-ant','log4j-ant','camel-log4j']


    results=[projects]
    print(projects)
    print()
    for i in range(len(names)):
        scores = list1[i]
        bcs = list2[i]
        dummy = []
        N = len(scores)
        for k in range(0, len(scores)):
            temp = 0
            for j in range(0, len(scores[k])):
                temp -= (bcs[k][j] * scores[k][j])
            total = -np.sum(bcs[k])
            dummy.append(np.round(temp / total, 2)*100)
        results.append(dummy)
    return results

if __name__ == "__main__":
    names = np.array(['TimeLIME','LIME','XTREE','Alves','Shatnawi','Oliveira','Random','CF'])

    results = main()
    result = pd.DataFrame(results)
    result.columns = result.iloc[0]
    result = result[1:].set_index(names)
    print(result.T)
    result.T.to_excel('results/RQs/RQ3_1_cross_project.xlsx')
