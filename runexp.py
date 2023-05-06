import numpy as np

from planner import *
import warnings

# Ignore all warning messages
warnings.simplefilter("ignore")
import matplotlib.pyplot as plt


def main():
    fnames = [
        ['jedit-4.0.csv', 'jedit-4.1.csv', 'jedit-4.2.csv'],
        ['camel-1.0.csv', 'camel-1.2.csv', 'camel-1.4.csv'],
        ['camel-1.2.csv', 'camel-1.4.csv', 'camel-1.6.csv'],
        ['log4j-1.0.csv', 'log4j-1.1.csv', 'log4j-1.2.csv'],
        ['xalan-2.4.csv', 'xalan-2.5.csv', 'xalan-2.6.csv'],
        ['ant-1.5.csv', 'ant-1.6.csv', 'ant-1.7.csv'],
        ['velocity-1.4.csv', 'velocity-1.5.csv', 'velocity-1.6.csv'],
        ['poi-1.5.csv', 'poi-2.5.csv', 'poi-3.0.csv'],
        ['synapse-1.0.csv', 'synapse-1.1.csv', 'synapse-1.2.csv'],

        # ['xalan-all.csv', 'ant-1.6.csv', 'ant-1.7.csv'],
        # ['log4j-all.csv', 'ant-1.6.csv', 'ant-1.7.csv'],
        # ['camel-all.csv', 'log4j-1.1.csv', 'log4j-1.2.csv']
    ]

    # CF planner
    scores_cf, bcs_cf = [], []
    size_cf, score_2cf = [], []
    records2_cf = []
    con_matrix1_cf = []
    for name in fnames:
        score, bc, size, score_2, rec, mat = CF(name)
        scores_cf.append(score)
        bcs_cf.append(bc)
        size_cf.append(size)
        score_2cf.append(score_2)
        records2_cf.append(rec)
        con_matrix1_cf.append(mat)

    paras = [True]
    explainer = None
    print("RUNNING FREQUENT ITEMSET LEARNING")
    # scores_c,bcs_c=[],[]
    # size_c,score_2c=[],[]
    old, new = [], []
    for par in paras:
        for name in fnames:
            o, n = historical_logs(name, 20, explainer, smote=True, small=.03, act=par)
            old.append(o)
            new.append(n)
    everything = []
    for i in range(len(new)):
        everything.append(old[i] + new[i])

    # TimeLIME planner
    paras = [True]
    explainer = None
    fnames = [
        ['jedit-4.0.csv', 'jedit-4.1.csv', 'jedit-4.2.csv'],
        ['camel-1.0.csv', 'camel-1.2.csv', 'camel-1.4.csv'],
        ['camel-1.2.csv', 'camel-1.4.csv', 'camel-1.6.csv'],
        ['log4j-1.0.csv', 'log4j-1.1.csv', 'log4j-1.2.csv'],
        ['xalan-2.4.csv', 'xalan-2.5.csv', 'xalan-2.6.csv'],
        ['ant-1.5.csv', 'ant-1.6.csv', 'ant-1.7.csv'],
        ['velocity-1.4.csv', 'velocity-1.5.csv', 'velocity-1.6.csv'],
        ['poi-1.5.csv', 'poi-2.5.csv', 'poi-3.0.csv'],
        ['synapse-1.0.csv', 'synapse-1.1.csv', 'synapse-1.2.csv']
        # ['xalan-all.csv', 'ant-1.6.csv', 'ant-1.7.csv'],
        # ['log4j-all.csv', 'ant-1.6.csv', 'ant-1.7.csv'],
        # ['camel-all.csv', 'log4j-1.1.csv', 'log4j-1.2.csv']

    ]

    scores_t, bcs_t = [], []
    size_t, score_2t = [], []
    records2 = []
    con_matrix1 = []
    i = 0
    for par in paras:
        for name in fnames:
            df = pd.DataFrame(everything[i])
            i += 1
            itemsets = convert_to_itemset(df)
            te = TransactionEncoder()
            te_ary = te.fit(itemsets).transform(itemsets, sparse=True)
            df = pd.DataFrame.sparse.from_spmatrix(te_ary, columns=te.columns_)
            rules = apriori(df, min_support=0.001, max_len=5, use_colnames=True)
            score, bc, size, score_2, rec, mat = TL(name, 20, rules, smote=True, act=par)
            scores_t.append(score)
            bcs_t.append(bc)
            size_t.append(size)
            score_2t.append(score_2)
            records2.append(rec)
            con_matrix1.append(mat)

    # Classical LIME planner
    paras = [False]
    explainer = None
    scores_f, bcs_f = [], []
    size_f, score_2f = [], []
    cm_f = []
    for par in paras:
        for name in fnames:
            score, bc, size, score_2, rec, matrix = planner(name, 20, explainer, smote=True, small=.03, act=par)
            scores_f.append(score)
            bcs_f.append(bc)
            size_f.append(size)
            score_2f.append(score_2)
            cm_f.append(matrix)

    # Random planner
    scores_rw, bcs_rw = [], []
    size_rw, score2_rw = [], []
    numbers = [4, 3, 5, 5, 5, 5, 4, 4, 5]
    cm_rw = []
    i = 0
    for name in fnames:
        score, bc, size, score_2, matrix = RW(name, 20, explainer, smote=False, small=.03, act=False, number=numbers[i])
        i += 1
        scores_rw.append(score)
        bcs_rw.append(bc)
        size_rw.append(size)
        score2_rw.append(score_2)
        cm_rw.append(matrix)

    # Alves
    scores_alve, bcs_alve, sizes_alve, scores2_alve = [], [], [], []
    cm_alve = []
    for name in fnames:
        matrix = []
        score, bc, size, score2, matrix = runalves(name, thresh=0.95)
        scores_alve.append(score)
        bcs_alve.append(bc)
        sizes_alve.append(size)
        scores2_alve.append(score2)
        cm_alve.append(matrix.copy())

    # Shatnawi
    scores_shat, bcs_shat, sizes_shat, scores2_shat = [], [], [], []
    cm_shat = []
    for name in fnames:
        matrix = []
        score, bc, size, score2, matrix = runshat(name, 0.5)
        scores_shat.append(score.copy())
        bcs_shat.append(bc)
        sizes_shat.append(size)
        scores2_shat.append(score2)
        cm_shat.append(matrix.copy())

    # Oliveira
    scores_oliv, bcs_oliv, sizes_oliv, scores2_oliv = [], [], [], []
    cm_oliv = []
    for name in fnames:
        matrix = []
        score, bc, size, score2, matrix = runolive(name)
        scores_oliv.append(score)
        bcs_oliv.append(bc)
        sizes_oliv.append(size)
        scores2_oliv.append(score2)
        cm_oliv.append(matrix.copy())

    # XTREE
    paras = [False]
    scores_x, bcs_x, sizes_x, scores2_x = [], [], [], []
    cm_x = []
    for par in paras:
        for name in fnames:
            score_x, bc_x, size_x, score2, matrix = xtree(name)
            scores_x.append(score_x)
            bcs_x.append(bc_x)
            sizes_x.append(size_x)
            scores2_x.append(score2)
            cm_x.append(matrix)
            pd.DataFrame(matrix).to_csv('cm_x' + str(i) + '.csv')

    pd.DataFrame(score_2t).to_csv("results/rq1_TimeLIME.csv")
    pd.DataFrame(score_2f).to_csv("results/rq1_LIME.csv")
    pd.DataFrame(scores2_x).to_csv("results/rq1_XTREE.csv")
    pd.DataFrame(scores2_alve).to_csv("results/rq1_Alves.csv")
    pd.DataFrame(scores2_oliv).to_csv("results/rq1_Oliv.csv")
    pd.DataFrame(scores2_shat).to_csv("results/rq1_Shat.csv")
    pd.DataFrame(score2_rw).to_csv("results/rq1_Random.csv")
    pd.DataFrame(score_2cf).to_csv("results/rq1_CF.csv")

    pd.DataFrame(scores_t).to_csv("results/rq2_TimeLIME.csv")
    pd.DataFrame(scores_f).to_csv("results/rq2_LIME.csv")
    pd.DataFrame(scores_x).to_csv("results/rq2_XTREE.csv")
    pd.DataFrame(scores_alve).to_csv("results/rq2_Alves.csv")
    pd.DataFrame(scores_oliv).to_csv("results/rq2_Oliv.csv")
    pd.DataFrame(scores_shat).to_csv("results/rq2_Shat.csv")
    pd.DataFrame(scores_rw).to_csv("results/rq2_Random.csv")
    pd.DataFrame(scores_cf).to_csv("results/rq2_CF.csv")

    pd.DataFrame(bcs_t).to_csv("results/rq3_TimeLIME.csv")
    pd.DataFrame(bcs_f).to_csv("results/rq3_LIME.csv")
    pd.DataFrame(bcs_x).to_csv("results/rq3_XTREE.csv")
    pd.DataFrame(bcs_alve).to_csv("results/rq3_Alves.csv")
    pd.DataFrame(bcs_oliv).to_csv("results/rq3_Oliv.csv")
    pd.DataFrame(bcs_shat).to_csv("results/rq3_Shat.csv")
    pd.DataFrame(bcs_rw).to_csv("results/rq3_Random.csv")
    pd.DataFrame(bcs_cf).to_csv("results/rq3_CF.csv")

    pd.DataFrame(con_matrix1_cf).to_csv("results/rq3_matrix_CF.csv")
    pd.DataFrame(con_matrix1).to_csv("results/rq3_matrix_Timeline.csv")



    # plt.subplots(figsize=(7, 7))
    # plt.rcParams.update({'font.size': 16})
    # # ind=np.arange(10)
    # N = len(scores2_x)
    # width = 0.25
    # dummy0, dummy1,dummy2,dummy3,dummy4,dummy5,dummy6,dummy7 = [],[],[],[],[],[],[],[]
    # for i in range(0,len(scores2_x)):
    #     dummy0.append(np.round(1-np.mean(score_2cf[i]),3)*20)
    #     dummy1.append(np.round(1-np.mean(score_2t[i]),3)*20)
    #     dummy2.append(np.round(1-np.mean(score_2f[i]),3)*20)
    #     dummy3.append(np.round(1-np.mean(scores2_x[i]),3)*20)
    #     dummy4.append(np.round(1-np.mean(scores2_alve[i]),3)*20)
    #     dummy5.append(np.round(1-np.mean(scores2_shat[i]),3)*20)
    #     dummy6.append(np.round(1-np.mean(scores2_oliv[i]),3)*20)
    #     dummy7.append(np.round(1-np.mean(score2_rw[i]),3)*20)
    # plt.scatter(np.arange(N), dummy2, label='Classical LIME', s=100, marker='o')
    # plt.scatter(np.arange(N), dummy3, label='XTREE', s=100, marker='o')
    # plt.scatter(np.arange(N), dummy4, label='Alves', s=100, marker='o')
    # plt.scatter(np.arange(N), dummy5, label='Shatnawi', s=100, marker='o')
    # plt.scatter(np.arange(N), dummy6, label='Oliveira', s=100, marker='o')
    # plt.scatter(np.arange(N), dummy7, label='RandomWalk', s=100, marker='v')
    # plt.plot(np.arange(N), dummy1, label='TimeLIME', marker='^', markersize=10, color='#22406D')
    # plt.plot(np.arange(N), dummy0, label='CF', marker='^', markersize=10, color='#22406D')
    #
    # # plt.ylim(-11,130)
    # plt.xticks(np.arange(N), ['jedit', 'camel1', 'camel2', 'log4j', 'xalan', 'ant', 'velocity', 'poi', 'synapse'])
    # plt.yticks([0, 2, 4, 6, 8, 10, 12])
    # plt.subplots_adjust(bottom=0.2, left=0, right=1.1)
    # plt.grid(axis='y')
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=3)
    # plt.savefig("rq1", dpi=200, bbox_inches='tight')
    # plt.show()

    return


if __name__ == "__main__":
    main()
