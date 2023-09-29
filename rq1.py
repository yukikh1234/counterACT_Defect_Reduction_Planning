from planner import *
from othertools import *
import matplotlib.pyplot as plt


def main():
    score_2t = readfile('results/rq1_TimeLIME.csv')
    score_2f = readfile('results/rq1_LIME.csv')
    scores2_x = readfile('results/rq1_XTREE.csv')
    scores2_alve = readfile('results/rq1_Alves.csv')
    scores2_shat = readfile('results/rq1_Shat.csv')
    scores2_oliv = readfile('results/rq1_Oliv.csv')
    score2_rw = readfile('results/rq1_Random.csv')
    score2_CF = readfile('results/rq1_CF.csv')

    # plt.subplots(figsize=(7, 7))
    # plt.rcParams.update({'font.size': 16})
    # ind=np.arange(10)
    N = len(scores2_x)
    print(N)
    width = 0.25
    dummy1, dummy2, dummy3, dummy4, dummy5, dummy6, dummy7, dummy8 = [], [], [], [], [], [], [], []
    for i in range(0, len(scores2_x)):
        dummy1.append(np.round(np.mean(score_2t[i]), 3))
        dummy2.append(np.round(np.mean(score_2f[i])*6, 3))
        dummy3.append(np.round(np.mean(scores2_x[i])*10, 3))
        dummy4.append(np.round(np.mean(scores2_alve[i])*10, 3))
        dummy5.append(np.round(np.mean(scores2_shat[i]), 3))
        dummy6.append(np.round(np.mean(scores2_oliv[i])*10, 3))
        dummy7.append(np.round(np.mean(score2_rw[i])*10, 3))
        dummy8.append(np.round(np.mean(score2_CF[i]), 3))
    result = [dummy1, dummy2, dummy3, dummy4, dummy5, dummy6, dummy7, dummy8]
    fig = plt.figure(figsize=(14, 8))
    plt.scatter(np.arange(N), dummy2, label='Classical LIME', s=100, marker='o')
    plt.scatter(np.arange(N), dummy3, label='XTREE', s=100, marker='o')
    plt.scatter(np.arange(N), dummy4, label='Alves', s=100, marker='o')
    plt.scatter(np.arange(N), dummy5, label='Shatnawi', s=100, marker='o')
    plt.scatter(np.arange(N), dummy6, label='Oliveira', s=100, marker='o')
    plt.scatter(np.arange(N), dummy7, label='RandomWalk', s=100, marker='v')
    plt.scatter(np.arange(N), dummy1, label='TimeLIME', marker='^', s=100, color='#22406D')
    plt.plot(np.arange(N), dummy8, label='CounterACT', marker='o', markersize=10, color='#22406D')

    # plt.ylim(-11,130)
    plt.xticks(np.arange(N), ['jedit', 'camel1', 'camel2', 'log4j', 'xalan', 'ant',
                'velocity', 'poi', 'synapse'], fontsize=12)
    plt.yticks([0, 2, 4, 6, 8, 10, 12],fontsize=12)
    plt.subplots_adjust(bottom=0.2)
    plt.grid(axis='y')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True,
               shadow=True, ncol=3,
               fontsize=14)
    plt.savefig("rq1.pdf", dpi=200, bbox_inches='tight', format='pdf')
    plt.show()

    return result


if __name__ == "__main__":
    names = np.array(['TimeLIME', 'LIME', 'Random', 'Alves', 'Shatnawi', 'Oliveira', 'XTREE', 'CF'])
    projects = ['jedit', 'camel1', 'camel2', 'log4j', 'xalan', 'ant',
                'velocity', 'poi', 'synapse','xlan\nant','log4j\nant','camel\nlog4j',
                'velocity\nsynapse', 'jedit\npoi', 'synapse\nxalan']

    results = main()
    result = pd.DataFrame(results)
    print(result)
    result.columns = projects
    result = result.set_index(names)
    print(result)
    # result.to_excel('results/RQ1.xlsx')
