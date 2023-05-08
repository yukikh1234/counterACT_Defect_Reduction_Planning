import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
from sklearn.ensemble import RandomForestClassifier
import lime.lime_tabular
from imblearn.over_sampling import SMOTE
from scipy import stats
import time
import itertools
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder  ## This file contains all tools needed to run planners.


def list2dataframe(lst):
    return pd.DataFrame(lst)


def prepareCommitData(fname):
    # file = fname
    file = os.path.join("Data", fname)
    df = pd.read_csv(file, sep=',')
    new_columns = ['cbo','lcom5','wmc','rfc','dit','noc','npm','avg_cc','max_cc','rfc.1','amc','loc']
    df.rename(columns=dict(zip(df.columns[2:-1], new_columns)), inplace=True)
    df.drop(columns=['rfc.1'],inplace=True)

    return df


def prepareData(fname):
    # file = name
    file = os.path.join("Data", fname)
    df = pd.read_csv(file, sep=',')
    cols = list(df.columns)
    cols[1:-1] = ['cbm', 'lcom3', 'rfc', 'max_cc', 'cbo', 'moa', 'avg_cc', 'noc', 'ce',
                  'npm', 'ca', 'mfa', 'lcom', 'amc', 'cam', 'dam', 'ic', 'wmc', 'loc',
                  'dit']
    for i in range(0, df.shape[0]):
        if df.iloc[i, -1] > 0:
            df.iloc[i, -1] = 1
        else:
            df.iloc[i, -1] = 0
    df = pd.DataFrame(df[cols])
    return df[cols]


def bugs(fname):
    # return the number of bugs in each row
    file = os.path.join("Data", fname)
    df = pd.read_csv(file, sep=',')
    return df.iloc[:, -1]


def translate1(sentence, name):
    # do not aim to change the column
    lst = sentence.strip().split(name)
    left, right = 0, 0
    if lst[0] == '':
        del lst[0]
    if len(lst) == 2:
        if '<=' in lst[1]:
            aa = lst[1].strip(' <=')
            right = float(aa)
        elif '<' in lst[1]:
            aa = lst[1].strip(' <')
            right = float(aa)
        if '<=' in lst[0]:
            aa = lst[0].strip(' <=')
            left = float(aa)
        elif '<' in lst[0]:
            aa = lst[0].strip(' <')
            left = float(aa)
    else:
        if '<=' in lst[0]:
            aa = lst[0].strip(' <=')
            right = float(aa)
            left = 0
        elif '<' in lst[0]:
            aa = lst[0].strip(' <')
            right = float(aa)
            left = 0
        if '>=' in lst[0]:
            aa = lst[0].strip(' >=')
            left = float(aa)
            right = 1
        elif '>' in lst[0]:
            aa = lst[0].strip(' >')
            left = float(aa)
            right = 1
    return left, right


def translate(sentence, name):
    # not used
    flag = 0
    threshold = 0
    lst = sentence.strip().split(name)
    #     print('LST',lst)
    if lst[0] == '':
        del lst[0]
    if len(lst) == 2:
        if '<=' in lst[1]:
            flag = 1
            aa = lst[1].strip(' <=')
            threshold1 = float(aa)
        elif '<' in lst[1]:
            flag = 1
            aa = lst[1].strip(' <')
            threshold1 = float(aa)
        if '<=' in lst[0]:
            flag = -1
            aa = lst[0].strip(' <=')
            threshold0 = float(aa)
        elif '<' in lst[0]:
            flag = -1
            aa = lst[0].strip(' <')
            threshold0 = float(aa)
        if threshold0 == 0:
            result = threshold1
            flag = 1
        elif (1 - threshold1) >= (threshold0 - 0):
            result = threshold1
            flag = 1
        else:
            result = threshold0
            flag = -1
    else:
        if '<=' in lst[0]:
            flag = 1
            aa = lst[0].strip(' <=')
            threshold = float(aa)
        elif '<' in lst[0]:
            flag = 1
            aa = lst[0].strip(' <')
            threshold = float(aa)
        if '>=' in lst[0]:
            flag = -1
            aa = lst[0].strip(' >=')
            threshold = float(aa)
        elif '>' in lst[0]:
            flag = -1
            aa = lst[0].strip(' >')
            threshold = float(aa)
        result = threshold
    return flag, result


def get_index(name):
    feature = ['cbm', 'lcom3', 'rfc', 'max_cc', 'cbo', 'moa', 'avg_cc', 'noc', 'ce',
               'npm', 'ca', 'mfa', 'lcom', 'amc', 'cam', 'dam', 'ic', 'wmc', 'loc',
               'dit']
    # feature = ['cbo','lcom5','wmc','rfc','dit','noc','npm','avg_cc','max_cc','amc','loc']
    for i in range(len(feature)):
        if name == feature[i]:
            return i
    return -1


def merge_plan_with_origin(plan, test):
    # Merge the two dataframes based on the common column
    result = [x for x in test.columns if x not in plan.columns]

    for col in result:
        plan[col] = test[col]
        plan[col] = plan[col].apply(lambda x: (x[0] - 0.05, x[1] + 0.05))
    return plan


def flip(data_row, local_exp, ind, clf, cols, n_feature=5, par=20,actionable=None):

    counter = 0
    rejected = 0
    cache = []
    trans = []
    # Store feature index in cache.
    cnt, cntp, cntn = [], [], []
    for i in range(0, len(local_exp)):
        cache.append(ind[i])
        trans.append(local_exp[i])
        if ind[i][1] > 0.01:
            cntp.append(i)
            cnt.append(i)
        elif ind[i][1] < -0.01:
            cntn.append(i)
            cnt.append(i)
    #         if ind[i][1]>0:
    #             cnt.append(i)
    record = [0 for n in range(par)]
    tem = data_row.copy()
    result = [[0 for m in range(2)] for n in range(par)]
    for j in range(0, len(local_exp)):
        act = True
        index = get_index(cols[cache[j][0]])
        if actionable:
            if actionable[index] == 0:
                act = False
        l, r = translate1(trans[j][0], cols[cache[j][0]])
        if j in cnt and counter < n_feature and act:
            if j in cntp:
                result[cache[j][0]][0], result[cache[j][0]][1] = 0, tem[index]
                record[index] = -1
            else:
                result[cache[j][0]][0], result[cache[j][0]][1] = tem[index], 1
                record[index] = 1
            #             if (l+r)/2<0.5:
            #                 if r+r-l<=1:
            #                     result[cache[j][0]][0],result[cache[j][0]][1] = r,r+(r-l)
            #                 else:
            #                     result[cache[j][0]][0],result[cache[j][0]][1] = r,1
            #             else:
            #                 if l-(r-l)>=0:
            #                     result[cache[j][0]][0],result[cache[j][0]][1] = l-(r-l),l
            #                 else:
            #                     result[cache[j][0]][0],result[cache[j][0]][1] = 0,l
            #             tem[cache[j][0]] = (result[cache[j][0]][0]+result[cache[j][0]][1])/2
            counter += 1
        else:
            if act:
                result[cache[j][0]][0], result[cache[j][0]][1] = tem[index] - 0.005, tem[index] + 0.005
            else:
                result[cache[j][0]][0], result[cache[j][0]][1] = tem[index] - 0.05, tem[index] + 0.05
    #             tem[index]-0.05,tem[index]+0.05
    return tem, result, record


def hedge(arr1, arr2):
    # returns a value, larger means more changes
    s1, s2 = np.std(arr1), np.std(arr2)
    m1, m2 = np.mean(arr1), np.mean(arr2)
    n1, n2 = len(arr1), len(arr2)
    num = (n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2
    denom = n1 + n2 - 1 - 1
    sp = (num / denom) ** .5
    delta = np.abs(m1 - m2) / sp
    c = 1 - 3 / (4 * (denom) - 1)
    return delta * c


def norm(df1, df2):
    # min-max scale the dataset
    X1 = df1.iloc[:, :-1].values
    mm = MinMaxScaler()
    mm.fit(X1)
    X2 = df2.iloc[:, :-1].values
    X2 = mm.transform(X2)
    df2 = df2.copy()
    df2.iloc[:, :-1] = X2
    return df2


def overlapCF(plan, actual):
    cnt = 20
    same = 0
    # metrics = plan.columns
    # rest = cnt - len(metrics)
    overlaps = []

    for (_, CF) in plan.iterrows():
        for (_, d1), (_, d2) in zip(CF.iteritems(), actual.iteritems()):
            if isinstance(d1, tuple):
                # The change recommended lie in a range of values
                if d1[0] <= d2.values[0] <= d1[1]:
                    same += 1

                elif d2.values[0] > 1:
                    if d1[1] >= 1:
                        same += 1
                else:
                    if d1[0] <= 0:
                        same += 1

        overlap = same / cnt
        overlaps.append(overlap)
        same = 0
    return max(overlaps), overlaps


def overlap(plan, actual):  # Jaccard similarity function
    cnt = 20
    right = 0
    # print(plan)
    for i in range(0, 20):
        if isinstance(plan[i], float):
            if np.round(actual[i], 4) == np.round(plan[i], 4):
                right += 1
        else:
            if actual[i] >= 0 and actual[i] <= 1:
                if actual[i] >= plan[i][0] and actual[i] <= plan[i][1]:
                    right += 1
            elif actual[i] > 1:
                if plan[i][1] >= 1:
                    right += 1
            else:
                if plan[i][0] <= 0:
                    right += 1
    return right / cnt


def similar(ins):
    out = []
    for i in range(0, len(ins.as_list(label=1))):
        out.append(ins.as_list(label=1)[i][0])
    return out


def overlap1(orig, plan, actual):  # Jaccard similarity function
    cnt = 20
    right = 0
    for i in range(0, cnt):
        if isinstance(plan[i], list):
            if actual[i] >= plan[i][0] and actual[i] <= plan[i][1]:
                right += 1
        else:
            if actual[i] == plan[i]:
                right += 1
    return right / cnt


def size_interval(plan):
    out = []
    for i in range(len(plan)):
        if not isinstance(plan[i], float):
            out.append(plan[i][1] - plan[i][0])
        else:
            out.append(0)
    return out


def apply3(row, cols, pk_best):
    newRow = row
    rec = [0] * 20
    for idx, col in enumerate(cols):
        try:
            thres = pk_best[col][1]
            proba = pk_best[col][0]
            if thres is not None:
                if newRow[idx] > thres:
                    rec[idx] = 1
                    #                     print("Yes",thres,proba)
                    newRow[idx] = (0, thres)
        #                     if random(0, 100) < proba else \newRow[idx]
        except:
            pass
    return newRow, rec


def apply2(changes, row):
    rec = [0] * 20
    new_row = row
    for idx, thres in enumerate(changes):
        if thres is not None:
            try:
                if new_row[idx] > thres:
                    rec[idx] = 1
                    new_row[idx] = (0, thres)
            except:
                pass
    return new_row, rec


def apply4(changes, row):
    rec = [0 for i in range(20)]
    new_row = row
    for idx, thres in enumerate(changes):
        if thres is not None:
            try:
                if new_row[idx] > thres:
                    rec[idx] = 1
                    new_row[idx] = (0, thres)
            except:
                pass

    # delta = np.array(new_row) - np.array(row)
    # delta_bool = [1 if a > 0 else -1 if a < 0 else 0 for a in delta]
    return new_row, rec


def cf(lst):
    res = []
    for i in range(len(lst)):
        res.append([float(each) for each in lst[i].strip('[]').split(',')])
    return res


def readfile(fname):
    file = pd.read_csv(fname, sep=',')
    file.drop(columns=file.columns[0], inplace=True)
    result = []
    N = file.shape[0]
    for i in range(N):
        temp = []
        for k in range(file.shape[1]):

            if pd.isnull(file.iloc[i, k]):
                continue
            else:
                temp.append(file.iloc[i, k])
        result.append(temp)
    return result


def track1(old, new):
    rec = []
    for i in range(len(old)):
        if old[i][0] <= new[i] <= old[i][1]:
            rec.append(0)
        elif old[i][0] > new[i]:
            rec.append(-1)
        else:
            rec.append(1)
    return rec


def track(old, new):
    rec = []
    for i in range(len(old)):
        if old[i] != new[i]:
            if new[i] > old[i]:
                rec.append(1)
            else:
                rec.append(-1)
        else:
            rec.append(0)
    return rec


def frequentSet(name):
    start_time = time.time()
    files = [name[0], name[1], name[2]]
    df1 = prepareData(name[0])
    df2 = prepareData(name[1])
    df3 = prepareData(name[2])
    bug1 = bugs(name[0])
    bug2 = bugs(name[1])
    bug3 = bugs(name[2])
    df11 = df1.iloc[:, 1:]
    df22 = df2.iloc[:, 1:]
    df33 = df3.iloc[:, 1:]

    df1n = norm(df11, df11)
    df2n = norm(df11, df22)
    df3n = norm(df11, df33)

    X_train1 = df1n.iloc[:, :-1]
    y_train1 = df1n.iloc[:, -1]
    X_test1 = df2n.iloc[:, :-1]
    y_test1 = df2n.iloc[:, -1]
    X_test2 = df3n.iloc[:, :-1]
    y_test2 = df3n.iloc[:, -1]
    records = []
    for i in range(0, len(y_test1)):
        for j in range(0, len(y_test2)):
            if df3.iloc[j, 0] == df2.iloc[i, 0]:
                actual = X_test2.values[j]
                old = X_test1.values[i]
                rec = track(old, actual)
                records.append(rec)
                break

    for i in range(0, len(y_train1)):
        for j in range(0, len(y_test1)):
            if df2.iloc[j, 0] == df1.iloc[i, 0]:
                actual = X_test1.values[j]
                old = X_train1.values[i]
                rec = track(old, actual)
                records.append(rec)
                break
    return records


def transform(df2, lo, col):
    # transform single column of index 'col' by discretized data
    df22 = df2.copy()
    low = lo[col]
    start = low[0]
    for i in range(df22.shape[0]):
        for j in range(len(low)):
            if df22.iloc[i, col] >= low[j]:
                start = low[j]
            if df22.iloc[i, col] <= low[j]:
                end = low[j]
                df22.iloc[i, col] = (start + end) / 2
                break
    return df22


def abcd(ori, plan, actual, act):
    #     print(act)
    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(len(ori)):
        if act[i] != 0:
            if isinstance(plan[i], tuple) or isinstance(plan[i], list):
                if plan[i][0] <= actual[i] <= plan[i][1]:
                    tp += 1
                else:
                    fp += 1
            else:
                if plan[i] == actual[i]:
                    tp += 1
                else:
                    fp += 1
        else:
            if isinstance(plan[i], tuple) or isinstance(plan[i], list):
                if plan[i][0] <= actual[i] <= plan[i][1]:
                    tn += 1
                else:
                    fn += 1
            else:
                if plan[i] == actual[i]:
                    tn += 1
                else:
                    fn += 1

    return tp, tn, fp, fn


def convert_to_itemset(df):
    itemsets = []
    for i in range(df.shape[0]):
        item = []
        temp = df.iloc[i, :]
        for j in range(df.shape[1]):
            if temp[j] == 1:
                item.append("inc" + str(j))
            elif temp[j] == -1:
                item.append("dec" + str(j))
        if len(item) > 0:
            itemsets.append(item)
    return itemsets


def mine_rules(itemsets):
    test = itemsets.copy()
    te = TransactionEncoder()
    te_ary = te.fit(test).transform(test, sparse=True)
    df = pd.DataFrame.sparse.from_spmatrix(te_ary, columns=te.columns_)
    rules = apriori(df, min_support=0.001, use_colnames=True)
    return rules


def get_support(string, rules):
    for i in range(rules.shape[0]):
        if set(rules.iloc[i, 1]) == set(string):
            return rules.iloc[i, 0]
    return 0


def find_supported_plan(plan, rules, top=5):
    proposed = []
    max_change = top
    max_sup = 0
    result_id = []
    pool = []
    for j in range(len(plan)):
        if plan[j] == 1:
            result_id.append(j)
            proposed.append("inc" + str(j))
        elif plan[j] == -1:
            result_id.append(-j)
            proposed.append("dec" + str(j))
    #     if max_change==top:
    #         max_sup = get_support(proposed,rules)
    while (max_sup == 0):
        pool = list(itertools.combinations(result_id, max_change))
        for each in pool:
            temp = []
            for k in range(len(each)):
                if each[k] > 0:
                    temp.append("inc" + str(each[k]))
                elif each[k] < 0:
                    temp.append("dec" + str(-each[k]))
            #             print('temp',temp)
            temp_sup = get_support(temp, rules)
            if temp_sup > max_sup:
                max_sup = temp_sup
                result_id = each
        max_change -= 1
        if max_change <= 0:
            print("Failed!!!")
            break
    return result_id
