import lime.discretize

from othertools import *
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import f_classif
from XTREE import XTREE
import random
from actionrules.actionRulesDiscovery import ActionRulesDiscovery
from tqdm import tqdm
import ast

from sklearn.preprocessing import MinMaxScaler

# print the full dataframe
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
# pd.set_option('display.width', None)
# pd.set_option('display.max_colwidth', -1)


def RandomWalk(data_row, number):
    tem = data_row.copy()
    result = [[0 for m in range(2)] for n in range(20)]
    lis = list(np.arange(20))
    act = random.sample(lis, number)
    rec = [0] * 20
    for j in range(0, len(tem)):
        if j in act:
            rec[j] = 1
            num1 = np.random.rand(1)[0]
            num2 = np.random.rand(1)[0]
            if (num1 <= num2 and tem[j] != 0) or tem[j] == 1:
                result[j][0], result[j][1] = 0, tem[j] - 0.05
            else:
                result[j][0], result[j][1] = tem[j] + 0.05, 1
            tem[j] = (num1 + num2) / 2
        else:
            result[j][0], result[j][1] = tem[j] - 0.05, tem[j] + 0.05
    return tem, result, rec


def RW(name, par, explainer=None, smote=False, small=0.05, act=False, number=5):
    files = [name[0], name[1], name[2]]
    freq = [0] * 20
    deltas = []
    for j in range(0, len(files) - 2):
        df1 = prepareData(files[j])
        df2 = prepareData(files[j + 1])
        for i in range(1, 21):
            col1 = df1.iloc[:, i]
            col2 = df2.iloc[:, i]
            deltas.append(hedge(col1, col2))
    #             if not (hedge(col1,col2,small)):
    #                 freq[i-1]+=1
    deltas = sorted(range(len(deltas)), key=lambda k: deltas[k], reverse=True)
    #     changed = dict()
    #     for i in range(20):
    #         freq[i] = 100*freq[i]/(len(files)-1)
    #         changed.update({df1.columns[i]:freq[i]})
    #     changed = list(changed.values())
    #     actionable = []
    #     for each in changed:
    #         actionable.append(1) if each!=0 else actionable.append(0)
    actionable = []
    for i in range(0, len(deltas)):
        if i in deltas[0:5]:
            actionable.append(1)
        else:
            actionable.append(0)
    print(actionable)
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
    df2n = norm(df22, df22)
    df3n = norm(df22, df33)

    X_train1 = df1n.iloc[:, :-1]
    y_train1 = df1n.iloc[:, -1]
    X_test1 = df2n.iloc[:, :-1]
    y_test1 = df2n.iloc[:, -1]
    X_test2 = df3n.iloc[:, :-1]
    y_test2 = df3n.iloc[:, -1]

    score = []
    bugchange = []
    size = []
    score2 = []
    matrix = []
    para = 20
    clf1 = RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_split=10, n_jobs=-1)
    #     clf1 =  MLPClassifier(hidden_layer_sizes=[10],max_iter=2000,early_stopping=False,learning_rate='adaptive')
    if smote:
        sm = SMOTE()
        X_train1_s, y_train1_s = sm.fit_resample(X_train1, y_train1)
        clf1.fit(X_train1_s, y_train1_s)
    else:
        clf1.fit(X_train1, y_train1)
    for i in range(0, len(y_test1)):
        for j in range(0, len(y_test2)):
            actual = X_test2.values[j]
            if df3.iloc[j, 0] == df2.iloc[i, 0] and y_test1[i] != 0:
                print('df2', i, 'df3', j)
                temp = X_test1.values[i].copy()
                tem, plan, rec = RandomWalk(temp, number)
                score.append(overlap(plan, actual))
                score2.append(overlap(plan, X_test1.values[i]))
                size.append(size_interval(plan))
                bugchange.append(bug3[j] - bug2[i])  # negative if reduced #bugs, positive if added
                tp, tn, fp, fn = abcd(temp, plan, actual, rec)
                matrix.append([tp, tn, fp, fn])
                break
    print(name[0])
    print('>>>')
    print('>>>')
    print('>>>')
    return score, bugchange, size, score2, matrix


def planner(name, par, explainer=None, smote=False, small=0.05, act=False):
    # classic LIME
    start_time = time.time()
    files = [name[0], name[1], name[2]]
    freq = [0] * 20
    deltas = []
    for j in range(0, len(files) - 2):
        df1 = prepareData(files[j])
        df2 = prepareData(files[j + 1])
        for i in range(1, 21):
            col1 = df1.iloc[:, i]
            col2 = df2.iloc[:, i]
            deltas.append(hedge(col1, col2))
    deltas = sorted(range(len(deltas)), key=lambda k: deltas[k], reverse=True)

    actionable = []
    for i in range(0, len(deltas)):
        if i in deltas[0:5]:
            actionable.append(1)
        else:
            actionable.append(0)
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
    df2n = norm(df22, df22)
    df3n = norm(df22, df33)

    X_train1 = df1n.iloc[:, :-1]
    y_train1 = df1n.iloc[:, -1]
    X_test1 = df2n.iloc[:, :-1]
    y_test1 = df2n.iloc[:, -1]
    X_test2 = df3n.iloc[:, :-1]
    y_test2 = df3n.iloc[:, -1]

    score = []
    bugchange = []
    size = []
    score2 = []
    records = []
    matrix = []
    par = 20
    clf1 = RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_split=10, n_jobs=-1)
    #     clf1 =  MLPClassifier(hidden_layer_sizes=[10],max_iter=2000,early_stopping=False,learning_rate='adaptive')
    if smote:
        sm = SMOTE()
        X_train1_s, y_train1_s = sm.fit_resample(X_train1, y_train1)
        clf1.fit(X_train1_s, y_train1_s)
        explainer = lime.lime_tabular.LimeTabularExplainer(training_data=X_train1_s.values, training_labels=y_train1_s.values,
                                                           feature_names=df11.columns,
                                                           discretizer='entropy', feature_selection='lasso_path',
                                                           mode='classification')
    else:
        clf1.fit(X_train1, y_train1)
        explainer = lime.lime_tabular.LimeTabularExplainer(training_data=X_train1.values, training_labels=y_train1,
                                                           feature_names=df11.columns,
                                                           discretizer='entropy', feature_selection='lasso_path',
                                                           mode='classification')
    for i in range(0, len(y_test1)):
        for j in range(0, len(y_test2)):
            actual = X_test2.values[j]
            if df3.iloc[j, 0] == df2.iloc[i, 0] and y_test1[i] != 0:

                if True:
                    ins = explainer.explain_instance(data_row=X_test1.values[i], predict_fn=clf1.predict_proba,
                                                     num_features=20,
                                                     num_samples=5000)
                    ind = ins.local_exp[1]
                    temp = X_test1.values[i].copy()
                    if act:
                        tem, plan, rec = flip(temp, ins.as_list(label=1), ind, clf1, df1n.columns, par,
                                              actionable=actionable)
                    else:
                        tem, plan, rec = flip(temp, ins.as_list(label=1), ind, clf1, df1n.columns, par, actionable=None)
                    score.append(overlap(plan, actual))
                    size.append(size_interval(plan))
                    score2.append(overlap(plan, temp))
                    records.append(rec)
                    tp, tn, fp, fn = abcd(temp, plan, actual, rec)
                    matrix.append([tp, tn, fp, fn])
                    bugchange.append(bug3[j] - bug2[i])  # negative if reduced #bugs, positive if added
                break
    print("Runtime:", time.time() - start_time)
    print(name[0], par)
    print('>>>')
    print('>>>')
    print('>>>')
    return score, bugchange, size, score2, records, matrix


def CF(name, number_act=5):
    start_time = time.time()
    files = [name[0], name[1], name[2]]
    freq = [0] * 20
    deltas = []
    score = []
    size = []
    score2 = []
    records = []
    matrix = []
    bugchange = []

    for j in range(0, len(files) - 2):
        df1 = prepareData(files[j])
        df2 = prepareData(files[j + 1])
        for i in range(1, 21):
            col1 = df1.iloc[:, i]
            col2 = df2.iloc[:, i]
            deltas.append(hedge(col1, col2))
    deltas = sorted(range(len(deltas)), key=lambda k: deltas[k], reverse=True)

    actionable = []
    for i in range(0, len(deltas)):
        if i in deltas[0:number_act]:
            actionable.append(1)
        else:
            actionable.append(0)

    actionable_set_names = [list(df1.columns)[1:-1][i] for i in range(len(df1.columns[1:-1])) if actionable[i] == 1]

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
    df2n = norm(df22, df22)
    df3n = norm(df22, df33)

    X_train1 = df1n.iloc[:, :-1]
    y_train1 = df1n.iloc[:, -1]
    X_test1 = df2n.iloc[:, :-1]
    y_test1 = df2n.iloc[:, -1]
    X_test2 = df3n.iloc[:, :-1]
    y_test2 = df3n.iloc[:, -1]

    # Learn the action rules
    # First we descritize the data using the Entropy algorithm
    X_train1_to_disc = X_train1.copy()
    X_train1_to_disc['version'] = '2.4'
    X_test1_to_disc = X_test1.copy()
    X_test1_to_disc['version'] = '2.5'
    train = pd.concat([X_train1_to_disc, X_test1_to_disc]).reset_index(drop=True)
    versions = train['version']
    train = train.drop(columns=['version'])
    defects = pd.concat([y_train1, y_test1]).reset_index(drop=True)

    desc = lime.discretize.EntropyDiscretizer(train.values, categorical_features=[], labels=defects.values,
                                              feature_names=X_train1.columns)
    names = desc.names

    mapping_tuple = {}
    for i in range(X_train1.columns.shape[0]):
        col = X_train1.columns[i]
        tuples = {}
        for index, value in enumerate(names[i]):
            left, right = translate1(value, col)
            tuples[index] = (left - 0.005, right + 0.005)

        mapping_tuple[col] = tuples

    discretize_train_data = desc.discretize(train.values)
    discretize_train_data = pd.DataFrame(discretize_train_data, columns=train.columns)

    for key, values in mapping_tuple.items():
        discretize_train_data[key] = discretize_train_data[key].map(values)

    discretize_train_data['bug'] = defects
    actionRulesDiscovery = ActionRulesDiscovery()
    actionRulesDiscovery.load_pandas(discretize_train_data)

    print('-------- Learn Action rules start ----------')
    actionRulesDiscovery.fit(stable_attributes=[],
                             flexible_attributes=actionable_set_names,
                             consequent="bug",
                             conf=40,
                             supp=1,
                             desired_classes=["0"],
                             is_nan=False,
                             is_reduction=True,
                             min_stable_attributes=0,
                             min_flexible_attributes=1,
                             max_stable_attributes=0,
                             max_flexible_attributes=5)

    discretize_train_data['version'] = versions
    X_test1 = discretize_train_data[discretize_train_data['version'] == "2.5"].drop(
        columns=['bug', 'version']).reset_index(drop=True)

    for i in tqdm(range(0, len(y_test1))):
        for j in range(0, len(y_test2)):
            actual = X_test2.iloc[j].to_frame().T
            if df3.iloc[j, 0] == df2.iloc[i, 0] and y_test1[i] != 0:
                if True:
                    defect = X_test1.iloc[i].to_frame().T
                    try:
                        plan = actionRulesDiscovery.predict(defect)
                    except:
                        continue
 

                    recommended_plan = plan[plan['support after'] == plan['support after'].max()]
                    recommended_plan = recommended_plan.filter(regex='recommended')

                    columns = [s.replace('-recommended', "") for s in recommended_plan.columns]

                    tmp_recommended = recommended_plan[
                        [col for col in recommended_plan.columns if 'recommended' in col]].dropna(axis=1)

                    rec_col = [s.replace('-recommended', "") for s in tmp_recommended.columns]

                    # Impute the missing data,if the value is Nan it mean you don't change it so with fill it with the original value
                    for col in columns:
                        try:
                            recommended_plan['{}-recommended'.format(col)] = recommended_plan[
                                '{}-recommended'.format(col)].apply(
                                lambda x: defect[col].values[0] if x is np.nan else ast.literal_eval(x))
                        except:
                            continue

                    # ADD the not changed metrics to the plan to compute the overlap
                    recommended_plan.columns = columns
                    recommended_plan = merge_plan_with_origin(recommended_plan, defect)
                    recommended_plan = recommended_plan.reindex(columns=actual.columns.to_list())

                    max_overlap, list_overlap = overlapCF(recommended_plan, actual)
                    rec = [1 if item in rec_col else 0 for item in defect.columns]
                    size.append(size_interval([ast.literal_eval(x) for x in tmp_recommended.values[0]]))
                    score2.append(len([n for n in rec if n != 0]))
                    records.append(rec)

                    tp, tn, fp, fn = abcd(defect.values[0], recommended_plan.values[0], actual.values[0], actionable)
                    matrix.append([tp, tn, fp, fn])

                    bugchange.append(bug3[j] - bug2[i])  # negative if reduced #bugs, positive if added

                    score.append(max_overlap)
    print("Runtime:", time.time() - start_time)
    print(name[0])
    print('>>>')
    print('>>>')
    print('>>>')
    print(np.median(score))
    return score, bugchange, size, score2, records, matrix


def TL(name, par, rules, smote=False, act=False):
    start_time = time.time()
    files = [name[0], name[1], name[2]]
    freq = [0] * 20
    deltas = []
    for j in range(0, len(files) - 2):
        df1 = prepareData(files[j])
        df2 = prepareData(files[j + 1])
        for i in range(1, 21):
            col1 = df1.iloc[:, i]
            col2 = df2.iloc[:, i]
            deltas.append(hedge(col1, col2))
    deltas = sorted(range(len(deltas)), key=lambda k: deltas[k], reverse=True)

    actionable = []
    for i in range(0, len(deltas)):
        if i in deltas[0:5]:
            actionable.append(1)
        else:
            actionable.append(0)

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
    df2n = norm(df22, df22)
    df3n = norm(df22, df33)

    X_train1 = df1n.iloc[:, :-1]
    y_train1 = df1n.iloc[:, -1]
    X_test1 = df2n.iloc[:, :-1]
    y_test1 = df2n.iloc[:, -1]
    X_test2 = df3n.iloc[:, :-1]
    y_test2 = df3n.iloc[:, -1]

    plans = []
    actuals = []
    score = []
    bugchange = []
    size = []
    score2 = []
    records = []
    matrix = []
    seen = []
    seen_id = []
    par = 20
    clf1 = RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_split=10, n_jobs=-1)
    #     clf1 =  MLPClassifier(hidden_layer_sizes=[10],max_iter=2000,early_stopping=False,learning_rate='adaptive')
    #     clf1 = SVC(gamma='auto',probability=True)
    if smote:
        sm = SMOTE()
        X_train1_s, y_train1_s = sm.fit_resample(X_train1, y_train1)
        clf1.fit(X_train1_s, y_train1_s)

        explainer = lime.lime_tabular.LimeTabularExplainer(training_data=X_train1_s.values,
                                                           training_labels=y_train1_s.values,
                                                           feature_names=df11.columns,
                                                           discretizer='entropy', feature_selection='lasso_path',
                                                           mode='classification')
    else:
        clf1.fit(X_train1, y_train1)
        explainer = lime.lime_tabular.LimeTabularExplainer(training_data=X_train1.values, training_labels=y_train1,
                                                           feature_names=df11.columns,
                                                           discretizer='entropy', feature_selection='lasso_path',
                                                           mode='classification')
    for i in tqdm(range(0, len(y_test1))):
        for j in range(0, len(y_test2)):
            actual = X_test2.values[j]
            if df3.iloc[j, 0] == df2.iloc[i, 0] and y_test1[i] != 0:
                #                 print('df2',i,'df3',j)
                #                 if clf1.predict([X_test1.values[i]])==0:
                if True:
                    ins = explainer.explain_instance(data_row=X_test1.values[i], predict_fn=clf1.predict_proba,
                                                     num_features=20,
                                                     num_samples=5000)
                    ind = ins.local_exp[1]
                    temp = X_test1.values[i].copy()
                    if act:

                        tem, plan, rec = flip(temp, ins.as_list(label=1), ind, clf1, df1n.columns, par,
                                              actionable=actionable)
                    else:
                        tem, plan, rec = flip(temp, ins.as_list(label=1), ind, clf1, df1n.columns, par, actionable=None)
                    if act:
                        if rec in seen_id:
                            supported_plan_id = seen[seen_id.index(rec)]
                        else:
                            #                             if seen_id:
                            #                                 for i in range(len(seen_id)):
                            #                                     print(rec == seen_id[i])
                            supported_plan_id = find_supported_plan(rec, rules, top=5)
                            seen_id.append(rec.copy())
                            seen.append(supported_plan_id)

                        for k in range(len(rec)):
                            if rec[k] != 0:
                                if (k not in supported_plan_id) and ((0 - k) not in supported_plan_id):
                                    plan[k][0], plan[k][1] = tem[k], tem[k]
                                    rec[k] = 0

                    # plans.append(plan)
                    # actuals.append(actual)
                    score.append(overlap(plan, actual))
                    size.append(size_interval(plan))
                    score2.append(len([n for n in rec if n != 0]))
                    records.append(rec)
                    tp, tn, fp, fn = abcd(temp, plan, actual, rec)

                    matrix.append([tp, tn, fp, fn])
                    bugchange.append(bug3[j] - bug2[i])  # negative if reduced #bugs, positive if added
                break
    print("Runtime:", time.time() - start_time)
    print(name[0], par)
    print('>>>')
    print('>>>')
    print('>>>')
    print('>>>>>>>>>>>>>>>>>>>>>>>>' * 20)
    print(np.median(score))
    return score, bugchange, size, score2, records, matrix


def _ent_weight(X, scale):
    try:
        loc = X["loc"].values  # LOC is the 10th index position.
    except KeyError:
        try:
            loc = X["$WCHU_numberOfLinesOfCode"].values
        except KeyError:
            loc = X["$CountLineCode"]

    return X.multiply(loc, axis="index") / scale


def alves(train, X_test, y_test, thresh=0.95):
    if isinstance(X_test, list):
        test = list2dataframe(X_test)

    if isinstance(X_test, str):
        test = list2dataframe([X_test])

    if isinstance(train, list):
        train = list2dataframe(train)

    #     train.loc[train[train.columns[-1]] == 1, train.columns[-1]] = True
    #     train.loc[train[train.columns[-1]] == 0, train.columns[-1]] = False
    metrics = [met[1:] for met in train[train.columns[:-1]]]

    X = train  # Independent Features (CK-Metrics)
    changes = []

    """
    As weight we will consider
    the source lines of code (LOC) of the entity.
    """

    loc_key = "loc"
    tot_loc = train.sum()["loc"]
    X = _ent_weight(X, scale=tot_loc)

    """
    Divide the entity weight by the sum of all weights of the same system.
    """
    denom = pd.DataFrame(X).sum().values
    norm_sum = pd.DataFrame(pd.DataFrame(X).values / denom, columns=X.columns)

    """
    Find Thresholds
    """
    #     y = train[train.columns[-1]]  # Dependent Feature (Bugs)
    #     pVal = f_classif(X, y)[1]  # P-Values
    cutoff = []

    def cumsum(vals):
        return [sum(vals[:i]) for i, __ in enumerate(vals)]

    def point(array, thresh):
        for idx, val in enumerate(array):
            if val > thresh:
                return idx

    for idx in range(len(train.columns[:-1])):
        # Setup Cumulative Dist. Func.
        name = train.columns[idx]
        loc = train[loc_key].values
        vals = norm_sum[name].values
        sorted_ids = np.argsort(vals)
        cumulative = [sum(vals[:i]) for i, __ in enumerate(sorted(vals))]
        cutpoint = point(cumulative, thresh)
        cutoff.append(vals[sorted_ids[cutpoint]] * tot_loc / loc[sorted_ids[cutpoint]] * denom[idx])

    """
    Apply Plans Sequentially
    """

    modified = []
    recs = []
    for n in range(X_test.shape[0]):
        new_row, rec = apply4(cutoff, X_test.iloc[n].values.tolist())
        modified.append(new_row)
        recs.append(rec)
    return pd.DataFrame(modified, columns=X_test.columns), recs


def runalves(name, thresh=0.7):
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
    df2n = norm(df22, df22)
    df3n = norm(df22, df33)

    X_train1 = df1n.iloc[:, :-1]
    y_train1 = df1n.iloc[:, -1]
    X_test1 = df2n.iloc[:, :-1]
    y_test1 = df2n.iloc[:, -1]
    X_test2 = df3n.iloc[:, :-1]
    y_test2 = df3n.iloc[:, -1]

    plans, recs = alves(X_train1, X_test1, y_test1, thresh=thresh)
    score = []
    score2 = []
    bugchange = []
    size = []
    matrix = []
    for i in range(0, len(y_test1)):
        for j in range(0, len(y_test2)):
            actual = X_test2.values[j]
            if df3.iloc[j, 0] == df2.iloc[i, 0] and y_test1[i] != 0:
                temp = X_test1.values[i].copy()
                plan = plans.iloc[i, :].values
                rec = recs[i]
                #                 print('actual',actual)
                #                 print('id1',plan[0][0])
                score.append(overlap(plan, actual))
                score2.append(overlap(plan, X_test1.values[i]))
                bugchange.append(bug3[j] - bug2[i])  # negative if reduced #bugs, positive if added
                size.append(size_interval(plan))
                tp, tn, fp, fn = abcd(temp, plan, actual, rec)
                matrix.append([tp, tn, fp, fn])
                break
    print(name[0])
    print('>>>')
    print('>>>')
    print('>>>')
    return score, bugchange, size, score2, matrix


def VARL(coef, inter, p0=0.05):
    """
    :param coef: Slope of   (Y=aX+b)
    :param inter: Intercept (Y=aX+b)
    :param p0: Confidence Interval. Default p=0.05 (95%)
    :return: VARL threshold

              1   /     /  p0   \             \
    VARL = ----- | log | ------ | - intercept |
           slope \     \ 1 - p0 /             /

    """
    return (np.log(p0 / (1 - p0)) - inter) / coef


def shatnawi(X_train, y_train, X_test, y_test, p):
    """
    Implements shatnavi's threshold based planner.
    :param train:
    :param test:
    :param rftrain:
    :param tunings:
    :param verbose:
    :return:
    """
    "Compute Thresholds"

    if isinstance(X_test, list):
        X_test = list2dataframe(X_test)

    if isinstance(X_test, str):
        X_test = list2dataframe([X_test])

    if isinstance(X_train, list):
        X_train = list2dataframe(X_train)

    changed = []
    metrics = [str[1:] for str in X_train[X_train.columns[:]]]
    ubr = LogisticRegression(solver='lbfgs')  # Init LogisticRegressor
    inter = []
    coef = []
    pVal = []
    for i in range(len(X_train.columns)):
        X = pd.DataFrame(X_train.iloc[:, i])  # Independent Features (CK-Metrics)
        y = y_train  # Dependent Feature (Bugs)
        ubr = LogisticRegression(solver='lbfgs')  # Init LogisticRegressor
        ubr.fit(X, y)  # Fit Logit curve
        inter.append(ubr.intercept_[0])  # Intercepts
        coef.append(ubr.coef_[0])  # Slopes
        pVal.append(f_classif(X, y)[1])  # P-Values
    changes = len(metrics) * [-1]
    "Find Thresholds using VARL"
    for Coeff, P_Val, Inter, idx in zip(coef, pVal, inter,
                                        range(len(metrics))):  # range(len(metrics)):
        thresh = VARL(Coeff, Inter, p0=p)  # default VARL p0=0.05 (95% CI)
        if P_Val < 0.05:

            changes[idx] = thresh
    """
    Apply Plans Sequentially
    """
    modified = []
    recs = []
    for n in range(X_test.shape[0]):
        new_row, rec = apply2(changes, X_test.iloc[n, :].values.tolist())
        modified.append(new_row)
        recs.append(rec)
    return pd.DataFrame(modified, columns=X_test.columns), recs


def runshat(name, p=0.05):
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
    df2n = norm(df22, df22)
    df3n = norm(df22, df33)

    X_train1 = df1n.iloc[:, :-1]
    y_train1 = df1n.iloc[:, -1]
    X_test1 = df2n.iloc[:, :-1]
    y_test1 = df2n.iloc[:, -1]
    X_test2 = df3n.iloc[:, :-1]
    y_test2 = df3n.iloc[:, -1]

    plans, recs = shatnawi(X_train1, y_train1, X_test1, y_test1, p=p)
    score = []
    score2 = []
    bugchange = []
    size = []
    matrix = []
    for i in range(0, len(y_test1)):
        for j in range(0, len(y_test2)):
            actual = X_test2.values[j]
            if df3.iloc[j, 0] == df2.iloc[i, 0] and y_test1[i] != 0:
                temp = X_test1.values[i].copy()
                plan = plans.iloc[i, :].values
                rec = recs[i]

                score.append(overlap(plan, actual))
                score2.append(overlap(plan, X_test1.values[i]))
                bugchange.append(bug3[j] - bug2[i])  # negative if reduced #bugs, positive if added
                size.append(size_interval(plan))
                tp, tn, fp, fn = abcd(temp, plan, actual, rec)
                matrix.append([tp, tn, fp, fn])
                break
    print(name[0])
    print('>>>')
    print('>>>')
    print('>>>')
    return score, bugchange, size, score2, matrix


def get_percentiles(df):
    percentile_array = []
    q = dict()
    for i in np.arange(0, 100, 1):
        for col in df.columns:
            try:
                q.update({col: np.percentile(df[col].values, q=i)})
            except:
                pass

        elements = dict()
        for col in df.columns:
            try:
                elements.update({col: df.loc[df[col] >= q[col]].median()[col]})
            except:
                pass

        percentile_array.append(elements)

    return percentile_array


def oliveira(train, test):
    """
    Implements shatnavi's threshold based planner.
    :param train:
    :param test:
    :param rftrain:
    :param tunings:
    :param verbose:
    :return:
    """
    "Helper Functions"

    def compliance_rate(k, train_columns):
        return len([t for t in train_columns if t <= k]) / len(train_columns)

    def penalty_1(p, k, Min, compliance):

        comply = Min - compliance
        if comply >= 0:
            return (Min - compliance) / Min
        else:
            return 0

    def penalty_2(k, Med):
        if k > Med:
            return (k - Med) / Med
        else:
            return 0

    "Compute Thresholds"

    #     if isinstance(test, list):
    #         test = list2dataframe(test)

    #     if isinstance(test, str):
    #         test = list2dataframe([test])

    #     if isinstance(train, list):
    #         train = list2dataframe(train)

    lo, hi = train.min(), train.max()
    quantile_array = get_percentiles(train)
    changes = []

    pk_best = dict()

    for metric in train.columns[:]:
        min_comply = 10e32
        vals = np.empty([10, 100])
        for p_id, p in enumerate(np.arange(0, 100, 10)):
            p = p / 100
            for k_id, k in enumerate(np.linspace(lo[metric], hi[metric], 100)):
                try:
                    med = quantile_array[90][metric]
                    compliance = compliance_rate(k, train[metric])
                    penalty1 = penalty_1(p, k, compliance=compliance, Min=0.9)
                    penalty2 = penalty_2(k, med)
                    comply_rate_penalty = penalty1 + penalty2
                    vals[p_id, k_id] = comply_rate_penalty

                    if (comply_rate_penalty < min_comply) or (
                            comply_rate_penalty == min_comply and p >= pk_best[metric][0] and k <= pk_best[metric][1]):
                        min_comply = comply_rate_penalty
                        try:
                            pk_best[metric] = (p, k)
                        except KeyError:
                            pk_best.update({metric: (p, k)})
                except:
                    pk_best.update({metric: (p, None)})

    """
    Apply Plans Sequentially
    """

    modified = []
    recs = []
    for n in range(test.shape[0]):
        new_row, rec = apply3(test.iloc[n].values.tolist(), test.columns, pk_best)
        modified.append(new_row)
        recs.append(rec)
    return pd.DataFrame(modified, columns=test.columns), recs


def runolive(name):
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
    df2n = norm(df22, df22)
    df3n = norm(df22, df33)

    X_train1 = df1n.iloc[:, :-1]
    y_train1 = df1n.iloc[:, -1]
    X_test1 = df2n.iloc[:, :-1]
    y_test1 = df2n.iloc[:, -1]
    X_test2 = df3n.iloc[:, :-1]
    y_test2 = df3n.iloc[:, -1]

    plans, recs = oliveira(X_train1, X_test1)
    score = []
    bugchange = []
    size = []
    score2 = []
    matrix = []
    for i in range(0, len(y_test1)):
        for j in range(0, len(y_test2)):
            actual = X_test2.values[j]
            if df3.iloc[j, 0] == df2.iloc[i, 0] and y_test1[i] != 0:
                temp = X_test1.values[i].copy()
                plan = plans.iloc[i, :].values
                rec = recs[i]
                #                 print("plan",plan)
                #                 print('actual',actual)
                #                 print('id1',plan[0][0])
                score.append(overlap(plan, actual))
                bugchange.append(bug3[j] - bug2[i])  # negative if reduced #bugs, positive if added
                size.append(size_interval(plan))
                score2.append(overlap(plan, X_test1.values[i]))
                tp, tn, fp, fn = abcd(temp, plan, actual, rec)
                matrix.append([tp, tn, fp, fn])
                break
    print(name[0])
    print('>>>')
    print('>>>')
    print('>>>')
    return score, bugchange, size, score2, matrix


def xtree(name):
    start = time.time()
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
    X_test = pd.concat([X_test1, y_test1], axis=1, ignore_index=True)
    X_test.columns = df1.columns[1:]

    xtree_arplan = XTREE(strategy="closest", alpha=0.95, support_min=int(X_train1.shape[0] / 20))
    xtree_arplan = xtree_arplan.fit(X_train1)
    patched_xtree = xtree_arplan.predict(X_test)
    print("Runtime for Xtree:", time.time() - start)
    XTREE.pretty_print(xtree_arplan)
    overlap_scores = []
    bcs = []
    size = []
    score2 = []
    matrix = []
    for i in range(0, X_test1.shape[0]):
        for j in range(0, X_test2.shape[0]):
            if df3.iloc[j, 0] == df2.iloc[i, 0] and y_test1[i] != 0:
                temp = X_test1.values[i].copy()
                plan = patched_xtree.iloc[i, :-1]
                rec = [0 for n in range(20)]
                for k in range(20):
                    if not isinstance(plan[k], float):
                        if plan[k][0] != plan[k][1]:
                            rec[k] = 1

                actual = X_test2.iloc[j, :]
                overlap_scores.append(overlap1(plan, plan, actual))
                bcs.append(bug3[j] - bug2[i])
                score2.append(overlap( plan, X_test1.values[i]))
                size.append(size_interval(plan))
                tp, tn, fp, fn = abcd(temp, plan, actual, rec)
                matrix.append([tp, tn, fp, fn])
                break
    return overlap_scores, bcs, size, score2, matrix


def historical_logs(name, par, explainer=None, smote=False, small=0.05, act=False):
    start_time = time.time()
    files = [name[0], name[1], name[2]]
    freq = [0] * 20
    deltas = []
    for j in range(0, len(files) - 2):
        df1 = prepareData(files[j])
        df2 = prepareData(files[j + 1])
        for i in range(1, 21):
            col1 = df1.iloc[:, i]
            col2 = df2.iloc[:, i]
            deltas.append(hedge(col1, col2))
    deltas = sorted(range(len(deltas)), key=lambda k: deltas[k], reverse=True)

    actionable = []
    for i in range(0, len(deltas)):
        if i in deltas[0:5]:
            actionable.append(1)
        else:
            actionable.append(0)

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

    old_change = []
    new_change = []
    par = 0
    clf1 = RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_split=10, n_jobs=-1)
    #     clf1 =  MLPClassifier(hidden_layer_sizes=[10],max_iter=2000,early_stopping=False,learning_rate='adaptive')
    if smote:
        sm = SMOTE()
        X_train1_s, y_train1_s = sm.fit_resample(X_train1, y_train1)
        clf1.fit(X_train1_s, y_train1_s)
        explainer = lime.lime_tabular.LimeTabularExplainer(training_data=X_train1_s.values,
                                                           training_labels=y_train1_s.values,
                                                           feature_names=df11.columns,
                                                           discretizer='entropy', feature_selection='lasso_path',
                                                           mode='classification', sample_around_instance=True)
    else:
        clf1.fit(X_train1, y_train1)
        explainer = lime.lime_tabular.LimeTabularExplainer(training_data=X_train1.values, training_labels=y_train1,
                                                           feature_names=df11.columns,
                                                           discretizer='entropy', feature_selection='lasso_path',
                                                           mode='classification', sample_around_instance=True)
    for i in range(0, len(y_test1)):
        for j in range(0, len(y_test2)):
            if df3.iloc[j, 0] == df2.iloc[i, 0] and y_test1[i] != 0:
                actual = X_test2.values[j]
                ins = explainer.explain_instance(data_row=X_test1.values[i], predict_fn=clf1.predict_proba,
                                                 num_features=20,
                                                 num_samples=5000)
                ind = ins.local_exp[1]
                temp = X_test1.values[i].copy()
                if act:
                    tem, plan, rec = flip(temp, ins.as_list(label=1), ind, clf1, df1n.columns, 0, actionable=actionable)
                else:
                    tem, plan, rec = flip(temp, ins.as_list(label=1), ind, clf1, df1n.columns, 0, actionable=None)
                o = track1(plan, temp)
                n = track1(plan, actual)
                old_change.append(o)
                new_change.append(n)


                break
    print("Runtime:", time.time() - start_time)
    print(name[0], par)
    print('>>>')
    print('>>>')
    print('>>>')
    return old_change, new_change
