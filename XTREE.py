import os
import sys
import numpy as np
import pandas as pd
from pdb import set_trace
from collections import Counter

root = os.path.join(os.getcwd().split('src')[0], 'src')
if root not in sys.path:
    sys.path.append(root)

from frequent_items.item_sets import ItemSetLearner
from tools.Discretize import discretize, fWeight
from sklearn.base import BaseEstimator
from tools.containers import Thing


__author__ = 'Rahul Krishna <i.m.ralk@gmail.com>'
__copyright__ = 'Copyright (c) 2018 Rahul Krishna'
__license__ = 'MIT License'


class XTREE(BaseEstimator):
    def __init__(self, min_levels=1, dependent_var_col_id=-1,
                 prune=False, max_levels=10, info_prune=1, alpha=0.33,
                 bins=3, support_min=90, strategy="itemset"):
        """
        XTREE Planner

        Parameters
        ----------
        min: int (default 1)
            Minimum tree depth
        dependent_var_col_id: int (default -1)
            Column index of the dependent variable
        prune: bool (default False)
            Prune to keep only top freatures
        max_levels: int (default 10)
            Maximum depth of the tree
        info_prune: float (default 1.0)
            Maximum fraction of features to keep. Only used if prune==True.
        alpha: float (default 0.66)
            A destination node is considered "better" it is alpha times
            lower than current node
        bins: int (default 3)
            Number of bins to discretize data into
        support_min: int (default 50)
            Minimum support for mining frequent item sets
        """

        self.min_levels = min_levels
        self.klass = dependent_var_col_id
        self.prune = prune
        self.max_levels = max_levels
        self.info_prune = info_prune
        self.alpha = alpha
        self.bins = bins
        self.support_min = support_min
        self.strategy = strategy

    @staticmethod
    def _entropy(x):
        """
        Compute entropy

        Parameters
        ----------
        x: List[int]
            A list of discrete values

        Returns
        -------
        float:
            Entropy of the elements in a list
        """
        counts = Counter(x)
        N = len(x)
        return sum([-counts[n] / N * np.log(
            counts[n] / N) for n in counts.keys()])

    def pretty_print(self, tree=None, lvl=-1):
        """
        Print tree on console as an ASCII

        Parameters
        ----------
        tree: Thing (default None)
            Tree node
        lvl: int (default -1)
            Tree level

        Note
        ----
        + Thing is a generic container, in this case its a node in the tree.
        + You'll find it in <src.tools.containers>
        """

        if tree is None:
            tree = self.tree
        if tree.f:
            print(('|...' * lvl) + str(tree.f) + "=" + "(%0.2f, %0.2f)" %
                  tree.val + "\t:" + "%0.2f" % (tree.score), end="")
        if tree.kids:
            print("")
            for k in tree.kids:
                self.pretty_print(k, lvl + 1)
        else:
            print("")

    def _nodes(self, tree, lvl=0):
        """
        Enumerates all the nodes in the tree

        Parameters
        ----------
        tree: Thing
            Tree node
        lvl: int (default 0)
            Tree level

        Yields
        ------
        Thing:
            Current child node
        int:
            Level of current child node

        Note
        ----
        + Thing is a generic container, in this case its a node in the tree.
        + You'll find it in <src.tools.containers>
        """

        if tree:
            yield tree, lvl
            for kid in tree.kids:
                lvl1 = lvl
                for sub, lvl1 in self._nodes(kid, lvl1 + 1):
                    yield sub, lvl1

    @staticmethod
    def _path_from_root(node):
        """
        All the attributes in the path from root to node

        Parameters
        ----------
        node : Thing
            The tree node object

        Returns
        -------
        list:
            A list of all the attributes from root to node
        """

        path_names = [keys for keys in map(lambda x: x[0], node.branch)]
        return path_names

    def _leaves(self, thresh=float("inf")):
        """
        Enumerate all leaf nodes

        Parameters
        ----------
        thresh: float (optional)
            When provided. Only leaves with values less than thresh are returned

        Yields
        ------
        Thing:
            Leaf node

        Note
        ----
        + Thing is a generic container, in this case its a node in the tree.
        + You'll find it in <src.tools.containers>
        """

        for node, _ in self._nodes(self.tree):
            if not node.kids and node.score <= thresh:
                yield node

    def _find(self, test_instance, tree_node=None):
        """
        Find the leaf node that a given row falls in.

        Parameters
        ----------
        test_instance: <pandas.frame.Series>
            Test instance

        Returns
        -------
        Thing:
            Node where the test instance falls

        Note
        ----
        + Thing is a generic container, in this case its a node in the tree.
        + You'll find it in <src.tools.containers>
        """

        if len(tree_node.kids) == 0:
            found = tree_node
        else:
            for kid in tree_node.kids:
                found = kid
                if kid.val[0] <= test_instance[kid.f] < kid.val[1]:
                    found = self._find(test_instance, kid)
                elif kid.val[1] == test_instance[kid.f] \
                                == self.tree.t.describe()[kid.f]['max']:
                    found = self._find(test_instance, kid)

        return found

    def _tree_builder(self, dframe, lvl=-1, as_is=float("inf"),
                      parent=None, branch=[], f=None, val=None):
        """
        Construct decision tree

        Parameters
        ----------
        dframe: <pandas.core.Frame.DataFrame>
            Raw data as a dataframe
        lvl: int (default -1)
            Level of the tree
        as_is: float (defaulf "inf")
            Entropy of the class variable in the current rows
        parent: Thing (default None)
            Parent Node
        branch: List[Thing] (default [])
            Parent nodes visitied to reach current node
        f: str (default None)
            Name of the attribute represented by the current node
        val: Tuple(low, high)
            The minimum and maximum range of the attribute in the current node

        Returns
        -------
        Thing:
            The root node of the tree

        Notes
        -----
        + Thing is a generic container, in this case it's a node in the tree.
        + You'll find it in <src.tools.containers>
        """

        current = Thing(t=dframe, kids=[], f=f, val=val,
                        parent=parent, lvl=lvl, branch=branch)

        features = fWeight(dframe)

        if self.prune and lvl < 0:
            features = fWeight(dframe)[:int(len(features) * self.info_prune)]

        name = features.pop(0)
        remaining = dframe[features + [dframe.columns[self.klass]]]
        feature = dframe[name].values
        dependent_var = dframe[dframe.columns[self.klass]].values
        N = len(dependent_var)
        current.score = np.mean(dependent_var)
        splits = discretize(feature, dependent_var)
        low = min(feature)
        high = max(feature)
        cutoffs = [t for t in self.pairs(
            sorted(list(set(splits + [low, high]))))]

        if lvl > (self.max_levels if self.prune else int(
                len(features) * self.info_prune)):
            return current
        if as_is == 0:
            return current
        if len(features) < 1:
            return current

        def _rows():
            for span in cutoffs:
                new = []
                for f, row in zip(feature, remaining.values.tolist()):
                    if span[0] <= f < span[1]:
                        new.append(row)
                    elif f == span[1] == high:
                        new.append(row)
                yield pd.DataFrame(new, columns=remaining.columns), span

        for child, span in _rows():
            n = child.shape[0]
            to_be = self._entropy(child[child.columns[self.klass]])
            if self.min_levels <= n < N:
                current.kids += [
                    self._tree_builder(child, lvl=lvl + 1, as_is=to_be,
                                       parent=current, branch=branch +
                                       [(name, span)],
                                       f=name, val=span)]

        return current

    def fit(self, train_df):
        """
        Fit the current data to generate a decision tree

        Parameter
        ---------
        train_df: <pandas.core.frame.DataFrame>
            Training data

        Return
        ------
        self:
            Pointer to self
        """
        X_train = train_df
        # print(X_train.shape)
        # [train_df.columns[1:]]
        self.tree = self._tree_builder(X_train)
        return self

    @staticmethod
    def pairs(lst):
        """
        Return pairs of values form a list

        Parameters
        ----------
        lst: list
            A list of values

        Yields
        ------
        tuple:
            Pair of values

        Example
        -------

        BEGIN
        ..
        lst = [1,2,3,5]
        ..
        returns -> 1,2
        lst = [2,3,5]
        ..
        returns -> 2,3
        lst = [3,5]
        ..
        returns -> 3,5
        lst = []
        ..
        END
        """
        while len(lst) > 1:
            yield (lst.pop(0), lst[0])

    @staticmethod
    def jaccard_similarity_score(set1, set2):
        """
        Jaccard similarity index
        Parameters
        ----------
        set1: set
            First set
        set2: set
            Second set
        Returns
        -------
        float:
            Jaccards similarity index
        Notes
        -----
        + Jaccard's measure is computed as follows
                                  |A <intersection> B|
            Jaccard Index = --------------------------------
                            |A| + |B| - |A <intersection> B|
        + See https://en.wikipedia.org/wiki/Jaccard_index
        """
        # -- If we have lists, convert them to sets --
        if isinstance(set1, list):
            set1 = set(set1)
        if isinstance(set2, list):
            set2 = set(set2)

        # -- Compute the Jaccard score --
        intersect_length = len(set1.intersection(set2))
        set1_length = len(set1)
        set2_length = len(set2)
        return intersect_length / (set1_length + set2_length - intersect_length)

    def best_plan(self, better_nodes, item_sets):
        """
        Obtain the best plan that has the maximum jaccard index
        with elements in an item set.

        Parameters
        ----------
        better_nodes: List[Thing]
            A list of terminal nodes that are "better" than the node
            which the current test instance lands on.
        item_set: List[set]
            A list containing all the frequent itemsets.

        Returns
        -------
        Thing:
            Best leaf node

        Note
        ----
        + Thing is a generic container, in this case its a node in the tree.
        + You'll find it in <src.tools.containers>
        """
        max_intersection = float("-inf")

        # Sort better nodes by score
        better_nodes.sort(key=lambda X: X.score)

        # Initialize the best path
        best_path = better_nodes[0]

        # Try and find a better path, with a higher overlap with item sets
        for node in better_nodes:
            change_set = set([bb[0] for bb in node.branch])
            for item_set in item_sets:
                jaccard_index = self.jaccard_similarity_score(
                    item_set, change_set)
                if 0 < jaccard_index >= max_intersection:  # TODO: Check
                    best_path = node
                    max_intersection = jaccard_index

        return best_path

    def best_plan_closest(self, better_nodes, current_node):
        """
        Obtain the best plan by picking a node from better nodes that is 
        closest to the current_node

        Parameters
        ----------
        better_nodes: List[Thing]
            A list of terminal nodes that are "better" than the node
            which the current test instance lands on.
        current_node : [type]
            The node where the current test case falls into

        Returns
        -------
        Thing:
            Best leaf node

        Note
        ----
        + Thing is a generic container, in this case its a node in the tree.
        + You'll find it in <src.tools.containers>
        """
        current_path_components = self._path_from_root(current_node)
        min_dist = float("inf")
        best_path = current_node
        for other_path in better_nodes:
            other_path_components = self._path_from_root(other_path)
            jaccard_index = self.jaccard_similarity_score(
                current_path_components, other_path_components)
            if jaccard_index <= min_dist:
                min_dist = jaccard_index
                best_path = other_path
        return best_path

    def predict(self, X_test):
        """
        Recommend plans for a test data

        Parameters
        ----------
        test_df: <pandas.core.frame.DataFrame>
            Testing data

        Returns
        -------
        <pandas.core.frame.DataFrame>:
            Recommended changes
        """

        new = []
        # y = y_test
        # X = X_test
        y = X_test[X_test.columns[-1]]
        X = X_test[X_test.columns[1:-1]]

        # ----- Itemset Learning -----
        if self.strategy == "itemset":
            # -- Instantiate item set learning --
            isl = ItemSetLearner(bins=self.bins, support_min=self.support_min)
            # -- Fit the data to itemset learner --
            isl.fit(X, y)
            # -- Transform into itemsets --
            item_sets = isl.transform()

        # ----- Obtain changes -----
        for row_num in range(len(X_test)):
            if X_test.iloc[row_num]["bug"] > 0:
                print('counter',row_num)
                cur = X_test.iloc[row_num]
                # Find the location of the current test instance on the tree
                pos = self._find(cur, tree_node=self.tree)
                print(pos.score)
                # Find all the leaf nodes on the tree that atleast alpha
                # times smaller that current test instance
                better_nodes = [leaf for leaf in self._leaves(
                    thresh=self.alpha * pos.score)]
                # TODO: Check this
                if better_nodes:
                    # - Find the path with the highest overlap with itemsets -
                    # -- Choose the startegy based on how we want to do it --
                    # ---- Use item sets ----
                    if self.strategy == "itemset":
                        best_path = self.best_plan(better_nodes, item_sets)
                    # ---- Find the closest ----
                    elif self.strategy == "closest":
                        best_path = self.best_plan_closest(better_nodes, pos)
                    else:
                        raise ValueError(
                            "Invalid argument for. Use either \"itemset\" or \"closest\" ")
                    result = [[0 for m in range(2)] for n in range(len(cur.values))]
                    for i in range(0,len(cur.values)):
                        result[i][0],result[i][1] = cur.values[i],cur.values[i]
                    for entities in best_path.branch:
                        loc = X_test.columns.get_loc(entities[0])
                        result[loc][0], result[loc][1]= entities[1][0],entities[1][1]
                    new.append(result)
                else:
                    new.append(X_test.iloc[row_num].values.tolist())
            else:
                new.append(X_test.iloc[row_num].values.tolist())

        new = pd.DataFrame(new, columns=X_test.columns)
        return new
