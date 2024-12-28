import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_selection import RFECV, RFE

from scipy.stats import spearmanr


class ClassificationFeatureEngineer:
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def logistic_feature_selection(self, start=-3, end=3, tops=5, viz=False):
        regularizers = np.logspace(start=end, stop=start, num=100, base=10)
        tracker = pd.DataFrame()

        for regularizer in regularizers:
            model = LogisticRegression(C=regularizer, penalty='l1', solver='saga')
            model.fit(self.features, self.labels)
            coefficient = model.coef_

            if coefficient.shape[0] == 1:
                coefficient = coefficient.flatten()
                coefficient = pd.DataFrame(coefficient.reshape(1, -1), columns=self.features.columns)
                if tracker.empty:
                    tracker = coefficient
                else:
                    tracker = pd.concat([tracker, coefficient], ignore_index=True, axis=0)
            else:
                coefficient = coefficient.sum(axis=0)
                coefficient = coefficient.flatten()
                coefficient = pd.DataFrame(coefficient.reshape(1, -1), columns=self.features.columns)
                if tracker.empty:
                    tracker = coefficient
                else:
                    tracker = pd.concat([tracker, coefficient], ignore_index=True, axis=0)

        counter = tracker.copy()
        counter[counter != 0] = 1
        counter = counter.sum(axis=1)

        top_features = []
        for idx in range(len(counter) - 1, -1, -1):
            if counter.iloc[idx] >= tops:
                top_features = tracker.iloc[idx, :]
                top_features = top_features.loc[top_features != 0]
                top_features = top_features.sort_values(ascending=False, key=lambda x: abs(x))
                break
            else:
                continue

        # Printing the top features
        print(' [Logistic Regression] Feature Selection '.center(54, '='))
        for rank in range(0, len(top_features)):
            if rank + 1 > tops:
                break
            else:
                print(f'{rank + 1}. {str(top_features.index[rank])}    weight: {top_features.iloc[rank]:>6.3f}')

        # Visualize the top features
        if viz:
            plt.figure(figsize=(16, 9))

            for feature in range(0, tracker.shape[1]):
                if tracker.columns[feature] in top_features.index:
                    tracker.iloc[:, feature].plot(
                        kind='line',
                        label=str(tracker.columns[feature])
                    )
                else:
                    tracker.iloc[:, feature].plot(
                        kind='line',
                        alpha=0.5,
                        lw=0.8,
                        color='gray',
                        label='_nolegend_'
                    )

            plt.axhline(0, color='black', linestyle='--', linewidth=0.8)

            plt.legend(loc='upper right')
            plt.title("Logistic Regression Feature Selection")
            plt.xlabel("Regularizer's Significance")
            plt.ylabel("Features' Significance")
            plt.show()
            plt.close()

    def gen_feature_selection(self, model='LR', tops=5, cv=10):
        model_parallel = 1 if cv > 1 else -1

        if model == 'LR':
            title = 'Logistic Regression'
            model = LogisticRegression(n_jobs=model_parallel)
        elif model == 'ADA':
            title = 'Adaboost Classifier'
            model = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=5), n_estimators=100, algorithm='SAMME')
        elif model == 'LSVM':
            title = 'LinearSVM Classifier'
            model = LinearSVC(penalty='l1')
        elif model == 'RF':
            title = 'Random Forest Classifier'
            model = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=model_parallel)
        else:
            title = 'Logistic Regression'
            model = LogisticRegression(n_jobs=model_parallel)

        if cv > 1:
            rfe_machine = RFECV(estimator=model, cv=cv, scoring='accuracy', n_jobs=-1)
            rfe_machine.fit(self.features, self.labels)
        else:
            rfe_machine = RFE(estimator=model, n_features_to_select=tops)
            rfe_machine.fit(self.features, self.labels)

        ranking = rfe_machine.ranking_
        selected_features = ranking.argsort()[: tops]
        feature_names = self.features.columns

        print(f' [{title} RFECV] Feature Selection '.center(54, '='))
        for rank, index in enumerate(selected_features):
            print(f'{rank + 1}. {str(feature_names[index])}')

        return pd.Series(selected_features, name=f'{title}_ranking')

    def syn_feature_selection(self, tops=5):
        models = ['LR', 'LSVM', 'RF', 'ADA']
        feature_names = self.features.columns.to_list()

        # Enumerate on all models
        temps = []
        candidates = set()
        for model in models:
            temp_features = self.gen_feature_selection(model=model, tops=tops, cv=0).to_list()
            candidates.update(temp_features)
            temps.append(temp_features)

        ranking_dict = {candidate: 0 for candidate in candidates}
        individual_df = pd.DataFrame(None, columns=models, index=list(candidates))

        for idx, temp in enumerate(temps):
            losers = candidates - set(temp)

            for rank in range(tops):
                ranking_dict[temp[rank]] += rank
                individual_df.loc[temp[rank], models[idx]] = rank
            for loser in losers:
                ranking_dict[loser] += tops
                individual_df.loc[loser, models[idx]] = tops

        # Ranking the features
        selected_features_sr = pd.Series(list(ranking_dict.values()),
                                         index=list(ranking_dict.keys()),
                                         name='synthesized_ranking')
        selected_features_sr = selected_features_sr.sort_values()

        # Regularize the Rankings
        temp_sr = selected_features_sr.copy()
        for idx, _ in enumerate(selected_features_sr.values):
            if selected_features_sr.iloc[idx] == selected_features_sr.iloc[idx - 1] and idx >= 1:
                temp_sr.iloc[idx] = temp_sr.iloc[idx - 1]
            else:
                temp_sr.iloc[idx] = idx
        selected_features_sr = temp_sr.copy()

        # Convert to a ranking dictionary
        selected_features_dict = {}
        for feature_idx, rank in zip(selected_features_sr.index, selected_features_sr.values):
            if rank + 1 > tops:
                break

            selected_features_dict[rank] = selected_features_dict.get(rank, []) + [feature_names[feature_idx]]

        # Print out the rankings
        print(f' [Synthesized RFECV] Feature Selection '.center(54, '='))
        for rank, names in selected_features_dict.items():
            print(f'{rank + 1}. ', end='')
            for name in names:
                print(f'{str(name)} ', end='')
            print()

        return selected_features_sr, individual_df

    def PCA_feature_selection(self, tops:int = 5, viz:bool = True) -> tuple[pd.DataFrame, pd.Series]:
        '''
        Perform PCA with visualization.

        :param tops: How many top features to look into for specific weights
        :param viz: To visualize the results or not
        :return: Dataframe(specific weights for top features) | Series(variances of all PCs)
        '''
        feature_names = self.features.columns
        centered_features = self.features - self.features.mean(axis=0)
        feature_num = len(self.features.columns)

        # Get the pca with assigned tops & Get all pca with var
        target_pca = PCA(n_components=tops)
        target_pca.fit(centered_features)

        var_pca = PCA(n_components=feature_num)
        var_pca.fit(centered_features)
        variance_sr = pd.Series(var_pca.explained_variance_ratio_, name='pca_vars', index=[f'PC{i}' for i in range(1, feature_num + 1)])
        variance_diff = np.abs(np.diff(variance_sr.to_list())).tolist()
        variance_diff.insert(0, variance_sr.iloc[0])

        # Calculate the Meaningful cutoffs among all PCs
        cutoff_index = feature_num
        for idx, diff in enumerate(variance_diff):
            if diff <= 0.01:
                cutoff_index = idx
                break

        weight_df = pd.DataFrame(target_pca.components_.T, index=feature_names, columns=[f"Top{rank + 1}" for rank in range(tops)])
        PC_idx = [f'IPC{i}' for i in range(1, cutoff_index + 1)] + [f'PC{i}' for i in range(cutoff_index + 1, feature_num + 1)]
        variance_sr.index = PC_idx

        # Plotting
        if viz:
            # Global Plotting Configuration
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9))

            # Specific Importance for Given Tops
            ax1.matshow(target_pca.components_, cmap='hot')
            ax1.set_yticks([rank for rank in range(tops)])
            ax1.set_yticklabels([f'PC{rank + 1}' for rank in range(tops)])
            ax1.set_xticks(np.arange(len(feature_names)))
            ax1.set_xticklabels(list(feature_names), rotation=60, ha='left')
            ax1.set_title('PCA Feature Selection')
            ax1.set_xlabel("Component Weights")
            ax1.set_ylabel("Principle Component")

            # Plot all Variance ratios of all PCs and the Differences Trends
            ax2.bar(np.arange(1, cutoff_index + 1), variance_sr.to_list()[: cutoff_index], color='blue', alpha=0.9)
            ax2.bar(np.arange(cutoff_index + 1, feature_num + 1), variance_sr.to_list()[cutoff_index:], color='gray', alpha=0.5)
            ax2_twin = ax2.twinx()
            ax2_twin.plot(np.arange(1, feature_num + 1), variance_diff, color='red', marker='o', linestyle='-', alpha=0.9)
            ax2.set_xticks(np.arange(1, len(feature_names) + 1))
            ax2.set_xticklabels(variance_sr.index, rotation=60, ha='left')
            ax2_twin.grid()
            ax2_twin.set_ylabel('Variance Diff')
            ax2.set_title('Variance of PCs')
            ax2.set_xlabel('PC Indexes')
            ax2.set_ylabel('Variance(%)')

            # Visualize the Plot
            plt.show()
            plt.close()

        return weight_df, variance_sr

    @staticmethod
    def scp_feature_restriction(rankings, method='rankIC', agg='mean', viz=False):
        # choosing the aggregation of restriction
        def _agg_wrap(ori_list, method=agg):
            evaluation_collector = np.array(ori_list)
            if method == 'mean':
                indicator = evaluation_collector.mean()
            elif method == 'median':
                indicator = evaluation_collector.median()
            elif method == 'sum':
                indicator = evaluation_collector.sum()
            elif method == 'max':
                indicator = evaluation_collector.max()
            elif method == 'min':
                indicator = evaluation_collector.min()
            else:
                indicator = evaluation_collector.mean()

            return indicator

        # calculate the rankIC of all pairs of the ranking dataframe
        def rankIC_pairwise(ranks:pd.DataFrame):
            rankIC_collector = []

            for outer_col in range(len(ranks.columns)):
                for inner_col in range(outer_col + 1, len(ranks.columns)):
                    rankIC_collector.append(spearmanr(ranks.iloc[:, outer_col].to_list(), ranks.iloc[:, inner_col].to_list()))

            return _agg_wrap(rankIC_collector, method=agg)

        # calculate the set difference of all pairs of the ranking dataframe
        def setdiff_pairwise(ranks:pd.Series):
            setdiff_collector = []

            for outer_row in range(len(ranks.index)):
                for inner_row in range(outer_col + 1, len(ranks.index)):
                    setdiff_collector.append(len(set(ranks.iloc[:, outer_row].to_list()) - set(ranks.iloc[:, inner_row].to_list())))

            return _agg_wrap(setdiff_collector, method=agg)

        if isinstance(rankings, list):
            rankings_collector = []

            for ranking in rankings:
                # choosing the method of restriction
                if method.lower() == 'rankic':
                    indicator = rankIC_pairwise(ranking)
                elif method.lower() == 'setdiff':
                    indicator = setdiff_pairwise(pd.Series(set_collector))
                else:
                    indicator = setdiff_pairwise(ranking)

                rankings_collector.append(indicator)

            if viz:
                plt.figure(figsize=(16, 9))
                plt.title("RankIC vs. Feature Size")
                plt.xlabel("Feature Size")
                plt.ylabel("RankIC")

                plt.plot(np.arange(0, len(rankings_collector)), rankings_collector)
                margin = 0.1 * (max(rankings_collector) - min(rankings_collector))
                plt.fill_between(np.arange(0, len(rankings_collector)),
                                 [min(rankings_collector) - margin] * len(rankings_collector),
                                 rankings_collector,
                                 alpha=0.2,
                                 color='blue')
                plt.show()

            return rankings_collector

        else:
            res = rankIC_single(rankings)
            print(f'The rankIC for this ranking dataframe is: {res: .3f}')
            return res









