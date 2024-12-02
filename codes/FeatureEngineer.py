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


class ClassificationFeatureEngineer:
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def logistic_selection(self, start=-3, end=3, tops=5, viz=False):
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
            model = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=3), n_estimators=50, algorithm='SAMME')
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

        return selected_features

    def syn_feature_ranking(self, tops=5):
        models = ['LR', 'LSVM', 'RF', 'ADA']
        feature_names = self.features.columns

        # Enumerate on all models
        temps = []
        candidates = set()
        for model in models:
            temp_features = self.gen_feature_selection(model=model, tops=tops, cv=0)
            candidates.update(temp_features)
            temps.append(temp_features)

        ranking_dict = {candidate: 0 for candidate in candidates}
        for temp in temps:
            losers = candidates - set(temp)

            for rank in range(tops):
                ranking_dict[temp[rank]] += rank
            for loser in losers:
                ranking_dict[loser] += tops

        # Ranking the features
        selected_features_sr = pd.Series(list(ranking_dict.values()),
                                         index=list(ranking_dict.keys()),
                                         name='feature_ranking')
        selected_features_sr = selected_features_sr.sort_values()

        # Regularize the Rankings
        for idx, rank in enumerate(selected_features_sr.values):
            if selected_features_sr.iloc[idx] == (selected_features_sr.iloc[idx - 1] if idx >= 1 else 0):
                continue
            else:
                selected_features_sr.iloc[idx] = idx

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

        return selected_features_sr

    def PCA_feature_selection(self, tops=5, viz=True):
        scaler = StandardScaler()
        scaler.fit(self.features)
        scaled_features = scaler.transform(self.features)

        pca = PCA(n_components=tops)
        pca.fit(scaled_features)

        feature_names = self.features.columns

        if viz:
            plt.figure(figsize=(16, 9))
            plt.matshow(pca.components_, cmap='hot')
            plt.yticks([rank for rank in range(tops)], [f'Top {rank + 1}th component' for rank in range(tops)])
            plt.colorbar()
            plt.xticks([fea for fea in range(len(feature_names))], feature_names, rotation=60, ha='left')

            plt.title('PCA Feature Selection')
            plt.xlabel("Component Weights")
            plt.ylabel("Principle Component")
            plt.show()
            plt.close()

        return pd.DataFrame(pca.components_.T, index=feature_names, columns=[f"Top{rank + 1}" for rank in range(tops)])


