import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

import time

sns.set_context('talk')

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1, random_state=42),
    GaussianProcessClassifier(max_iter_predict=1000, random_state=42),
    DecisionTreeClassifier(max_depth=5, random_state=42),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, random_state=42),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

data = pd.read_csv('data/metavalent_1_4.csv')
cols = data.columns

# X = data[cols[2:-1]]
X = data[['Globularity', 'dnorm', 'EN_diff']]
X.index = data.index
y = data[cols[-1]]

std = StandardScaler()
X_scaled = std.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
X_scaled.index = data.index

clf_scores = []
clf_train_scores = []
times = []
avg_scores = []

ss = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
# ss = StratifiedKFold(n_splits=5, random_state=0)
for name, clf in zip(names, classifiers):
    train_score = []
    score = []

    for train_idx, test_idx in ss.split(X_scaled, y):
        X_train, X_test = X_scaled.iloc[train_idx], X_scaled.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # svc = SVC(kernel='rbf')
        svc = clf
        svc.fit(X_train, y_train)
        train_score.append(svc.score(X_train, y_train))
        score.append(svc.score(X_test, y_test))

    avg_scores.append(sum(score) / 5)
    ind = score.index(max(score))
    train_idx, test_idx = list(ss.split(X_scaled, y))[ind]
    X_train, X_test = X_scaled.iloc[train_idx], X_scaled.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    svc = clf
    t = time.time()
    svc.fit(X_train, y_train)
    times.append(time.time() - t)
    clf_scores.append(svc.score(X_test, y_test))
    clf_train_scores.append(svc.score(X_train, y_train))

scores = pd.DataFrame({'Names': names, 'Classification Scores': clf_scores,
                       'Time': times, 'Average Score': avg_scores, 'Training Scores': clf_train_scores})
scores = scores.sort_values(by='Classification Scores', ascending=False)
sns.barplot(x='Classification Scores', y='Names', data=scores, palette='flare')
plt.savefig('figs/Comparison', bbox_inches='tight', dpi=300)

predict_data = pd.read_csv('data/MVB_data_1500.csv')
X_pred = predict_data[['Globularity', 'dnorm', 'EN_diff']]
X_pred.index = predict_data['Name']

X_scaled_pred = std.transform(X_pred)
predict_data['Estimate'] = svc.predict(X_scaled_pred)
predict_data.to_csv('outputs/MVB_data_1500_predict.csv')

