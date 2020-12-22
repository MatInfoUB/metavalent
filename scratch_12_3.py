import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE
from statsmodels.api import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context('talk')

data = pd.read_csv('data/metavalent.csv')
cols = data.columns

X = data[cols[2:]]
y = data[cols[1]]

X_scaled = StandardScaler().fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

n_components = 15
pca = PCA(n_components=n_components)
pca.fit(X_scaled)
X_pca_scaled = pca.transform(X_scaled)
X_pca_scaled = pd.DataFrame(data=X_pca_scaled,
                            columns=['PC '+str(i+1) for i in range(n_components)],
                            index=data.Compounds)

ss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
train_score = []
score = []

for train_idx, test_idx in ss.split(X_pca_scaled, y):
    X_train, X_test = X_pca_scaled.iloc[train_idx], X_pca_scaled.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    svc = SVC(kernel='rbf')
    svc.fit(X_train, y_train)
    train_score.append(svc.score(X_train, y_train))
    score.append(svc.score(X_test, y_test))

# print('Classification accuracy: ', svc.score(X_test, y_test))

