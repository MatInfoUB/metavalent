import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.svm import SVC

sns.set_context('talk')

data = pd.read_csv('data/metavalent_12_16.csv')
cols = ['CVD_Sum', 'SI_Sum', 'Globularity_avg', 'dnorm_mean_avg',
        'SI_mean_avg', 'CVD_mean_avg', 'Area_avg', 'Volume_avg']

X = data[cols]
X.index = data.Compounds
y = data['Category']

X_scaled = StandardScaler().fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

bx = sns.boxplot(data=X_scaled)
bx.set_xticklabels(bx.get_xticklabels(), rotation=90)
plt.savefig('figs/boxplot_scaled', bbox_inches='tight', dpi=300)
plt.close()

n_components = 5
pca = PCA(n_components=n_components)
pca.fit(X_scaled)

ss = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
train_score = []
score = []


for train_idx, test_idx in ss.split(X_scaled, y):
    X_train, X_test = X_scaled.iloc[train_idx], X_scaled.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    svc = SVC(kernel='poly')
    svc.fit(X_train, y_train)
    train_score.append(svc.score(X_train, y_train))
    score.append(svc.score(X_test, y_test))

ind = score.index(max(score))
train_idx, test_idx = list(ss.split(X_scaled, y))[ind]
X_train, X_test = X_scaled.iloc[train_idx], X_scaled.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

svc = SVC(kernel='rbf')
svc.fit(X_train, y_train)
y_est = svc.predict(X_test)

cm = pd.crosstab(y_test, y_est)
print(cm)