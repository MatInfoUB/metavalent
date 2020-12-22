import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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

# Violin plot to understand the mean, median, mode and skewness
v = sns.violinplot(data=X)
v.set_xticklabels(v.get_xticklabels(), rotation=90)
plt.savefig('figs/violinplot', bbox_inches='tight', dpi=300)
plt.close()

X_scaled = StandardScaler().fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

v = sns.violinplot(data=X_scaled)
v.set_xticklabels(v.get_xticklabels(), rotation=90)
plt.savefig('figs/violinplot_scaled', bbox_inches='tight', dpi=300)
plt.close()

bx = sns.boxplot(data=X_scaled)
bx.set_xticklabels(bx.get_xticklabels(), rotation=90)
plt.savefig('figs/boxplot_scaled', bbox_inches='tight', dpi=300)
plt.close()

# Computing Variance Inflation Factor (VIF)
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                          for i in range(len(X.columns))]
vif_data["VIF_scaled"] = [variance_inflation_factor(X_scaled.values, i)
                          for i in range(len(X_scaled.columns))]
b = sns.barplot(x='feature', y='VIF_scaled', data=vif_data, color='blue')
b.set_xticklabels(b.get_xticklabels(), rotation=90)
b.set_yscale('log')
plt.savefig('figs/vifplot', bbox_inches='tight', dpi=300)
plt.close()

n_components = 15
pca = PCA(n_components=n_components)
pca.fit(X_scaled)
X_pca_scaled = pca.transform(X_scaled)
X_pca_scaled = pd.DataFrame(data=X_pca_scaled,
                            columns=['PC '+str(i+1) for i in range(n_components)],
                            index=data.Compounds)
loadings = pd.DataFrame(pca.components_.T, columns=X_pca_scaled.columns,
                        index=X_scaled.columns)
hm = sns.heatmap(data=loadings.abs(), annot=True, annot_kws={"size": 7}, yticklabels=True)
b, t = plt.ylim()
b += 1.
t -= 1.
plt.ylim(b, t)
hm.set_yticklabels(hm.get_ymajorticklabels(), fontsize = 6)
plt.savefig('figs/absolute_loadings', bbox_inches='tight', dpi=300)

bx = sns.barplot(x=X_pca_scaled.columns, y=pca.explained_variance_ratio_.cumsum(), color='blue')
bx.set_xticklabels(bx.get_xticklabels(), rotation=90)
plt.savefig('figs/scree_plot', bbox_inches='tight', dpi=300)


# tsne = TSNE(n_components=3)
# tsne.fit(X_scaled)

from sklearn.cluster import SpectralClustering
sc = SpectralClustering(3, n_init=100)
sc.fit(X_pca_scaled)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
svc = SVC(kernel='rbf')

