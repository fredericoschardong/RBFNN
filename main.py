import numpy as np
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler

class RBFClassifier(BaseEstimator):
    def __init__(self, k=2, n_neighbors=2, plot=False, n_selection=2):
        self.k = k
        self.n_neighbors = n_neighbors
        self.plot = plot
        self.n_selection = n_selection
        
    def euclidean_distance(self, x1, x2):
        return np.linalg.norm(x1 - x2)
    
    def rbf_hidden_layer(self, X):
        def activation(x, c, s):
            return np.exp(-self.euclidean_distance(x, c) / 2 * (s ** 2))
            
        return np.array([[activation(x, c, s) for (c, s) in zip(self.cluster_, self.std_list_)] for x in X])
        
    def fit(self, X, y):
        def convert_to_one_hot(y, n_classes):
            arr = np.zeros((y.size, n_classes))
            arr[np.arange(y.size), y.astype(np.uint)] = 1
            return arr
            
        kmeans = KMeans(n_clusters=self.k, random_state=0)
        kmeans_prediction = kmeans.fit_predict(X)
        
        if self.plot:
            plt.scatter(X[:, 0], X[:, 1], c=kmeans_prediction)
            plt.savefig('figs/k-means with k=%d.png' % self.k)
            plt.clf()

        self.cluster_ = kmeans.cluster_centers_
        cond = self.k if self.n_neighbors > self.k or self.n_neighbors == 0 else self.n_neighbors

        # Select N clusters centroids at "random"
        if self.n_selection == 0:
            self.std_list_ = np.array([[self.euclidean_distance(c1, c2) for c1 in self.cluster_] for c2 in self.cluster_[: cond]])
        else:
            self.std_list_ = np.sort(np.array([[self.euclidean_distance(c1, c2) for c1 in self.cluster_] for c2 in self.cluster_]))
            
            # Select N clusters centroids by distance (closest last)
            if self.n_selection == 2:
                self.std_list_ = self.std_list_[::-1]
                
            self.std_list_ = self.std_list_[:, : cond]
            
        self.std_list_ = np.mean(self.std_list_, axis=1)
        
        RBF_X = self.rbf_hidden_layer(X)

        self.w_ = np.linalg.pinv(RBF_X.T @ RBF_X) @ RBF_X.T @ convert_to_one_hot(y, np.unique(y).size)
        
        rbs_prediction = np.array([np.argmax(x) for x in self.rbf_hidden_layer(X) @ self.w_])
        
        if self.plot:
            plt.scatter(X[:, 0], X[:, 1], c=rbs_prediction)
            plt.savefig('figs/rbs train k=%d, n_neighbors=%f.png' % (self.k, self.n_neighbors))
            plt.clf()

    def predict(self, X):
        rbs_prediction = np.array([np.argmax(x) for x in self.rbf_hidden_layer(X) @ self.w_])
        
        if self.plot:
            plt.scatter(X[:, 0], X[:, 1], c=rbs_prediction)
            plt.savefig('figs/rbs predict k=%d, n_neighbors=%f.png' % (self.k, self.n_neighbors))
            plt.clf()
        
        return rbs_prediction
        
    def get_params(self, deep=True):
        return {"k": self.k, "n_neighbors": self.n_neighbors, "plot": self.plot, "n_selection": self.n_selection}

data = np.loadtxt(open("dataset.csv", "rb"), delimiter=",", skiprows=1)

for i in range(2):
    x = data[:, i]
    hist, bins = np.histogram(x)
    plt.plot(bins[:hist.size], hist / np.sum(hist))
    print(i, 'min %.2f max %.2f mean %.2f std %.2f' %(np.min(x), np.max(x), np.mean(x), np.std(x)))

plt.xlabel('Values')
plt.ylabel('Proportions')
plt.savefig('Histogram before normalization.png')
plt.clf()

scaler = MinMaxScaler()
scaler.fit(data[:, 0:2])

X = scaler.transform(data[:, 0:2])

xTrain, xTest, yTrain, yTest = train_test_split(X, data[:, 2], test_size = 0.2, random_state = 0)

for i in range(2):
    x = xTrain[:, i]
    hist, bins = np.histogram(x)
    plt.plot(bins[:hist.size], hist / np.sum(hist))
    print(i, 'min %.2f max %.2f mean %.2f std %.2f' %(np.min(x), np.max(x), np.mean(x), np.std(x)))

plt.xlabel('Values')
plt.ylabel('Proportions')
plt.savefig('Histogram after normalization.png')
plt.clf()

plt.scatter(X[:, 0], X[:, 1], c=data[:, 2])
plt.savefig('correct result.png')
plt.clf()


# See how f1-score goes for k=50, trying different N selection methods
k = 50

for n_selection in range(3):
    results = []

    for n in range(2, k + 1):
        clf = RBFClassifier(k, n, False, n_selection)
        clf.fit(xTrain, yTrain)
        results.append(classification_report(yTest, clf.predict(xTest), output_dict=True)['weighted avg']['f1-score'])
        
    plt.plot(results)
    plt.ylabel('f1-score')
    plt.xlabel('N')
    plt.savefig('f1-score for k = %d, N from 2 to %d selected at %s.png' % (k, k, ('random' if n_selection == 0 else ('sorted' if n_selection == 1 else 'sorted backwards'))))
    plt.clf()
    
# Now that we know the best N selection method, let's take a look at how f1-score goes for different amount of neurons at the hidden layer
results = []

for k in range(2, 51):
    clf = RBFClassifier(k, 2, True)
    clf.fit(xTrain, yTrain)
    results.append(classification_report(yTest, clf.predict(xTest), output_dict=True)['weighted avg']['f1-score'])
    #print(confusion_matrix(yTest, clf.predict(xTest)))
    
plt.plot(results)
plt.ylabel('f1-score')
plt.xlabel('k')
plt.savefig('f1-score for k = 2...50, N = 2 (furthest neighbor).png')
plt.clf()
