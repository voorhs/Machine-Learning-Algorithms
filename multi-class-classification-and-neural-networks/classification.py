import numpy as np
from numpy.lib.function_base import gradient
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from scipy.special import expit

class LogisticRegression:
    def __init__(self, n_classes=2, regul=0, f_mapping=False, max_exp=None):
        self.n_classes = n_classes
        self.regul = regul
        self.f_mapping = f_mapping
        self.max_exp = max_exp

    def fit(self, X, y, learning_rate=0.01, epochs=300):
        self.n_samples = X.shape[0]
        X = np.array(X)

        if self.f_mapping:
            X = feature_mapping(X, self.max_exp)            
        else:
            X = np.vstack([np.ones(X.shape[0]), X.T]).T
        
        self.n_features = X.shape[1]
        
        encoder = OneHotEncoder(sparse=False)
        y = encoder.fit_transform(y.reshape((-1, 1)))            

        self.theta = self.gd(X, y, learning_rate, epochs)
        return self
    
    def predict(self, data):
        X = np.array(data)

        if self.f_mapping:
            X = feature_mapping(X, self.max_exp)
        else:
            X = np.vstack([np.ones(X.shape[0]), X.T]).T

        tags = np.argmax(self.hypothesis(X, self.theta), axis=1)
        probas = np.max(self.hypothesis(X, self.theta), axis=1)
        result = np.vstack([tags, probas])
        return result

    def hypothesis(self, X, theta):
        return expit(np.dot(X, theta))
    
    def gradient(self, X, y, theta):
        mask = np.ones_like(theta)
        mask[0] = 0
        y_hat = self.hypothesis(X, theta)
        return np.dot(X.T, y_hat - y) / X.shape[0] + self.regul / X.shape[0] * theta * mask

    def cost_function(self, X, y, theta):
        m = X.shape[0]
        predicted = self.hypothesis(X, theta)
        y_hat = predicted - np.sign(predicted - 0.5) * 0.00001
        total_cost = -(1 / m) * np.sum(
            y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)
            ) + self.regul * np.sum(theta ** 2) / (2 * m)
        return total_cost

    def score(self, X, y):
        y_hat = self.predict(X)[0].astype(int)
        return np.mean(y == y_hat)
    
    def __score__(self, X, y, theta):
        return self.cost_function(X, y, theta)
    
    def gd(self, X, y, learning_rate, epochs):
        theta = np.zeros((self.n_features, self.n_classes))
        cost_history = np.zeros((self.n_classes, epochs))
        
        for k in range(self.n_classes):
            for i in range(epochs):
                cost_history[k, i] = self.__score__(X, y[:, k], theta[:, k])
                delta = learning_rate * self.gradient(X, y[:, k], theta[:, k])
                theta[:, k] -= delta
        
        self.cost_history = cost_history

        return theta
    
def feature_mapping(X, max_exp):
    X = np.array(X)
    n_features = (max_exp + 2) * (max_exp + 1) // 2
    X_new = np.zeros((n_features, X.shape[0]))
    k = 0
    for i in range(max_exp + 1):
        for j in range(max_exp + 1 - i):
            X_new[k] = X[:, 0] ** i * X[:, 1] ** j
            k += 1
    return X_new.T

def contour_plot(model, max_exp, data, s=[30, 30]):
    r = np.zeros((2, 2))
    r[0, 0] = data[:, 0].min()
    r[0, 1] = data[:, 0].max()
    r[1, 0] = data[:, 1].min()
    r[1, 1] = data[:, 1].max()
    t1 = np.linspace(r[0, 0], r[0, 1], s[0])
    t2 = np.linspace(r[1, 0], r[1, 1], s[1])
    xv, yv = np.meshgrid(t1, t2)
    z = np.zeros(s)

    for i in range(s[0]):
        for j in range(s[1]):
            feats = np.array([xv[i, j], yv[i, j]]).reshape(1, 2)
            z[i, j] = model.predict(feature_mapping(feats, max_exp))

    plt.contourf(z)
    plt.xlabel('x1')
    plt.ylabel('x2')
        

def cv_score(X, y, n_splits=10, n_classes=None, regul=0, f_mapping=False, max_exp=None):
    kf = KFold(n_splits=n_splits)
    scores = []
    for ti, vi in kf.split(X, y):    
        X_train = X[ti]
        X_test = X[vi]
        y_train = y[ti]
        y_test = y[vi]
    
        m = LogisticRegression(n_classes, regul, f_mapping, max_exp)
        m.fit(X_train, y_train)
        scores.append(m.score(X_test, y_test))
    
    return np.array(scores)