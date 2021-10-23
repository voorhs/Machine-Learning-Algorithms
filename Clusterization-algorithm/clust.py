import numpy as np
from time import time
from time import localtime
from numpy.lib.npyio import save
from sklearn.datasets import make_blobs
from os import mkdir
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, X, n_clusters, n_features=2):
        """
        X is array of points with dimension `n_features`
        """

        self.X = X
        self.n_clusters = n_clusters
        self.n_features = n_features
        self.n_points = X.shape[0]
    
    def init_random(self):
        """
        initializes array of centroids using 'random' strategy:
        random points from the area of X
        """

        min_coords = np.min(self.X, axis=0)
        max_coords = np.max(self.X, axis=0)

        coords = np.zeros((self.n_features, self.n_clusters))
        for i in range(self.n_features):
            coords[i] = np.random.rand(self.n_clusters) * (max_coords[i] - min_coords[i]) + min_coords[i]

        self.centroids = coords.T

    def init_sample(self):
        """
        initializes array of centroids using 'sample' strategy:
        random simple sample from X
        """

        self.indeces = np.random.choice(np.arange(self.X.shape[0]), self.n_clusters, replace=False)
        self.centroids = np.copy(self.X[self.indeces])
    
    def compute_distance_matrix(self, centr_ind):   
        """
        высчитывает и возвращает матрицу расстояний от всех точек до центроидов,
        построенных `centr_int - 1` центроидов.
        нужно для реализации `distant`-стратегии инициализации центроидов
        """     
        res = np.zeros((centr_ind, self.n_points))
        
        for i in range(centr_ind):
            res[i] = np.linalg.norm(self.X - self.centroids[i], axis=-1)

        res = res.T
        if (hasattr(self, 'indeces')):
            res[self.indeces[:centr_ind]] = np.nan

        return res

    def init_distant(self):
        """
        initializes array of centroids using 'distant' strategy:
        1st centroid is random point of X, 2nd is the farthest point of X from 1st centroid,
        3rd is the farthest point of X from 1st ans 2nd centroids etc.
        """
        self.indeces = np.zeros(self.n_clusters, dtype=int)
        self.centroids = np.zeros((self.n_clusters, self.n_features))

        t = np.random.randint(0, self.n_points, size=1)
        self.indeces[0] = t
        self.centroids[0] = self.X[t]

        for i in range(1, self.n_clusters):
            t = np.nanargmax(np.nanmean(self.compute_distance_matrix(i), axis=-1))
            self.indeces[i] = t
            self.centroids[i] = self.X[t]

    def init_centr(self, heur):
        """
        initializes centroids by given heuristic
        """
        dct = {
            "sample": self.init_sample,
            "random": self.init_random,
            "distant": self.init_distant,
        }
        dct[heur]()
    
    def near_center(self, point):
        """
        returns the index of the nearest centroid from given point
        """
        return np.argmin(np.linalg.norm(self.centroids - point, axis=-1))

    def what_cluster(self, point):
        """
        returns the coords of nearect centroid
        """
        return self.centroids[self.near_center(point)]

    def set_labels(self):
        """
        marks points from X in accordance with which centroid is closer
        """
        self.labels = np.zeros(self.n_points)
        for i in range(self.n_points):
            self.labels[i] = self.near_center(self.X[i])  

    def update_centers(self):
        """
        updates centroids for every cluster
        """
        self.centroids = np.zeros((self.n_clusters, self.n_features))
        for i in range(self.n_clusters):
            self.centroids[i] = self.X[self.labels == i].mean(axis=0)
        return self.centroids
    
    def perform(self, heur, prec):
        """
        полный алгоритм: инициализация, установка тегов, итерации.
        """
        self.init_centr(heur)
        self.set_labels()

        while(np.linalg.norm(self.centroids - self.update_centers()) > prec):
            self.set_labels()
        
        self.set_labels()
        self.compute_SSW()
        self.save_config()  # чтобы потом сравнить с другими конфигурациями

        return self
    
    def save_config(self):
        """
        saves centroids, labels, X, total SSW and SSW vector to 'configs' attribute
        """
        self.configs.append([self.centroids, self.labels, self.X, self.SSW, self.within_cluster_squares])
    
    def sort_points(self):
        """
        sorts points of X by cluster labels, distributes points to clusters,
        computes within cluster distances
        """
        together = np.hstack([self.labels.T.reshape((self.n_points, 1)), self.X])
        together = sorted(together, key=lambda x: x[0])
        
        self.X = np.array(together)[:, 1:]
        self.labels = np.array(together)[:, 0]

        self.clusters = []      # list of clusters
        self.distances = []     # list of within cluster distances
        for i in range(self.n_clusters):
            self.clusters.append(self.X[self.labels == i])
            self.distances.append(np.linalg.norm(self.clusters[i] - self.centroids[i], axis=1))
    
    def compute_SSW(self):
        """
        computes total SSW for current cluster configuration
        """
        self.within_cluster_squares = np.zeros(self.n_clusters)
        for i in range(self.n_clusters):
                self.within_cluster_squares[i] = np.sum(np.linalg.norm(self.X[self.labels == i] - self.centroids[i], axis=-1)**2)

        self.SSW = np.sum(self.within_cluster_squares)
    
    def find_best(self, n_start):
        """
        finds the best cluster configuration by total SSW
        """
        SSW_values = np.zeros(n_start)
        for i in range(n_start):
            SSW_values[i] = self.configs[i][3]
        
        return self.configs[np.argmin(SSW_values)]

    def fit(self, heur="sample", n_start=3, prec=0.001):
        """
        запускает `n_start` конфигураций алгоритма и выбирает наилучший результат по total SSW,
        перед завершением запускает сортировку точек
        """
        self.configs = []
        for i in range(n_start):
            self.perform(heur, prec)
        
        best_config = self.find_best(n_start)
        self.centroids = best_config[0]
        self.labels = best_config[1]
        self.X = best_config[2]
        self.SSW = best_config[3]
        self.within_cluster_squares = best_config[4]
        self.sort_points()

        return self

class elbow:
    """
    the way to find optimal clusters count
    """
    def __init__(self, X, n_features=2, heur='sample', kmax=10, n_start=3):
        self.kmax = kmax
        self.variances_for_k = np.zeros(kmax)
        self.models = []
        for k in range(kmax):
            m = KMeans(X, n_clusters=k+1, n_features=n_features).fit(heur=heur, n_start=n_start)
            self.variances_for_k[k] = m.SSW
            self.models.append(m)
        
    def choose_k(self):
        """
        researches the values of SSW to find optimal count
        """
        diffrences = np.zeros(self.kmax)
        for k in range(1, self.kmax):
            diffrences[k] = self.variances_for_k[k - 1] - self.variances_for_k[k]
        
        proportions = np.zeros(self.kmax)
        for k in range(1, self.kmax):
            proportions[k] = diffrences[k] / self.variances_for_k[k]
        
        
        return np.argmin(proportions[1:]) + 1


class pair_matrix:
    """
    реализует построение и отрисовку матриц попарных расстояний
    """
    def compute_pair_distances(model):
        """
        заполняет матрицу попарных расстояний 
        """
        model.pair_distances = np.zeros((model.n_points, model.n_points))

        for i in range(model.n_points):
            model.pair_distances[i] = np.linalg.norm(model.X - model.X[i], axis=-1)
        

    def get_pair_submatrix(model, *clust_nums):
        """
        для матрицы попарных расстояний возвращает её главную подматрицу с кластерами,
        обозначенными переданными индексами `*clust_nums`
        """
        mask = np.isin(model.labels, clust_nums)
        return model.pair_distances[mask].T[mask]
    
    def plot(n_points=2000, n_clusters=5):
        """
        рисует таблицу графиков, состоящую из трёх столбцов: кластеры на плоскости,
        матрица попарных расстояний текущего кластера, гистограмма распределения
        расстояний от точек кластера до центра
        """
        X, tags = make_blobs(n_samples=n_points, n_features=2, centers=n_clusters, random_state=0)

        m = KMeans(X, n_clusters=5).fit()
        pair_matrix.compute_pair_distances(m)

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        labels = np.take(colors, m.labels.astype(int))

        fig, ax = plt.subplots(1 + m.n_clusters, 3, figsize=(20,40))
        ax[0, 0].scatter(m.X[:, 0], m.X[:, 1], c=labels)
        for i in range(m.n_clusters):
            ax[0, 0].annotate(str(i + 1), m.centroids[i], c="white", fontsize=20)

        ax[0, 0].tick_params(axis="y", which="both", left=False, labelleft=False)
        ax[0, 0].tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        ax[0, 0].set_title("Кластеры")

        ax[0, 1].matshow(m.pair_distances)
        ax[0, 1].tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        ax[0, 1].set_title("Матрицы попарных расстояний")

        ax[0, 2].bar(np.arange(1, m.n_clusters + 1, dtype=int), m.within_cluster_squares, color=colors)
        ax[0, 2].set_title("Вариация в кластерах")
        ax[0, 2].grid(axis="y")
        ax[0, 2].set_xlabel("Кластеры")

        for i in range(1, m.n_clusters + 1):
            ax[i, 0].scatter(m.clusters[i - 1][:, 0], m.clusters[i - 1][:, 1], c=colors[i - 1])
            ax[i, 0].scatter(m.centroids[i - 1, 0], m.centroids[i - 1, 1], c=colors[i], s=130)

            ax[i, 0].tick_params(axis="y", which="both", left=False, labelleft=False)
            ax[i, 0].tick_params(axis="x", which="both", bottom=False, labelbottom=False)
            ax[i, 0].set_ylabel(f"Кластер {i}")

            ax[i, 1].matshow(pair_matrix.get_pair_submatrix(m, i - 1))    

            ax[i, 2].hist(m.distances[i - 1], bins=15, color=colors[i - 1])
            ax[i, 2].set_xlim([0, 5])

        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        date = localtime()
        save_path = f"pair_matrix_{date.tm_year}_{date.tm_mon}_{date.tm_mday}_{date.tm_hour}_{date.tm_min}_{date.tm_sec}"
        plt.savefig(save_path)
        plt.show()


class Test_time:
    """
    нужно для исследования зависимости времени выполнения от объёма данных и отрисовки
    соответствующего графика
    """
    def test(self, points_range=[10, 101], points_step=10, n_clusters=7):
        """
        прогоняет `kmeans` по разным входным данным, у которых количество точек меняется в пределах `points_range`
        с шагом `points_step`; число кластеров изменяется от 1 до `n_clusters'.
        во время работы сохраняет результат в файл `data.npy` в директории 
        f"time_{date.tm_year}_{date.tm_mon}_{date.tm_mday}_{date.tm_hour}_{date.tm_min}_{date.tm_sec}"
        """
        date = localtime()
        current_dir = f"time_{date.tm_year}_{date.tm_mon}_{date.tm_mday}_{date.tm_hour}_{date.tm_min}_{date.tm_sec}"
        mkdir(current_dir)
        result = np.ones((n_clusters, (points_range[1] - points_range[0]) // points_step + 1, 3))
        for i in range(0, n_clusters):
            for j in range(points_range[0], points_range[1], points_step):
                X, y = make_blobs(n_samples=j, centers=i+1)

                start_time = time()
                KMeans(X, i+1).fit(heur="random")
                result[i, (j - points_range[0]) // points_step, 0] = time() - start_time

                start_time = time()
                KMeans(X, i+1).fit(heur="sample")
                result[i, (j - points_range[0]) // points_step, 1] = time() - start_time

                start_time = time()
                KMeans(X, i+1).fit(heur="distant")
                result[i, (j - points_range[0]) // points_step, 2] = time() - start_time
        
        np.save(f"{current_dir}/time", result)
        self.dir = current_dir
        self.points_range = points_range
        self.points_step = points_step
        self.n_clusters = n_clusters
        self.data = result
        return self
    
    def plot(self, heur):
        """
        строит `plt.stackplot` с заголовком f"Время работы `kmeans` ({heur})"
        и сохраняет его в директории теста, где сохранялся файл `data.npy`
        """
        fig, ax = plt.subplots(figsize=(15,5))
        ax.grid(axis='y', alpha=0.4)
        ax.set_xlim([self.points_range[0], self.points_range[1]])

        dct = {
            "random": 0,
            "sample": 1,
            "distant": 2
        }

        tags = [f"{i+1} кластеров" for i in range(self.n_clusters)]

        for i in range(self.n_clusters - 1, 0, -1):  # for each n_clusters
            ax.stackplot(np.arange(self.points_range[0], self.points_range[1], self.points_step), self.data[i, :, dct[heur]], labels=tags[i])

        ax.legend(title="Число кластеров", loc='upper left')
        ax.set_ylabel("Секунды")
        ax.set_xlabel("Число точек")
        ax.set_title(f"Время работы `kmeans` ({heur})", fontsize=22)
        plt.savefig(f"{self.dir}/plot")


class Test_heur:    
    """
    нужен для исследования разных стратегий инициализации на точность результата
    строит соответствующую табллицу
    """
    def test(self, points_range=[100, 1000], points_step = 100, n_clusters=10):
        """
        прогоняет `kmeans` по разным входным данным, у которых количество точек меняется в пределах `points_range`
        с шагом `points_step`; число кластеров изменяется от 1 до `n_clusters`.
        """
        date = localtime()
        current_dir = f"heur_{date.tm_year}_{date.tm_mon}_{date.tm_mday}_{date.tm_hour}_{date.tm_min}_{date.tm_sec}"
        mkdir(current_dir)
        for i in range(n_clusters):
            for j in range(points_range[0], points_range[1] + 1, points_step):
                X, y = make_blobs(n_samples=j, centers=i+1)
                save_dir = f"{current_dir}/{i}_{j}"
                mkdir(save_dir)

                np.save(f"{save_dir}/X", np.array(X))
                np.save(f"{save_dir}/tags", np.array(y))            
                
                m = KMeans(X, i+1).fit(heur="random")
                np.save(f"{save_dir}/random_X", m.X)
                np.save(f"{save_dir}/random_tags", m.labels)

                m = KMeans(X, i+1).fit(heur="sample")
                np.save(f"{save_dir}/sample_X", m.X)
                np.save(f"{save_dir}/sample_tags", m.labels)

                m = KMeans(X, i+1).fit(heur="distant")
                np.save(f"{save_dir}/distant_X", m.X)
                np.save(f"{save_dir}/distant_tags", m.labels)
        
        self.dir = current_dir
        self.points_range = points_range
        self.points_step = points_step
        self.n_points = (points_range[1] - points_range[0]) // points_step + 1
        self.n_clusters = n_clusters
        return self
    
    def read(self, heur, n_clusters, n_points):
        """
        вспомогательная функция чтения из файла `data.npy`
        """
        current_dir = f"{self.dir}/{n_clusters}_{n_points}" 
        X = np.load(f"{current_dir}/{heur}_X.npy")
        y = np.load(f"{current_dir}/{heur}_tags.npy")
        X_r = np.load(f"{current_dir}/X.npy")
        y_r = np.load(f"{current_dir}/tags.npy")
        return (X, y, X_r, y_r)
    
    def plot(self, heur):        
        """
        вдоль горизонтали увеличивается количество точек (от 100 до 1000), а вдоль вертикали увеличивается
        число кластеров (от 1 до 10). В каждой ячейке таблицы результат применения `Kmeans`.
        В качестве тестовых данных `sklearn.datasets.make_blobs`.
        """
        fig, ax = plt.subplots(self.n_clusters, self.n_points, figsize=(40, 40), sharex=True, sharey=True)

        for i in range(self.n_clusters):
            for j in range(self.n_points):
                X, y, *_ = self.read(heur, i, self.points_range[0] + j * self.points_step)
                ax[i,j].scatter(X[:, 0], X[:, 1], c=y)


        plt.subplots_adjust(wspace=0, hspace=0)

        fig.suptitle(f'"{heur}" strategy', fontsize=60)

        for i in range(self.n_clusters):
            for j in range(self.n_points):
                ax[i, j].tick_params(axis="x", which="both", bottom=False, labelbottom=False)
                ax[i, j].tick_params(axis="y", which="both", left=False, labelleft=False)

        for j in range(self.n_points):
            ax[0, j].set_title(str(self.points_range[0] + j * self.points_step), fontsize=30)
            ax[-1, j].set_xlabel(str(self.points_range[0] + j * self.points_step), fontsize=30)

        for i in range(self.n_clusters):
            ax[i, 0].set_ylabel(str(i + 1), fontsize=30)
            ax[i, -1].set_ylabel(str(i + 1), fontsize=30)
            ax[i, -1].yaxis.set_label_position("right")

        plt.savefig(f"{self.dir}/{heur}")
        plt.show()
    