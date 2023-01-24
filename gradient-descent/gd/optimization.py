import numpy as np
from gd.oracles import BinaryLogistic
from time import time
from scipy.special import expit
from collections import defaultdict


class GDClassifier:
    def __init__(
        self, loss_function='binary_logistic', step_alpha=1, step_beta=0,
        tolerance=1e-5, max_iter=1000, **kwargs
    ):
        """
        loss_function - строка, отвечающая за функцию потерь классификатора.
        Может принимать значения:
        - 'binary_logistic' - бинарная логистическая регрессия

        step_alpha - float, параметр выбора шага из текста задания

        step_beta- float, параметр выбора шага из текста задания

        tolerance - точность, по достижении которой, необходимо прекратить оптимизацию.
        Необходимо использовать критерий выхода по модулю разности соседних значений функции:
        если |f(x_{k+1}) - f(x_{k})| < tolerance: то выход

        max_iter - максимальное число итераций

        **kwargs - аргументы, необходимые для инициализации
        """

        self.step_alpha = step_alpha
        self.step_beta = step_beta
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.l2_coef = None
        if 'l2_coef' in kwargs:
            self.l2_coef = kwargs['l2_coef']
        else:
            self.l2_coef = 0

    def fit(self, X, y, w_0=None, trace=False):
        """
        Обучение метода по выборке X с ответами y

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        y - одномерный numpy array

        w_0 - начальное приближение в методе

        trace - переменная типа bool

        Если trace = True, то метод должен вернуть словарь history, содержащий информацию
        о поведении метода. Длина словаря history = количество итераций + 1 (начальное приближение)

        history['time']: list of floats, содержит интервалы времени между двумя итерациями метода
        history['func']: list of floats, содержит значения функции на каждой итерации
        (0 для самой первой точки)
        """

        self.solver = BinaryLogistic(self.l2_coef)
        if w_0 is None:
            self.coefs = np.zeros(X.shape[1])
        else:
            self.coefs = w_0.copy()
        step = self.step_alpha

        history = defaultdict(list)
        func_prev = self.get_objective(X, y)
        timer = time()

        for i in range(1, self.max_iter + 1):
            self.coefs -= step * self.get_gradient(X, y)
            func = self.get_objective(X, y)

            history['time'].append(time() - timer)
            history['func'].append(func_prev)
            history['weights'].append(self.coefs.copy())
            if np.abs(func - func_prev) < self.tolerance:
                history['func'].append(func)
                break

            func_prev = func
            timer = time()

            step = self.step_alpha / i ** self.step_beta

        if trace:
            return {key: np.array(val) for key, val in history.items()}

    def predict(self, X):
        return np.sign(X @ self.coefs)

    def predict_proba(self, X):
        proba = expit(X @ self.coefs)
        return np.vstack([1 - proba, proba]).T

    def get_objective(self, X, y):
        return self.solver.func(X, y, self.coefs)

    def get_gradient(self, X, y):
        return self.solver.grad(X, y, self.coefs)

    def get_weights(self):
        return self.coefs


class SGDClassifier(GDClassifier):
    """
    Реализация метода стохастического градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """

    def __init__(
        self, loss_function='binary_logistic', batch_size=100, step_alpha=1, step_beta=0,
        tolerance=1e-5, max_iter=1000, random_seed=153, **kwargs
    ):
        """
        loss_function - строка, отвечающая за функцию потерь классификатора.
        Может принимать значения:
        - 'binary_logistic' - бинарная логистическая регрессия

        batch_size - размер подвыборки, по которой считается градиент

        step_alpha - float, параметр выбора шага из текста задания

        step_beta- float, параметр выбора шага из текста задания

        tolerance - точность, по достижении которой, необходимо прекратить оптимизацию
        Необходимо использовать критерий выхода по модулю разности соседних значений функции:
        если |f(x_{k+1}) - f(x_{k})| < tolerance: то выход

        max_iter - максимальное число итераций (эпох)

        random_seed - в начале метода fit необходимо вызвать np.random.seed(random_seed).
        Этот параметр нужен для воспроизводимости результатов на разных машинах.

        **kwargs - аргументы, необходимые для инициализации
        """

        self.batch_size = batch_size
        self.step_alpha = step_alpha
        self.step_beta = step_beta
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.random_seed = random_seed
        if 'l2_coef' in kwargs:
            self.l2_coef = kwargs['l2_coef']
        else:
            self.l2_coef = 0

    def fit(self, X, y, w_0=None, trace=False, log_freq=1):
        """
        Обучение метода по выборке X с ответами y

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        y - одномерный numpy array

        w_0 - начальное приближение в методе

        Если trace = True, то метод должен вернуть словарь history, содержащий информацию
        о поведении метода. Если обновлять history после каждой итерации, метод перестанет
        превосходить в скорости метод GD. Поэтому, необходимо обновлять историю метода лишь
        после некоторого числа обработанных объектов в зависимости от приближённого номера эпохи.
        Приближённый номер эпохи:
            {количество объектов, обработанных методом SGD} / {количество объектов в выборке}

        log_freq - float от 0 до 1, параметр, отвечающий за частоту обновления.
        Обновление должно проиходить каждый раз, когда разница между двумя значениями приближённого номера эпохи
        будет превосходить log_freq.

        history['epoch_num']: list of floats, в каждом элементе списка будет записан приближённый номер эпохи:
        history['time']: list of floats, содержит интервалы времени между двумя соседними замерами
        history['func']: list of floats, содержит значения функции после текущего приближённого номера эпохи
        history['weights_diff']: list of floats, содержит квадрат нормы разности векторов весов с соседних замеров
        (0 для самой первой точки)
        """
        if self.random_seed:
            np.random.seed(self.random_seed)

        self.solver = BinaryLogistic(self.l2_coef)

        # initialize solution
        if w_0 is None:
            self.coefs = np.zeros(X.shape[1])
        else:
            self.coefs = w_0.copy()

        # initialize step
        step = self.step_alpha
        visited_batches_counter = 0

        # initialize convergence history
        history = defaultdict(list)
        func_prev = self.get_objective(X, y)
        weights_prev = self.coefs.copy()
        epoch_num_prev = 0
        timer = time()

        for i in range(1, self.max_iter + 1):
            indices = np.random.choice(X.shape[0], size=self.batch_size)
            self.coefs -= step * self.get_gradient(X[indices],
                                                   y[indices])
            func = self.get_objective(X, y)
            visited_batches_counter += 1

            epoch_num = visited_batches_counter * self.batch_size / X.shape[0]
            if epoch_num - epoch_num_prev > log_freq or abs(func - func_prev) < self.tolerance:
                epoch_num_prev = epoch_num
                history['epoch_num'].append(epoch_num)
                history['time'].append(time() - timer)
                timer = time()
                history['func'].append(func)
                history['weights_diff'].append(
                    np.sum((self.coefs - weights_prev) ** 2))
                history['weights'].append(self.coefs.copy())
                weights_prev = self.coefs.copy()
                if np.abs(func - func_prev) < self.tolerance:
                    break

            func_prev = func
            step = self.step_alpha / i ** self.step_beta

        if trace:
            return history
