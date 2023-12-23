import numpy as np


def grad_finite_diff(function, w, eps=1e-8, **kwargs):
    """
    Возвращает численное значение градиента, подсчитанное по следующией формуле:
        result_i := (f(w + eps * e_i) - f(w)) / eps,
        где e_i - следующий вектор:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    def util(x):
        return (function(w=w+x, **kwargs) - function(w=w, **kwargs)) / eps    
    return np.apply_along_axis(util, axis=1, arr=np.eye(w.size) * eps)
    # return np.apply_along_axis(util, axis=1, arr=np.eye(w.size) * eps)
