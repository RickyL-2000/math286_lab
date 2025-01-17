# %%
import pandas as pd
import numpy as np
from typing import List, Tuple


# %%
def f1(t, y):
    return y*y + t*y + t*t

def f2(t, y):
    return y*y*y + t*y*y + t*t*y + t*t*t

# %%
def analyse_step_len(f, method, a, b, t0, y0, *h):
    """
    The main process to analyse a set of results with different step lengths

    :param f: the f function of the IVP
    :param method: the numerical method
    :param a: left bound
    :param b: right bound
    :param t0: initial t
    :param y0: initial y
    :param h: a list of step lengths to be analysed, h=(0.01, 0.005, 0.001)
    :return: a DataFrame of numerical results
    """

    if not h:
        h = (0.01, 0.005, 0.001)

    df = pd.DataFrame()
    space = h[0]
    df['t'] = np.linspace(a, b, round((b - a) / space) + 1)
    for i in range(len(h)):
        t, y = method(f, a, b, t0, y0, h[i])
        df['y with h='+str(h[i])] = [y[j] for j in range(0, len(y), round(space / h[i]))]

    return df

# %%
def euler_explicit(f, a, b, t0, y0, h) -> Tuple[np.ndarray, np.ndarray]:
    """
    Explicit Euler Method

    :param f: the f function
    :param a: left bound
    :param b: right bound
    :param t0: initial t
    :param y0: initial y
    :param h: step length
    :return: list of numerical results of t and y
    """
    assert a <= t0 <= b
    t_list, y_list = [], []
    ti, yi = t0, y0
    t_temp, y_temp = [], []
    for _ in range(round((t0 - a)/h)):
        y_ = yi - h * f(ti, yi)
        t_temp.append(ti-h), y_temp.append(y_)
        ti, yi = ti-h, y_

    if t_temp and y_temp:
        t_temp.reverse(), y_temp.reverse()
        t_list.extend(t_temp), y_list.extend(y_temp)

    t_list.append(t0), y_list.append(y0)

    ti, yi = t0, y0
    # while ti+h <= b:
    for _ in range(round((b - t0)/h)):
        y_ = yi + h * f(ti, yi)
        t_list.append(ti+h), y_list.append(y_)
        ti, yi = ti+h, y_

    return np.array(t_list), np.array(y_list)

# %%
def euler_implicit(f, a, b, t0, y0, h, threshold=1e-6, epochs=100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Implicit (backward) Euler Method

    :param f: the f function
    :param a: left bound
    :param b: right bound
    :param t0: initial t
    :param y0: initial y
    :param h: step length
    :param threshold: the threshold to control the iteration
    :param epochs: the maximum number of epochs used to control the iteration
    :return: list of numerical results of t and y
    """
    assert a <= t0 <= b
    t_list, y_list = [], []
    ti, yi = t0, y0
    t_temp, y_temp = [], []
    for _ in range(round((t0 - a)/h)):
        y_ = yi - h * f(ti, yi)
        epoch = 0
        while True:
            epoch += 1
            y__ = yi - h * f(ti-h, y_)
            if abs(y__ - y_) < threshold or epoch > epochs:
                break
            y_ = y__
        t_temp.append(ti-h), y_temp.append(y_)
        ti, yi = ti-h, y_

    if t_temp and y_temp:
        t_temp.reverse(), y_temp.reverse()
        t_list.extend(t_temp), y_list.extend(y_temp)

    t_list.append(t0), y_list.append(y0)

    ti, yi = t0, y0
    for _ in range(round((b - t0) / h)):
        y_ = yi + h * f(ti, yi)
        epoch = 0
        while True:
            epoch += 1
            y__ = yi + h * f(ti + h, y_)
            if abs(y__ - y_) < threshold or epoch > epochs:
                break
            y_ = y__
        t_list.append(ti + h), y_list.append(y_)
        ti, yi = ti + h, y_

    return np.array(t_list), np.array(y_list)

# %%
def euler_trapezium(f, a, b, t0, y0, h, threshold=1e-6, epochs=50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Implicit (backward) Euler Method

    :param f: the f function
    :param a: left bound
    :param b: right bound
    :param t0: initial t
    :param y0: initial y
    :param h: step length
    :param threshold: the threshold to control the iteration
    :param epochs: the maximum number of epochs used to control the iteration
    :return: list of numerical results of t and y
    """
    assert a <= t0 <= b
    t_list, y_list = [], []
    ti, yi = t0, y0
    t_temp, y_temp = [], []
    for _ in range(round((t0 - a) / h)):
        y_ = yi - h * f(ti, yi)
        epoch = 0
        while True:
            epoch += 1
            y__ = yi - 0.5 * h * (f(ti, yi) + f(ti - h, y_))
            if abs(y__ - y_) < threshold or epoch > epochs:
                break
            y_ = y__
        t_temp.append(ti - h), y_temp.append(y_)
        ti, yi = ti - h, y_

    if t_temp and y_temp:
        t_temp.reverse(), y_temp.reverse()
        t_list.extend(t_temp), y_list.extend(y_temp)

    t_list.append(t0), y_list.append(y0)

    ti, yi = t0, y0
    for _ in range(round((b - t0) / h)):
        y_ = yi + h * f(ti, yi)
        epoch = 0
        while True:
            epoch += 1
            y__ = yi + 0.5 * h * (f(ti, yi) + f(ti + h, y_))
            if abs(y__ - y_) < threshold or epoch > epochs:
                break
            y_ = y__
        t_list.append(ti + h), y_list.append(y_)
        ti, yi = ti + h, y_

    return np.array(t_list), np.array(y_list)

# %%
def euler_improved(f, a, b, t0, y0, h) -> Tuple[np.ndarray, np.ndarray]:
    """
    Improved Euler Method
    """
    assert a <= t0 <= b
    t_list, y_list = [], []
    ti, yi = t0, y0
    t_temp, y_temp = [], []
    for _ in range(round((t0 - a)/h)):
        y_ = yi - h * f(ti, yi)
        y_ = yi - 0.5 * h * (f(ti, yi) + f(ti-h, y_))
        t_temp.append(ti-h), y_temp.append(y_)
        ti, yi = ti-h, y_
    if t_temp and y_temp:
        t_temp.reverse(), y_temp.reverse()
        t_list.extend(t_temp), y_list.extend(y_temp)

    t_list.append(t0), y_list.append(y0)

    ti, yi = t0, y0
    for _ in range(round((b - t0)/h)):
        y_ = yi + h * f(ti, yi)
        y_ = yi + 0.5 * h * (f(ti, yi) + f(ti+h, y_))
        t_list.append(ti+h), y_list.append(y_)
        ti, yi = ti+h, y_

    return np.array(t_list), np.array(y_list)
