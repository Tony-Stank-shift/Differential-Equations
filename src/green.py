"""格林函数与汉克尔函数 —— 二维标量亥姆霍兹方程。

物理背景
--------
二维 TM 极化下,理想导体(PEC)柱体散射满足标量亥姆霍兹方程

    (d^2/dx^2 + d^2/dy^2 + k^2) u = 0,      u = E_z

其格林函数为

    G(r, r') = (i/4) H_0^{(2)}(k |r - r'|)

其中 H_0^{(2)} 是第二类零阶汉克尔函数。

本模块提供两条**相互独立**的计算路径
------------------------------------
1. ``hankel0_2``       —— |x| >= 阈值时直接调用 ``scipy.special.hankel2``(基准实现)
2. ``hankel0_2_small`` —— |x| 很小时用自实现的级数展开

第二条路径存在的理由
--------------------
当 kR -> 0 时 H_0^{(2)} 具有对数奇异:

    H_0^{(2)}(x) = J_0(x) - i Y_0(x)
    Y_0(x) = (2/pi) [ (ln(x/2) + gamma) J_0(x)
                      + sum_{m>=1} (-1)^{m+1} H_m (x/2)^{2m} / (m!)^2 ]

把 ``(2i/pi) ln(x/2)`` 这一项显式分离出来后,奇异积分(``quad.py``)
才能对自耦合项做解析处理 —— 用数值求积去啃对数奇异性必然失败。
这是本项目最容易翻车的地方之一。
"""
from __future__ import annotations

import numpy as np
from scipy.special import hankel2

__all__ = [
    "EULER_GAMMA",
    "SMALL_THRESHOLD",
    "hankel0_2",
    "hankel0_2_small",
    "green_2d",
    "log_singular_part",
]

#: 欧拉-马歇罗尼常数
EULER_GAMMA = 0.57721566490153286060651209008240243104215933593992

#: |x| 小于该阈值时改用级数分支
SMALL_THRESHOLD = 0.5


def hankel0_2_small(x, n_terms: int = 60):
    """|x| 很小时 H_0^{(2)}(x) 的级数展开(不依赖 scipy)。

    使用的展开式:

        J_0(x) = sum_{m>=0} (-1)^m (x/2)^{2m} / (m!)^2
        Y_0(x) = (2/pi) [ (ln(x/2) + gamma) J_0(x)
                          + sum_{m>=1} (-1)^{m+1} H_m (x/2)^{2m} / (m!)^2 ]

    其中 H_m = sum_{j=1..m} 1/j 为第 m 个调和数。

    Parameters
    ----------
    x : array_like
        自变量,通常为 kR。可以是标量、数组,允许复数。
    n_terms : int
        级数保留项数。|x| <= 0.5 时 60 项远超所需精度。

    Returns
    -------
    ndarray (complex)
        与输入同形状的 H_0^{(2)}(x)。标量输入返回 0 维数组,交给上层处理。
    """
    x = np.asarray(x, dtype=complex)
    xh = x / 2.0
    xh2 = xh * xh

    j0 = np.ones_like(xh)
    term = np.ones_like(xh)          # term = (-1)^m (x/2)^{2m} / (m!)^2
    harmonic = 0.0
    y0_sum = np.zeros_like(xh)

    for m in range(1, n_terms + 1):
        term = -term * xh2 / (m * m)
        j0 = j0 + term
        harmonic += 1.0 / m
        # (-1)^{m+1} H_m (x/2)^{2m} / (m!)^2 = -H_m * term
        y0_sum = y0_sum - harmonic * term

    y0 = (2.0 / np.pi) * ((np.log(xh) + EULER_GAMMA) * j0 + y0_sum)
    return j0 - 1j * y0


def hankel0_2(x, threshold: float = SMALL_THRESHOLD):
    """H_0^{(2)}(x),自动在基准分支与级数分支之间切换。

    支持标量与任意形状数组。|x| < threshold 走级数分支,
    否则调用 ``scipy.special.hankel2``。

    Parameters
    ----------
    x : array_like
        自变量。允许复数(级数分支天然支持;基准分支由 scipy 处理)。
    threshold : float
        两条分支的切换阈值,默认 ``SMALL_THRESHOLD``。

    Returns
    -------
    complex 或 ndarray (complex)
        标量输入返回 Python 复数,数组输入返回同形状复数组。
    """
    x_arr = np.atleast_1d(np.asarray(x, dtype=complex))
    out = np.empty_like(x_arr)
    small = np.abs(x_arr) < threshold
    if np.any(small):
        out[small] = hankel0_2_small(x_arr[small])
    if np.any(~small):
        out[~small] = hankel2(0, x_arr[~small])
    if np.ndim(x) == 0:
        return complex(out[0])
    return out.reshape(np.shape(x))


def green_2d(k, r, rp):
    """二维标量格林函数 G(r, r') = (i/4) H_0^{(2)}(k |r - r'|)。

    Parameters
    ----------
    k : float
        波数,实数(无耗情形)。
    r, rp : array_like, shape (..., 2)
        场点与源点坐标。

    Returns
    -------
    ndarray (complex)
        与广播后的 r - rp 形状一致(去掉最后一维坐标轴)。
    """
    r_arr = np.asarray(r, dtype=float)
    rp_arr = np.asarray(rp, dtype=float)
    R = np.linalg.norm(r_arr - rp_arr, axis=-1)
    return 0.25j * hankel0_2(k * R)


def log_singular_part(k, R):
    """H_0^{(2)} 的对数奇异部分:-(2i/pi) * [ ln(kR/2) + gamma ]。

    由 Y_0 的展开式

        H_0^{(2)} = J_0 - i Y_0
                  = J_0 - (2i/pi)[ (ln(x/2) + gamma) J_0 + ... ]
                  = -(2i/pi)[ ln(x/2) + gamma ] + [正则余项]

    这里取**完整的常数项**(含 Euler 常数 gamma),而不是只留 ln(x/2),
    使得余项在 x -> 0 时有干净极限:

        H_0^{(2)}(x) - log_singular_part(x)  ->  1

    这正是 ``quad.py`` 拆 "解析可积部分 + 正则余项" 时需要的性质 ——
    漏掉 gamma 会让余项极限变成 1 + 2i*gamma/pi,虽仍是常数但不再干净,
    后续解析积分容易算错。
    """
    arr = np.asarray(k * R, dtype=complex)
    return -(2.0j / np.pi) * (np.log(arr / 2.0) + EULER_GAMMA)
