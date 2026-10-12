"""格林函数与汉克尔函数 —— 二维标量亥姆霍兹方程。

⚠️ 全局约定(整个项目统一,不得单独更改其中任何一条)
---------------------------------------------------
    时谐因子      e^{+jωt}
    出射波        e^{-jkρ}        (与 H_0^{(2)} 的远场相位一致)
    源项          (∇² + k²) G = +δ(r - r')
    入射平面波    u_inc = exp(+j k (x cosθ_i + y sinθ_i))
    格林函数      G(r, r') = (j/4) H_0^{(2)}(k |r - r'|)

这五条是**绑在一起**的。单独改一条(例如给 G 前面的 (j/4) 加个负号)
会让整套符号不自洽,而不是"换个等价写法"。

若将来要改用 e^{-iωt} 约定(为对齐 Colton–Kress 等文献),**不是改个符号**,
必须同时改三处:
    * ``green_2d`` 把 H_0^{(2)} 换成 H_0^{(1)}
    * ``hankel0_2_small`` 增加 H^{(1)} 分支(``j0 - 1j * y0`` → ``j0 + 1j * y0``)
    * ``u_inc`` 的指数符号取反
动手前请先读 DECISIONS.md 的 D6。

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

import warnings

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
        级数保留项数。|x| <= 0.5 时远超所需精度 —— 实测要压到 1e-16
        只需 8 项,且相消放大因子仅 0.963(无灾难性相消)。
        本函数的**设计使用区间**是 |x| <= SMALL_THRESHOLD(0.5);超出该区间但
        仍在守卫阈值内时会发出 RuntimeWarning(不是错误,只是提醒它已经越界)。

        实测精度(与 scipy 的 hankel2 逐点比,2026-10-12 复核):
            |x| = 0.5  3.207e-16      |x| = 2    9.960e-17      |x| = 5    6.286e-16
            |x| = 5.01 4.936e-16(告警开始)  |x| = 10  4.396e-13  |x| = 20  7.560e-09
        即 |x| <= 5 仍可用(|x| = 1/2/5 都在 1e-16 量级),**|x| >= 10 起迅速失准**。
        守卫取 5.0 是照这个实测定的;早先版本在 docstring 里声明"只对 |x| <= 0.5
        负责"却把守卫放在 5.0,两者矛盾(复核 E1 指出),现已按实测统一。

    Returns
    -------
    ndarray (complex)
        与输入同形状的 H_0^{(2)}(x)。标量输入返回 0 维数组,交给上层处理。
    """
    x = np.asarray(x, dtype=complex)
    if np.any(np.abs(x) > 5.0):
        warnings.warn(
            "hankel0_2_small 被用在 |x| > 5 上:级数展开在此范围迅速失准"
            "(实测相对误差 |x|=5 约 6e-16,|x|=10 约 4e-13,|x|=20 约 8e-09)。"
            "正常路径请用 hankel0_2(),它会自动切到 scipy 分支。",
            RuntimeWarning,
            stacklevel=2,
        )
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

    Notes
    -----
    ``(j/4)`` 这个因子与时谐约定 ``e^{+jωt}`` 绑定(见模块顶部说明)。
    换约定要连带把 H_0^{(2)} 换成 H_0^{(1)},不是改个正负号。

    R = 0(自耦合)时返回 ``nan + nanj`` 并触发 RuntimeWarning。这不是缺陷:
    自耦合项在数学上本就奇异,必须由 ``quad.py`` 解析处理,不能靠数值求值。
    """
    r_arr = np.asarray(r, dtype=float)
    rp_arr = np.asarray(rp, dtype=float)
    R = np.linalg.norm(r_arr - rp_arr, axis=-1)
    return 0.25j * hankel0_2(k * R)


def log_singular_part(k, R):
    """H_0^{(2)} 的对数奇异部分:-(2j/pi) * [ ln(kR/2) + gamma ]。

    由 Y_0 的展开式

        H_0^{(2)} = J_0 - j Y_0
                  = J_0 - (2j/pi)[ (ln(x/2) + gamma) J_0 + ... ]
                  = -(2j/pi)[ ln(x/2) + gamma ] + [正则余项]

    这里取**完整的常数项**(含 Euler 常数 gamma),而不是只留 ln(x/2),
    使得余项在 x -> 0 时有干净极限:

        H_0^{(2)}(x) - log_singular_part(x)  ->  1

    若漏掉 gamma,余项极限会变成 ``1 - 2j*gamma/pi``(实测
    ``1 - 0.367466905196616j``)—— 仍是常数但不再干净,后续解析积分容易算错。

    ⚠️ 适用门槛(极易踩坑)
    ----------------------
    上式是**渐近**结论,收敛并不快。实测 ``|余项 - 1|``:

        kR = 1e-8    ~ 7e-15
        kR = 1e-6    ~ 2.4e-12
        kR = 1e-2    ~ 9.4e-5
        kR = 0.05    ~ 1.75e-3
        kR = 0.3     ~ 4.0e-2
        kR = 0.5     ~ 9.4e-2      <-- 偏离已近 10%

    **只有 kR ≲ 0.05 时才能把余项当常数 1 用。** ``quad.py`` 处理自耦合项
    必须按这个门槛判断,否则对角元会错约 10%。

    R = 0 时本函数未定义(返回 ``nan+infj`` 并告警)。自耦合项必须由调用方
    解析处理,不要把 R=0 传进来。
    """
    arr = np.asarray(k * R, dtype=complex)
    if np.any(arr == 0):
        warnings.warn(
            "log_singular_part 在 kR=0 处未定义(将返回 nan+infj)。"
            "自耦合项必须由调用方解析处理,不要传 R=0 进来。",
            RuntimeWarning,
            stacklevel=2,
        )
    return -(2.0j / np.pi) * (np.log(arr / 2.0) + EULER_GAMMA)
