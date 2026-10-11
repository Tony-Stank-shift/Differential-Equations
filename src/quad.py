"""奇异积分处理 —— 矩量法自耦合项的核积分。

奇异从哪来
----------
二维 TM 的格林函数是 ``G = (j/4) H_0^{(2)}(kR)``。R -> 0 时

    H_0^{(2)}(x) = 1 - (2j/pi)[ ln(x/2) + gamma ] + O(x^2 ln x)

带对数发散。矩量法里

    Z_mn = -∫_{段 n} G(r_m, r') dl'

当 m == n(自耦合项)时 r' 会扫过场点 r_m,核取到 R = 0。此时**任何**普通数值
求积(Gauss-Legendre、Simpson、梯形)都会失败 —— 被积函数本身发散。

主路径:级数逐项解析积分(``series_self_integral``)
------------------------------------------------
半径 ``x = k*a <= SMALL_THRESHOLD`` 时,把 H_0^{(2)} 的级数

    J_0(kt) = sum_m c_m (kt)^{2m},                c_m = (-1)^m / (4^m (m!)^2)
    Y_0(kt) = (2/pi)[ (ln(kt/2)+gamma) J_0(kt)
                      + sum_{m>=1} d_m (kt)^{2m} ],  d_m = (-1)^{m+1} H_m / (4^m (m!)^2)

逐项解析积分。用到的两个原函数:

    ∫_0^a (kt)^{2m} dt                        = A_m
    ∫_0^a (kt)^{2m} [ ln(kt/2) + gamma ] dt   = A_m * L - B_m

其中

    A_m = x^{2m} a / (2m+1),   B_m = A_m / (2m+1),   L = ln(x/2) + gamma

于是

    ∫_{-a}^{a} H_0^{(2)}(k|t|) dt
      = 2 * [ sum_m c_m A_m
              - (2j/pi) * ( sum_m c_m (A_m L - B_m) + sum_{m>=1} d_m A_m ) ]

**没有数值求积,精度只受级数截断限制**(ka <= 0.5 时 60 项远超双精度)。

ka 较大时:分段(``self_coupling_integral``)
-------------------------------------------
ka > 0.5 时级数收敛范围不够。取 ``delta = 0.5/k``,分成

    ∫_{-a}^{a} = ∫_{-delta}^{delta}   (级数,仍然精确)
               + 2 * ∫_{delta}^{a}    (k|t| >= 0.5,被积函数光滑,Gauss-Legendre)

两条路都绕开了奇点,又各自在擅长的区间工作。

备用路径:拆主部 + 数值求积(``regular_part`` / ``analytic_part_integral``)
------------------------------------------------------------------------
    H_0^{(2)}(x) = S(x) + x^2[c0 + c1 ln(x/2)] + R2(x)
    S(x)  = -(2j/pi)[ ln(x/2) + gamma ]
    c1    = j/(2pi)
    c0    = -1/4 - j(1-gamma)/(2pi)
    R2(x) = O(x^4 ln x)     <- 光滑,数值求积可用

这条路在本模块里**不是**自耦合项的默认实现(级数路径更快更准),保留它是为了
1.3 的近奇异积分:场点靠近但不在段上时,奇异点会移到段外,那时"拆解析主部 +
数值积余项"才是通用做法。

⚠️ 绝不要做的事
--------------
**不要把 R(x) = 1 + O(x^2 ln x) 里的余项当常数 1 解析积。** ``R -> 1`` 是渐近
结论且收敛不快:实测 kR = 0.5 时 ``|R - 1| ≈ 9.4e-2``,直接当 1 会让自耦合项错
近 10%。``naive_self_coupling`` 保留了这个错误做法,只为在自检里量化它的危害。
"""
from __future__ import annotations

import numpy as np

try:  # 作为包导入(import src.quad)
    from .green import EULER_GAMMA, SMALL_THRESHOLD, hankel0_2, log_singular_part
except ImportError:  # src/ 在 sys.path 上(import quad)
    from green import EULER_GAMMA, SMALL_THRESHOLD, hankel0_2, log_singular_part

__all__ = [
    "DEFAULT_QUAD_ORDER",
    "C0",
    "C1",
    "gauss_legendre",
    "log_term_integral",
    "poly2_term_integral",
    "analytic_part_integral",
    "regular_part",
    "series_self_integral",
    "self_coupling_integral",
    "direct_gauss",
    "naive_self_coupling",
]

#: Gauss-Legendre 默认节点数(只用在 ka > 0.5 的外段,那里被积函数光滑)
DEFAULT_QUAD_ORDER = 64

#: 二阶主部的对数系数:R(x) 中 x^2 ln(x/2) 的系数
C1 = 1j / (2.0 * np.pi)

#: 二阶主部的常数系数
C0 = -0.25 - 1j * (1.0 - EULER_GAMMA) / (2.0 * np.pi)


def gauss_legendre(order: int = DEFAULT_QUAD_ORDER):
    """[-1, 1] 上的 Gauss-Legendre 节点与权重。"""
    nodes, weights = np.polynomial.legendre.leggauss(int(order))
    return nodes, weights


# ======================================================================
# 解析主部(备用路径;近奇异积分会用)
# ======================================================================
def log_term_integral(k, half_len):
    """闭式积分 ``∫_{-a}^{a} S(k|t|) dt``,S 为对数主部,a = half_len。

        S(x)  = -(2j/pi)[ ln(x/2) + gamma ]
        ∫_0^a ln(kt/2) dt = a ln(ka/2) - a
        => ∫_{-a}^{a} S(k|t|) dt = -(2j/pi) * 2a * [ ln(ka/2) + gamma - 1 ]
    """
    k = float(k)
    a = np.asarray(half_len, dtype=float)
    return -(2.0j / np.pi) * 2.0 * a * (np.log(k * a / 2.0) + EULER_GAMMA - 1.0)


def poly2_term_integral(k, half_len):
    """闭式积分 ``∫_{-a}^{a} (k|t|)^2 [C0 + C1 ln(k|t|/2)] dt``。

    ∫_0^a t^{2m} ln(kt/2) dt = a^{2m+1}/(2m+1) * ln(ka/2) - a^{2m+1}/(2m+1)^2

    取 2m = 2 得到

        ∫_{-a}^{a} (k|t|)^2 [C0 + C1 ln(k|t|/2)] dt
          = 2 k^2 a^3 / 3 * [ C0 + C1 * ( ln(ka/2) - 1/3 ) ]
    """
    k = float(k)
    a = np.asarray(half_len, dtype=float)
    return 2.0 * k * k * a**3 / 3.0 * (C0 + C1 * (np.log(k * a / 2.0) - 1.0 / 3.0))


def analytic_part_integral(k, half_len):
    """``∫_{-a}^{a} [ S + 二阶主部 ] dt`` 的闭式和(备用路径用)。"""
    return log_term_integral(k, half_len) + poly2_term_integral(k, half_len)


def regular_part(k, t):
    """余项 ``R2(k|t|) = H_0^{(2)}(k|t|) - S - (k|t|)^2[C0 + C1 ln(k|t|/2)]``。

    它满足 ``R2(0) = 1`` 且 ``R2(x) - 1 = O(x^4 ln x)`` —— **正是这一项才是光滑到
    可以数值求积的**(只减对数主部时,``R - 1 = O(x^2 ln x)``,40 点以下的
    Gauss-Legendre 会明显不够)。

    Parameters
    ----------
    k : float
    t : array_like
        相对段中点的弧长坐标(可正可负,内部取 ``|t|``)。

    Returns
    -------
    ndarray (complex)
        与输入同形状。t = 0 处返回精确的 1 + 0j。
    """
    k = float(k)
    t_arr = np.asarray(t, dtype=float)
    t_flat = np.atleast_1d(t_arr)
    out = np.empty(t_flat.shape, dtype=complex)

    nonzero = np.abs(t_flat) > 0.0
    if np.any(nonzero):
        tt = np.abs(t_flat[nonzero])
        x = k * tt
        out[nonzero] = (
            hankel0_2(x)
            - log_singular_part(k, tt)
            - x**2 * (C0 + C1 * np.log(x / 2.0))
        )
    if np.any(~nonzero):
        out[~nonzero] = 1.0 + 0.0j

    return out.reshape(t_arr.shape)


# ======================================================================
# 主路径
# ======================================================================
def series_self_integral(k, half_len, n_terms: int = 60) -> complex:
    """级数逐项解析积分 ``∫_{-a}^{a} H_0^{(2)}(k|t|) dt``。

    **只在 x = k*a <= SMALL_THRESHOLD(0.5) 时有效** —— 这是 H_0^{(2)} 级数展开
    的收敛范围(x 稍大时级数项会先变大再变小,精度崩坏)。超出范围抛 ValueError;
    一般请直接调用 ``self_coupling_integral``,由它负责分段。

    Parameters
    ----------
    k, half_len : float
        波数与半段长。
    n_terms : int
        级数项数。x <= 0.5 时 60 项远超双精度所需。

    Returns
    -------
    complex
    """
    k = float(k)
    a = float(half_len)
    x = k * a
    if not x > 0.0:
        raise ValueError(f"需要 k * half_len > 0,收到 k={k!r}, half_len={half_len!r}")
    if x > SMALL_THRESHOLD:
        raise ValueError(
            f"级数路径只对 k*half_len <= {SMALL_THRESHOLD} 负责,收到 {x:.6g}。"
            "请用 self_coupling_integral(会自动分段)。"
        )

    L = float(np.log(x / 2.0) + EULER_GAMMA)
    x2 = x * x
    c = 1.0          # c_m
    xm = 1.0         # x^{2m}
    harmonic = 0.0
    j0_sum = 0.0     # sum c_m A_m
    y_sum = 0.0      # sum c_m (A_m L - B_m) + sum_{m>=1} d_m A_m

    for m in range(0, n_terms + 1):
        if m > 0:
            # c_m = (-1)^m / (4^m (m!)^2) —— 注意系数里**不含 x**,x^{2m} 由 xm 承担
            c = -c / (4.0 * m * m)
            xm *= x2
            harmonic += 1.0 / m
        A = xm * a / (2.0 * m + 1.0)
        B = A / (2.0 * m + 1.0)
        j0_sum += c * A
        y_sum += c * (A * L - B)
        if m > 0:
            y_sum += (-c * harmonic) * A   # d_m = -c_m * H_m

    return 2.0 * (j0_sum - 1j * (2.0 / np.pi) * y_sum)


def self_coupling_integral(
    k, half_len, order: int = DEFAULT_QUAD_ORDER, n_terms: int = 60
) -> complex:
    """段内自耦合的核积分 ``I = ∫_{-a}^{a} H_0^{(2)}(k|t|) dt``(**主路径**)。

    x = k*a <= 0.5 时直接走级数解析积分;x > 0.5 时按 ``delta = 0.5/k`` 分段:
    内段用级数(仍然精确),外段两端被积函数光滑,用 Gauss-Legendre。

    Parameters
    ----------
    k : float
        波数。
    half_len : float
        半段长 a > 0。
    order : int
        外段 Gauss-Legendre 节点数。
    n_terms : int
        级数项数。

    Returns
    -------
    complex

    Notes
    -----
    自耦合的矩阵元是 ``Z_mm = -(j/4) * I``(见 ``mom.py``)。
    """
    k = float(k)
    a = float(half_len)
    if not a > 0.0:
        raise ValueError(f"half_len 必须为正,收到 {half_len!r}")
    if not k > 0.0:
        raise ValueError(f"k 必须为正,收到 {k!r}")

    delta = min(a, SMALL_THRESHOLD / k)
    total = series_self_integral(k, delta, n_terms=n_terms)

    if delta < a:
        # 被积函数关于 t = 0 是偶函数,外段有 [delta, a] 与 [-a, -delta] 两条,
        # 只积正半轴、结果乘 2(漏掉这个 2 会让 ka > 0.5 的答案系统性偏小一半外段)
        nodes, weights = gauss_legendre(order)
        mid = 0.5 * (delta + a)
        half = 0.5 * (a - delta)
        t = mid + half * nodes
        total += 2.0 * half * complex(np.sum(weights * hankel0_2(k * t)))

    return total


# ======================================================================
# 错误做法(仅用于自检里量化危害,不要在生产路径调用)
# ======================================================================
def direct_gauss(k, half_len, order: int = DEFAULT_QUAD_ORDER) -> complex:
    """**错误做法**:对带奇异性的核直接上 Gauss-Legendre(不拆奇点)。"""
    nodes, weights = gauss_legendre(order)
    a = float(half_len)
    t = a * nodes
    return a * complex(np.sum(weights * hankel0_2(float(k) * np.abs(t))))


def naive_self_coupling(k, half_len, order: int = DEFAULT_QUAD_ORDER) -> complex:
    """**错误做法**:把余项当常数 1,只用对数主部闭式。

        I_naive = log_term_integral(k, a) + 2a * 1
    """
    del order
    return complex(log_term_integral(k, half_len)) + 2.0 * float(half_len) * (1.0 + 0.0j)
