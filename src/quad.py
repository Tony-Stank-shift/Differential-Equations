"""奇异积分处理 —— 矩量法自耦合项的核积分。

奇异从哪来
----------
二维 TM 的格林函数是 ``G = (j/4) H_0^{(2)}(kR)``。R -> 0 时

    H_0^{(2)}(x) = 1 - (2j/pi)[ ln(x/2) + gamma ] + O(x^2 ln x)

带对数发散。矩量法里(符号以 D6/D11 锁定的一套为准,`mom.py` 才是权威形式)

    Z_mn = +(k/4) ∫_{段 n} H_0^{(2)}(k |r_m - r'|) dl'

当 m == n(自耦合项)时 r' 会扫过场点 r_m,核取到 R = 0。此时**任何**普通数值
求积(Gauss-Legendre、Simpson、梯形)都会失败 —— 被积函数本身发散。

⚠️ 先说清"主路径"在**本文件**里指什么(2026-10-12 更正,原措辞会把读者引向相反的理解)
------------------------------------------------------------------------------------
D10 里写的"主路径"是**1.2 时点的决策结果**:把 1.2 当时的默认路径(分段)
换成了 Struve 闭式。但从 0→1.3 起,`self_coupling_integral(method="auto")`
才是实际入口,而它的自动规则是

    k*a <= SMALL_THRESHOLD(0.5)  ->  级数 ``series_self_integral``
    k*a >  0.5                   ->  闭式 ``closed_self_integral``

**在真实 MoM 网格下 ``k*a`` 恒为 1e-2 量级,所以 ``auto`` 恒选级数路径,
闭式在本项目里从未被 ``auto`` 选中过。** 实测(k=1):

    ka=0.5  N=64   k*a=2.453e-02  -> series
    ka=5    N=400  k*a=3.927e-02  -> series
    ka=20.5 N=820  k*a=7.854e-02  -> series
    ka=100  N=4000 k*a=7.854e-02  -> series

原因是 `mom.py` 用**弦长** `2R sin(pi/N)` 而不是弧长,故
`k*a = k*R*sin(pi/N) = ka*sin(pi/N) ~= pi*ka/N`;要越过 0.5 需要 `N < 2*pi*ka`
(细分不足两圈,当前输入校验把 N>=3 就挡住了——k=1 时连 N=6 都恰好不越界)。

两者**数值都正确**(级数对 mpmath 40 位真值逐位相同;闭式全域 1e-16~1e-14),
所以这是**文档口径 bug,不是代码 bug**。闭式的剩余价值是三条:①k*a > 0.5
的兜底(例如大 k 配粗网格);②独立于幂级数的交叉验证锚点(两条路互校);
③`check_mom.py` 的 J1/J2 黄金值直接钉在它上面。**论文里引用本题时,以
``self_coupling_integral`` 的自动规则为准,不要写"默认走 Struve 闭式"。**

Struve 闭式(``closed_self_integral``)—— k*a > 0.5 时的兜底实现
----------------------------------------------------------------
    I(k, a) = ∫_{-a}^{a} H_0^{(2)}(k|t|) dt = (2/k) * F(k*a)

其中

    F(z) = ∫_0^z H_0^{(2)}(u) du
         = z H_0^{(2)}(z) + (pi z / 2) [ H_1^{(2)}(z) * H_0(z) - H_0^{(2)}(z) * H_1(z) ]

``H_n`` 是 Struve 函数(``scipy.special.struve``)。来源是洛默(Lommel)函数恒等式

    d/dz [ z J_1(z) v_0(z) - z J_0(z) v_1(z) ] = z J_0(z)

取 H^{(2)} 作第二解时结构常数为 pi/2,积分常数由 ``F(0) = 0`` 定。

**O(1)、一段式、无分段、无级数、无求积。** 它是 ``self_coupling_integral``
在 ``k*a > 0.5`` 时的分支(见上方 ⚠️:真实网格下该分支不会被触发)。
双精度实测(z = k*a,基准是 mpmath 高精度真值):

    z = 1e-8 -> 0        z = 1    -> 3.1e-16    z = 100  -> 1.7e-15
    z = 0.5  -> 1.5e-16  z = 5    -> 2.6e-16    z = 1e3  -> 7.3e-15
    z = 1e4  -> 2.4e-14

(z >= 1e3 起精度按 ``sqrt(z) * eps`` 退化,来源是两项之间的相消;z = 1e4 仍远好于 1e-12。)

⚠️ **不能只用贝塞尔函数表出**:由 ``z H_1^{(2)}(z) = ∫_0^z u H_0^{(2)}(u) du + 2j/pi``
可知被积函数**多乘了一个 u**,所以 ``I(k,a) = 2a H_1^{(2)}(ka)`` **是错的**
(实测 k=2, a=0.3 相对差 5.3e-1;k=10, a=0.5 差 3.3e+0)。需要 Struve 函数;
它本身可用 1F2 / Meijer-G 表示,scipy 已实现。

``auto`` 下的实际工作路径:级数逐项解析积分(``series_self_integral``)
----------------------------------------------------------------------
``x = k*a <= 0.5`` 时把 H_0^{(2)} 的级数逐项解析积分(推导见函数 docstring)。
**这就是 ``auto`` 在全部真实 MoM 算例里选中的那条路**(理由见上方 ⚠️)。
与闭式**互相独立**(一条走幂级数,一条走特殊函数),已验证到与 mpmath 40 位
真值逐位相同。它同时承担两个角色:(a) 小 ka 的精确回归锚点;
(b) 与闭式互为交叉验证。

备用路径:拆主部 + 数值求积(``regular_part`` / ``analytic_part_integral``)
----------------------------------------------------------------------
    H_0^{(2)}(x) = S(x) + x^2[c0 + c1 ln(x/2)] + R2(x)
    S(x)  = -(2j/pi)[ ln(x/2) + gamma ]
    c1    = j/(2pi)
    c0    = -1/4 - j(1-gamma)/(2pi)
    R2(x) = O(x^4 ln x)     <- 光滑,数值求积可用

这条路在本模块里**不是**自耦合项的默认实现,保留它是为了 1.3 的近奇异积分:
场点靠近但**不在**段上时,核在段上没有零点,``|t|`` 那种对称性不存在,
上面的闭式也就用不上,此时"拆解析主部 + 数值积余项"才是通用做法。

⚠️ 它的精度**只在 ``k*a <= 0.5`` 时保证 1e-12**。实测 k*a = 1 -> 1.8e-11,
k*a = 5 -> **8.6e-8**。原因是 ``R2 - 1 = O(t^4 ln|t|)`` 在**区间内部** t = 0 处
有分支点(不是端点奇性),Gauss-Legendre 只能代数收敛(实测 n^-5:order 64 -> 8.6e-8,
512 -> 2.7e-12,1024 -> 3.4e-13)。k*a 较大时请提高 ``order``,或改用主路径。

⚠️ 绝不要做的事
--------------
**不要把 R(x) = 1 + O(x^2 ln x) 里的余项当常数 1 解析积。** ``R -> 1`` 是渐近
结论且收敛不快:两阶级数截断误差 ``|H_0^{(2)}(x) - S(x) - 1|`` 在 x = 0.5 时
约 **9.5e-2**,直接当 1 会让自耦合项错近 10%。``naive_self_coupling`` 保留了这个
错误做法,只为在自检里量化它的危害。
"""
from __future__ import annotations

import numpy as np
from scipy.special import hankel2 as _hankel2
from scipy.special import struve as _struve

try:  # 作为包导入(import src.quad)
    from .green import EULER_GAMMA, SMALL_THRESHOLD, hankel0_2, log_singular_part
except ImportError:  # src/ 在 sys.path 上(import quad)
    from green import EULER_GAMMA, SMALL_THRESHOLD, hankel0_2, log_singular_part

__all__ = [
    "DEFAULT_QUAD_ORDER",
    "DEFAULT_REGULAR_ORDER",
    "C0",
    "C1",
    "gauss_legendre",
    "hankel0_antiderivative",
    "closed_self_integral",
    "log_term_integral",
    "poly2_term_integral",
    "analytic_part_integral",
    "regular_part",
    "series_self_integral",
    "split_self_integral",
    "self_coupling_integral",
    "direct_gauss",
    "naive_self_coupling",
]

#: Gauss-Legendre 默认节点数。用于 ``split_self_integral`` 的外段、备用路径
#: (``direct_gauss`` / ``regular_part`` 的调用方)以及 ``mom.assemble_impedance``
#: 的非对角元。**自耦合主路径不需要它**(级数是递推,闭式是特殊函数)。
#:
#: ⚠️ 为什么是 16 而不是更大的数(2026-10-12 由实测标定,原值 64):
#: 在这套圆几何 + 弦长段上,**非对角元的求积在 Q >= 12 就完全饱和**,Q 再大
#: 结果一位都不变。四个算例(k*a 跨越 2.45e-02 ~ 7.85e-01)实测:
#:
#:     k*a         Q=4              Q=8              Q=12 ... Q=128
#:     2.45e-02    4.772529484e-04  4.773633271e-04  全部 4.773633282e-04
#:     3.93e-02    1.961659483e-04  1.963859227e-04  全部 1.963859257e-04
#:     7.854e-02   1.467122746e-03  1.475212701e-03  全部 1.475212810e-03
#:     7.854e-01   5.867658676e-02  5.868068980e-02  全部 5.868068986e-02
#:
#: 取 16 而非 12 是留 4/3 倍余量,防止换几何(如细长段、大 k 配粗网格)时
#: 提前失准。装配成本是 Theta(N^2 * Q)(实测约 290 ns/(N^2*Q)),故 64 -> 16
#: 直接省约 4 倍装配时间。**这是"太慢先于装不下"的主要改善点**:N=5000、Q=64
#: 需约 464 s,Q=16 降到约 116 s。
#:
#: ⚠️ 适用边界:上表是在**本项目的圆几何**上测的。若换用极端不等长段、
#: 或 k*a 远超 1 的配置,请重跑一次 Q 扫掠再定默认值(D10 的纪律:给数值函数
#: 定参数,先扫参数空间找失效边界)。
DEFAULT_QUAD_ORDER = 16

#: **备用奇异路径**上 ``regular_part`` 的 Gauss-Legendre 节点数,默认 **64**。
#:
#: ⚠️ 为什么不复用 ``DEFAULT_QUAD_ORDER``(2026-10-12 由回归实测分开的):
#: 这两个默认值服务的是**两套完全不同的判据**,共用会让其中一套静默失效。
#: 把备用路径也降到 16 的后果是实测的 —— `check_quad` 判据 F 直接变红:
#:
#:     (analytic_part_integral + GL(regular_part)) 在 ka <= 0.5 上:
#:         order=16 -> ka=0.3: 8.798e-11   ka=0.5: 8.058e-10   (tol 1e-12,红)
#:         order=64 -> 最差 8.576e-13                          (tol 1e-12,过)
#:
#: 即"备用路径在 ka <= 0.5 保证 1e-12"这条承诺**是靠 64 点撑住的**;
#: 而 `regular_part` 的余项 `R2 - 1 = O(t^4 ln|t|)` 分支点在区间内部 t = 0,
#: GL 只能代数收敛(实测 n^-5),所以提阶确实有用。两者不可混为一谈:
#:   · 非对角元装配(被积函数在段上光滑)  -> 16 点已饱和,提阶无益
#:   · 备用奇异路径(区间内部有分支点)  -> 需要 64 点以上才够 1e-12
DEFAULT_REGULAR_ORDER = 64

#: 二阶主部的对数系数:R(x) 中 x^2 ln(x/2) 的系数
C1 = 1j / (2.0 * np.pi)

#: 二阶主部的常数系数
C0 = -0.25 - 1j * (1.0 - EULER_GAMMA) / (2.0 * np.pi)


def gauss_legendre(order: int = DEFAULT_QUAD_ORDER):
    """[-1, 1] 上的 Gauss-Legendre 节点与权重。"""
    nodes, weights = np.polynomial.legendre.leggauss(int(order))
    return nodes, weights


# ======================================================================
# 主路径:Struve 闭式
# ======================================================================
def hankel0_antiderivative(z):
    """``F(z) = ∫_0^z H_0^{(2)}(u) du`` 的 Struve 闭式(``k*a > 0.5`` 分支)。

        F(z) = z H_0^{(2)}(z)
               + (pi z / 2) [ H_1^{(2)}(z) * H_0(z) - H_0^{(2)}(z) * H_1(z) ]

    ``H_n`` 为 Struve 函数 ``scipy.special.struve``。

    Parameters
    ----------
    z : float or array_like
        非负实数。z = 0 处返回精确的 0。

    Returns
    -------
    complex or ndarray (complex)

    Notes
    -----
    精度随 z 增长按 ``sqrt(z) * eps`` 退化(两项相消):z = 100 -> 1.7e-15,
    z = 1e3 -> 7.3e-15,z = 1e4 -> 2.4e-14。z 再大时如需严格精度,请用 mpmath
    高精度路径,不要在本函数里加"兜底级数"。

    ⚠️ ``z < 0`` 现在抛 ``ValueError``(2026-10-13 第 11 轮复核):此前正值与负值
    走同一个分支以外的兜底,``hankel0_antiderivative(-5.0)`` **静默返回 0j** ——
    一个"看起来像 F(0)"的值。z = 0 仍精确返回 0。
    """
    z_arr = np.asarray(z, dtype=float)
    if np.any(z_arr < 0.0):
        raise ValueError(
            f"hankel0_antiderivative 只对 z >= 0 有定义(F 是 0 到 z 的积分),"
            f"收到 {z!r}。此前负值会被静默当成 0。"
        )
    scalar = z_arr.ndim == 0
    zz = np.atleast_1d(z_arr)
    out = np.empty(zz.shape, dtype=complex)

    positive = zz > 0.0
    if np.any(positive):
        q = zz[positive]
        h0 = _hankel2(0, q)
        h1 = _hankel2(1, q)
        s0 = _struve(0, q)
        s1 = _struve(1, q)
        out[positive] = q * h0 + (np.pi * q / 2.0) * (h1 * s0 - h0 * s1)
    if np.any(~positive):
        out[~positive] = 0.0 + 0.0j  # F(0) = 0;负 z 无定义,也归到这里(见 docstring)

    if scalar:
        return complex(out.reshape(-1)[0])
    return out.reshape(z_arr.shape)


def closed_self_integral(k, half_len):
    """闭式 ``I(k, a) = ∫_{-a}^{a} H_0^{(2)}(k|t|) dt = (2/k) * F(k*a)``(**主路径**)。

    无分段、无求积、无级数;对任意 ``k*a > 0`` 有效(精度见 ``hankel0_antiderivative``)。

    Parameters
    ----------
    k : float
        波数 > 0。
    half_len : float or array_like
        半段长 a > 0。

    Returns
    -------
    complex or ndarray (complex)

    Notes
    -----
    自耦合的矩阵元是 ``Z_mm = +(k/4) * I``(符号以 D6/D11 与 ``mom.py`` 为准)。
    """
    k = float(k)
    if not k > 0.0:
        raise ValueError(f"k 必须为正,收到 {k!r}")
    a_arr = np.asarray(half_len, dtype=float)
    if np.any(a_arr <= 0.0):
        # 2026-10-13 第 11 轮复核实测:此前不校验 half_len,closed_self_integral(1.0,-0.5)
        # **静默返回 0j** —— 负/零段长会把整个对角元静默置零(Z 全错而不报错)。
        # 统一入口 self_coupling_integral 一直是抛 ValueError 的,这里与它对齐。
        raise ValueError(f"half_len 必须为正,收到 {half_len!r}")
    return (2.0 / k) * hankel0_antiderivative(k * a_arr)


# ======================================================================
# 解析主部(备用路径;1.3 的近奇异积分会用)
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
    """``∫_{-a}^{a} [ S + 二阶主部 ] dt`` 的闭式和(备用路径用)。

    ⚠️ 它本身精确到机器精度(z = 1 时 1.7e-16),但配套的 ``regular_part`` 求积
    在 ``k*a > 0.5`` 时精度不足 —— 整条备用路径的误差由 ``regular_part`` 决定
    (k*a = 5 时合成误差 8.6e-8)。详见模块 docstring。
    """
    return log_term_integral(k, half_len) + poly2_term_integral(k, half_len)


def regular_part(k, t):
    """余项 ``R2(k|t|) = H_0^{(2)}(k|t|) - S - (k|t|)^2[C0 + C1 ln(k|t|/2)]``。

    它满足 ``R2(0) = 1`` 且 ``R2(x) - 1 = O(x^4 ln x)`` —— **正是这一项才是光滑到
    可以数值求积的**(只减对数主部时,``R - 1 = O(x^2 ln x)``,40 点以下的
    Gauss-Legendre 会明显不够)。

    ⚠️ ``R2 - 1 = O(t^4 ln|t|)`` 的分支点在**区间内部** t = 0,不是端点,
    所以 GL 只能代数收敛(n^-5),不是谱收敛。``k*a <= 0.5`` 时默认 64 点足够
    (见 ``DEFAULT_REGULAR_ORDER``;注意**不是** ``DEFAULT_QUAD_ORDER`` 的 16 ——
    装配用的那次求积在 16 点已饱和,但这一次因区间内部有分支点,16 点只有 8e-10);
    k*a = 5 需要 512 点以上。

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
# 对照路径:级数逐项解析积分(与主路径互相独立)
# ======================================================================
def series_self_integral(k, half_len, n_terms: int = 60) -> complex:
    """级数逐项解析积分 ``∫_{-a}^{a} H_0^{(2)}(k|t|) dt``(**对照路径**)。

    **只在 x = k*a <= SMALL_THRESHOLD(0.5) 时有效** —— 这是 H_0^{(2)} 级数展开
    的收敛范围(x 稍大时级数项会先变大再变小,精度崩坏)。超出范围抛 ValueError。

    它给出的值与主路径(闭式)在 ka <= 0.5 内一致到 mpmath 40 位真值的机器精度,
    且实现完全独立(一条走特殊函数,一条走幂级数),所以是很好用的交叉验证锚点。

    Parameters
    ----------
    k, half_len : float
        波数与半段长。
    n_terms : int
        级数项数。x <= 0.5 时 60 项远超双精度所需(实测 8 项即到 1e-16)。

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
            "请用 self_coupling_integral(会自动选路径)。"
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


def split_self_integral(
    k, half_len, order: int = DEFAULT_QUAD_ORDER, n_terms: int = 60
) -> complex:
    """分段路径 ``∫_{-a}^{a} = ∫_{-delta}^{delta}(级数) + 2∫_{delta}^{a}(GL)``。

    历史上是 1.2 的主路径,现在降级为**备用**(主路径是闭式,见
    ``closed_self_integral``)。保留原因:它演示了"把奇点圈进解析段、外段交给
    数值求积"这个模式,1.3 的近奇异积分要沿用。

    ⚠️ ``order`` 的**实际默认值是 ``DEFAULT_QUAD_ORDER`` = 16,不是 64**
    (本 docstring 长期误写为"默认 order=64",2026-10-13 第 11 轮复核抓出)。
    这个默认值**精度很有限**,实测相对误差(k*a = 5 / 10 / 25 / 50 / 100):

        order= 8   7.66e-06  2.22e-04  1.73e-01  4.56e+00  6.18e+00
        order=16   1.09e-10  7.74e-08  4.24e-05  3.23e-02  3.19e+00   <-- 默认
        order=32   2.97e-15  1.71e-14  2.33e-09  7.63e-07  1.07e-03
        order=64   5.66e-15  3.54e-15  1.86e-14  1.00e-12  2.82e-09
        order=128  2.42e-14  2.29e-14  1.08e-13  1.88e-13  3.27e-13

    ⇒ **默认值(16)远达不到本模块 1e-12 的标准**;要用这条路径请显式传
    ``order=64`` 以上。``tests/check_quad.py`` 判据 C 现在**把默认值本身也纳入
    断言**(此前它用 ``vals[1:]`` 把 vals[0] 排除在外,等于默认值零覆盖:
    把默认从 16 改成 8 后 check_quad + check_mom 仍然全绿)。
    """
    k = float(k)
    a = float(half_len)
    if not a > 0.0:
        raise ValueError(f"half_len 必须为正,收到 {half_len!r}")
    if not k > 0.0:
        raise ValueError(f"k 必须为正,收到 {k!r}")

    delta = SMALL_THRESHOLD / k
    # a 恰为 0.5/k(或它 1e-15 的邻域)时,float(k*a) 可能略大于 0.5,若直接把 a
    # 交给 series 会抛无谓的 ValueError。用相对容差判定,并把 a 夹回 delta ——
    # 夹掉的相对长度 <= 1e-12,对结果无影响。
    if a <= delta * (1.0 + 1e-12):
        return series_self_integral(k, min(a, delta), n_terms=n_terms)

    total = series_self_integral(k, delta, n_terms=n_terms)

    # 被积函数关于 t = 0 是偶函数,外段有 [delta, a] 与 [-a, -delta] 两条,
    # 只积正半轴、结果乘 2(漏掉这个 2 会让 ka > 0.5 的答案系统性偏小)
    nodes, weights = gauss_legendre(order)
    mid = 0.5 * (delta + a)
    half = 0.5 * (a - delta)
    t = mid + half * nodes
    total += 2.0 * half * complex(np.sum(weights * hankel0_2(k * t)))

    return total


def self_coupling_integral(k, half_len, method: str = "auto") -> complex:
    """段内自耦合的核积分 ``I = ∫_{-a}^{a} H_0^{(2)}(k|t|) dt``(**统一入口**)。

    Parameters
    ----------
    k, half_len : float
        波数 k > 0,半段长 a > 0。
    method : {"auto", "closed", "series", "split"}
        - ``"auto"``(默认):``k*a <= 0.5`` 走级数(与闭式同为精确,且是独立实现),
          否则走闭式。⚠️ **真实 MoM 网格下 ``k*a`` 恒为 1e-2 量级,故这一支实际
          恒选级数**(见模块 docstring 顶端);若要复现 D10 的闭式结论,请显式
          传 ``method="closed"``。
        - ``"closed"``:强制 Struve 闭式(全域有效)。
        - ``"series"``:强制级数(仅 ``k*a <= 0.5``,超出抛 ValueError)。
        - ``"split"``:强制历史分段路径(仅供对照,``k*a > 50`` 后不可信)。

    Returns
    -------
    complex

    Notes
    -----
    自耦合的矩阵元是 ``Z_mm = +(k/4) * I``(符号以 D6/D11 与 ``mom.py`` 为准)。
    """
    k = float(k)
    a = float(half_len)
    if not a > 0.0:
        raise ValueError(f"half_len 必须为正,收到 {half_len!r}")
    if not k > 0.0:
        raise ValueError(f"k 必须为正,收到 {k!r}")

    if method == "auto":
        method = "series" if k * a <= SMALL_THRESHOLD else "closed"

    if method == "closed":
        return complex(closed_self_integral(k, a))
    if method == "series":
        return series_self_integral(k, a)
    if method == "split":
        return split_self_integral(k, a)
    raise ValueError(f"未知 method {method!r},可选 auto / closed / series / split")


# ======================================================================
# 错误做法(仅用于自检里量化危害,不要在生产路径调用)
# ======================================================================
def direct_gauss(k, half_len, order: int = DEFAULT_QUAD_ORDER) -> complex:
    """**错误做法**:对带奇异性的核直接上 Gauss-Legendre(不拆奇点)。

    ⚠️ 它的失效方式是**收敛太慢**,不是"不收敛"(2026-10-13 第 11 轮复核更正 ——
    此前 STATE/文档写"提高阶数无效、永远啃不动",那是错的)。实测误差 ∝ 1/order,
    是代数收敛:

        k*a=0.1  Q=16 1.757e-02  Q=64 4.491e-03  Q=256 1.129e-03  Q=1024 2.827e-04
        k*a=1    Q=16 3.760e-02  Q=64 9.606e-03  Q=256 2.415e-03  Q=1024 6.047e-04
        k*a=1    Q=8192 -> 7.562e-05

    要压到 1e-12 需要约 **1e11** 个点 ⇒ **作为生产路径不可用**这个结论仍然成立,
    但论据是"代数收敛 O(1/n)、点数不现实",不是"提高阶数无效"。
    """
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
