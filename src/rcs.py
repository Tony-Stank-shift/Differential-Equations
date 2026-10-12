# -*- coding: utf-8 -*-
"""二维 TM 散射的远场量:远场幅度 f(φ) 与散射宽度 σ_2D(φ)。

============================================================================
全局约定(与 green.py / quad.py / mom.py 一致,见 DECISIONS D6/D11,不得单独改)
============================================================================
    e^{+jωt} 时间因子
    出射波 e^{-jkρ}
    (∇² + k²)G = +δ(r - r')
    G(r, r') = (j/4) H₀⁽²⁾(kR)
    u_inc = exp(+jk(x cosθ_i + y sinθ_i))
    装配式 Z_mn = +(k/4)∫H₀⁽²⁾;电流重建场 E_z^sca = −(k/4)Σ_n J_n ∫H₀⁽²⁾

============================================================================
远场幅度定义(D19 —— 本模块锁定)
============================================================================
**唯一定义**(不含任何自由度):令 ρ → ∞,

    f(φ) := lim_{ρ→∞} √ρ · e^{+jkρ} · E_z^sca(ρ, φ)

即把出射波 e^{-jkρ}/√ρ 这个径向因子从场里除掉。由 E_z^sca ~ f(φ)e^{-jkρ}/√ρ
可直接看出这与"分离载波"是同一件事。

⚠️ 从 H₀⁽²⁾(x) 的渐近 √(2/(πx))·e^{-j(x-π/4)} 出发,柱面波的径向因子精确为

    C(k,ρ) = √(2/(πkρ)) · e^{-j(kρ-π/4)}

但**注意换算关系里还有一个常数 G**,因为 `C = G·e^{-jkρ}/√ρ`:

    G := √(2/(πk))·e^{+jπ/4}

所以正确的换算是

    f(φ) = G · lim_{ρ→∞} E_z^sca(ρ,φ) / C(k,ρ)          ← 这个 G 不能漏
    反过来:E_z^sca / C  →  f/G

⚠️ 2026-10-12 第二轮复核员实测:把 `lim E/C` **直接**当 f,与 f_M 的 rel-L2 是
**0.8935**(= |1/G − 1|,k=1);补上 G 后是 **1.35e-05**。按错的写,σ 会差 `1/|G|² = πk/2`
倍(k=1 时高估 57%)。—— 这恰好就是下面这句警告本身犯的错,写文档时尤其要小心。

本模块的 ``carrier`` 就是这个 C,``_amplitude_factor`` 就是这个 G。
**不要**写成裸的 e^{-jkρ}/√ρ:那会漏掉 G,f 整体差常数倍、σ 差常数倍,
而所有"自比"判据仍然全绿(判据 A 专门守这条)。

⚠️⚠️ D19 最重要的那条(2026-10-12 逐模态实测定位,接续必读):
**Mie 侧的角核是 `(−1)ⁿ·C_n`,既不是 `C_n`,也不是 `jⁿ·C_n`。**

    (−1)ⁿ·C_n     ← 正确。实测 A_n/C_n/(−1)ⁿ 在 ka=0.5/1/2/5/10 上是常数,
                    等于 √(2/(πk))e^{+jπ/4},相位散布 ≤ 4.8e-3(= MoM 离散误差)
    jⁿ·C_n        ← 错:相对正确值**旋转 ∓π/2**(方向图整体转 90°)
    C_n           ← 错:相对正确值**旋转 π**(方向图整体转 180°)

    ▸ 旋转角怎么来的:Σ c_n e^{jnφ} 里把 c_n 乘 e^{jnα} 等于把方向图平移 α。
      jⁿ = e^{jnπ/2}、C_n 相对正确值差因子 e^{jnπ} = (−1)ⁿ ⇒ 分别偏 90°、180°。
      (2026-10-12 复核员指出本文件曾把 C_n 那条写成"少 π/2",那是错的;两个错版本
       都是逐模态 O(1) 错,但旋转角不同。)

推导(n 步都不能砍,尤其第 2 步):

    ① 入射  e^{+jkx} = Σ_n jⁿ J_n(kρ) e^{jnφ}
    ② 边界条件 E_inc + E_sca = 0 (ρ=a) ⇒ a_n = −jⁿ J_n(ka)/H_n⁽²⁾(ka) = jⁿ C_n
    ③ H_n⁽²⁾(kρ) ≈ C(k,ρ)·e^{+jnπ/2} = C(k,ρ)·jⁿ   ← jⁿ 在渐近相位里
    ④ 角核 = a_n · (③ 的 jⁿ) = j^{2n} C_n = (−1)ⁿ C_n

最容易漏的是 ③:那个 jⁿ 不在系数里、而在 Hankel 的渐近相位里,于是写
"Σ_n C_n e^{jnφ}"(漏一次 jⁿ)或"Σ_n jⁿC_n e^{jnφ}"(用错成只乘一次)都看似合理。
判别捷径:提取真实角谱 A_n 后看 **A_n/(C_n·(−1)ⁿ)** 是否为常数 ——
用 jⁿ 或 1 去除都会留下 O(1) 的模态相关相位。
自检见 ``tests/check_rcs.py`` 判据 M(含把 jⁿ 塞回去的孪生反例 K1)。

============================================================================
两条独立路径给出的 f(φ)
============================================================================
配方(由渐近 G 直接得到;本模块实测 N=640、ka=1 时 max|f_J/f_M − 1| = 2.3e-05):

    f_J(φ) = √(2/(πk))·e^{+jπ/4}·[−(k/4)·Σ_m J_m L_m e^{+jk(x_m cosφ + y_m sinφ)}]

    ▸ 推导:H₀⁽²⁾(k|ρx̂ - r'|) ≈ √(2/(πkρ))e^{-j(kρ-π/4)}·e^{+jk x̂·r'},
      E_z^sca = −(k/4)Σ_m J_m ∫H₀⁽²⁾ dl' ⇒ f_J = lim √ρ e^{+jkρ}E_sca 即上式。

Mie 侧(**解析闭式,不选虚拟半径**):

    f_M(φ) = √(2/(πk))·e^{+jπ/4}·Σ_n (−1)ⁿ · C_n · e^{jn(φ-θ_i)}
    C_n = -J_n(ka)/H_n⁽²⁾(ka)

两条路径共用一个前置因子 √(2/(πk))e^{+jπ/4}(它来自载波归一化),
所以 f_J/f_M 中它被约掉 —— 这也是判据 M"只查常数因子"的原因。

============================================================================
散射宽度(二维"RCS")
============================================================================
本模块采用**每弧度**约定(与"总宽度 = ∫σ dφ"自洽):

    σ_2D(φ) = lim_{ρ→∞} ρ·|E_z^sca(ρ,φ)|² / |E_z^inc|²            [量纲:长度]

理由(2026-10-12 复核员用真实空间能流钉住,判据 S 守):二维散射功率(单位长度)
P = ρ∫|E_sca|²dφ,入射强度 = |E_inc|²,总散射宽度 σ_tot = P/强度 = ρ∫|E_sca|²dφ。
要让 σ_tot = ∫σ_2D dφ,σ_2D 必须取上式 —— **不带 2π**。

⚠️ 另有一个"回波宽度(echo width)"约定
    σ_echo(φ) = 2π·ρ·|E_sca|²/|E_inc|²
它比本模块的 σ_2D 大 2π;用那个约定时总宽度必须写 σ_tot = (1/2π)∫σ_echo dφ。
(⚠️ 该约定常见于二维散射的教材,但**本文件未逐条核对具体出处** —— 若论文要引用它,
须先查证原始文献再写。这里列出来只为说明"2π 记在哪"这件事本身容易混。)
两个约定不能混用。2026-10-12 复核员抓到本文件曾把 `2πρ` 的定义与不带 2π 的实现
并列写在一起(差正好 2π)。

⚠️ 这里有个容易说错的细节(**第二轮复核员用反例纠正了我,原话"判据 E 完全看不见 2π"
是错的**):E1 的右侧是**独立的物理闭式** `(4/k)Σ|C_n|²`,不是"写在实现约定里的同义量"。
实测:只把 `scattering_width` 乘 2π(σ 与闭式不一致)⇒ **E1 立刻红**
(`|∫−闭式|/闭式 = 5.283185 = 2π−1`)。真正看不见的是**一致换约定**:把 σ_2D 与闭式
参考**一起**乘 2π,E1/E2 都绿,只剩 **S**(真实空间能流)红(实测 0.8408)。
准确说法:**E 守"σ 与物理闭式一致",S 才守"约定本身选对了"。**

代入 f:ρ 约掉 ⇒ **σ_2D(φ) = |f(φ)|²/|A|²**。

能量恒等式(判据 E 守):
    ∫_0^{2π} σ_2D(φ) dφ = (4/k)·Σ_n |C_n|²

⚠️ 这个等式对 f_M 来说是 **Parseval 恒等式,不是独立物理证据**(2026-10-12 复核员
指出):f_M 本来就是 G·Σ(−1)ⁿC_n e^{jnφ},Parseval 直接给出
2π|G|²Σ|C_n|² = (4/k)Σ|C_n|² —— 代数上恒等(数值残差 ~4.5e-16,判据 E1 实测 1.5e-16,
**不是字面意义的 0**)。所以 E1 只守得住
**归一化**(它确实抓得住 K3),对**逐模态相位结构性失明** —— 把角核换成 C_n 或
jⁿC_n,E1 仍然全绿(这正是本模块自认最重要的那件事)。
真正独立的是 **E2**(f_J 来自 MoM 电流,不共享 C_n)与 **S**(真实空间能流积分)。

============================================================================
待办 D20:二维光学定理的归一化尚未锁定
============================================================================
本模块**不提供**消光宽度/光学定理。原因见 DECISIONS D20:
已实测的恒等式是

    Re(Σ_n C_n) = −Σ_n |C_n|²        (8 个 ka 上 ≤ 1.8e-14)

而"σ_ext = (系数)·Im f(前向)"的各种候选都对不上(且对不上得很一致,不是容差问题)。
RCS 交付判据不依赖它,故登记为待办而不是硬凑一个判据。
**不要在 tests/check_rcs.py 里加一条从不红的"光学定理"判据。**
"""
from __future__ import annotations

import numpy as np

try:  # 作为包导入
    from .mom import mie_scattered_field  # noqa: F401  (仅用于 docstring 交叉引用)
except ImportError:  # src/ 在 sys.path 上
    from mom import mie_scattered_field  # noqa: F401

__all__ = [
    "carrier",
    "far_field_amplitude_from_current",
    "mie_far_field_amplitude",
    "scattering_width",
]


def _as_real_array(x, name: str) -> np.ndarray:
    """转成 float 数组;拒绝复数与 NaN/inf。

    复数必须显式拒绝:``np.asarray(complex, dtype=float)`` 只发一个 ComplexWarning
    然后**静默丢掉虚部**(2026-10-12 复核员实测),这是最坏的一类输入错误。
    """
    if np.iscomplexobj(x):
        raise ValueError(f"{name} 不能是复数(会被静默丢虚部),收到 {x!r}")
    arr = np.asarray(x, dtype=float)
    if np.any(~np.isfinite(arr)):
        raise ValueError(f"{name} 必须是有限数,收到 {x!r}")
    return arr


def _require_finite(result, what: str):
    """把上溢/下溢造成的**静默** NaN/inf 变成 ValueError。

    2026-10-12 复核员实测的出口:``carrier(1e-320, 1.0)`` → ``inf``;
    ``scattering_width(k, f, 1e-200)`` → ``inf``(｜A｜² 下溢为 0);
    ``mie_far_field_amplitude(k=1, radius=1e-300)`` → ``nan``。
    这些都不该静默返回 —— 下游会拿去当真值用。
    """
    if not np.all(np.isfinite(np.asarray(result))):
        raise ValueError(
            f"{what} 返回了非有限值(NaN/inf)。极端小/大的 k、rho、radius 会让 "
            f"√(2/(πk)) 之类的因子在本机双精度下上溢或下溢;请检查输入量级。"
        )
    return result


def _as_real_scalar(x, name: str, *, positive: bool = False) -> float:
    """把标量转成 Python float,并拒绝三类坏输入。

    * **复数** —— ``np.isfinite(1+0j)`` 返回 True,随后的 ``<=`` 比较会抛
      **TypeError** 而不是 ValueError(违反本模块的输入契约);``theta_i=1j`` 更糟,
      会**静默返回有限垃圾**(2026-10-12 第二轮复核员实测)。
    * NaN / inf。
    * ``positive=True`` 时的非正数。
    """
    if np.iscomplexobj(x):
        raise ValueError(f"{name} 不能是复数,收到 {x!r}")
    try:
        v = float(x)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是实数标量,收到 {x!r}") from exc
    if not np.isfinite(v):
        raise ValueError(f"{name} 必须是有限数,收到 {x!r}")
    if positive and v <= 0.0:
        raise ValueError(f"{name} 必须是正的有限数,收到 {x!r}")
    return v


def carrier(k: float, rho: float):
    """径向载波因子 ``C(k,ρ) = √(2/(πkρ))·e^{−j(kρ−π/4)}``。

    H₀⁽²⁾(kρ) 的领头渐近。它是 ``f(φ)`` 与 ``E_z^sca`` 之间的换算关系
    (见模块 docstring 的"唯一定义")。

    ⚠️ 本函数过去**零覆盖**(2026-10-12 复核员:把它换成永远返回 42+0j 的桩函数,
    全套自检仍然 29 PASS/0 FAIL)。现已由 ``tests/check_rcs.py`` 判据 A 守护。
    """
    k = _as_real_scalar(k, "k", positive=True)
    rho = _as_real_array(rho, "rho")
    if np.any(rho <= 0.0):
        raise ValueError(f"rho 必须为正的有限数,收到 {rho!r}")
    out = np.sqrt(2.0 / (np.pi * k * rho)) * np.exp(-1j * (k * rho - 0.25 * np.pi))
    return _require_finite(out, "carrier")


def _amplitude_factor(k: float) -> complex:
    """两条路径共用的远场前置因子 ``√(2/(πk))·e^{+jπ/4}``。

    它来自 H₀⁽²⁾ 的渐近常数;f_J/f_M 中会被约掉,但**σ = ∫|f|²dφ 的正确性依赖它的模**
    (|factor|² = 2/(πk)),所以不能省。
    """
    return np.sqrt(2.0 / (np.pi * k)) * np.exp(1j * 0.25 * np.pi)


def far_field_amplitude_from_current(k: float, geom: dict, J: np.ndarray,
                                     phi) -> np.ndarray:
    """由表面电流给出远场幅度 ``f_J(φ)``。

        f_J(φ) = √(2/(πk))·e^{+jπ/4}·[−(k/4)·Σ_m J_m L_m e^{+jk(x_m cosφ + y_m sinφ)}]

    Parameters
    ----------
    k : float
        波数 > 0。
    geom : dict
        ``mom.circle_geometry`` 的返回值(需 ``mid (N,2)`` 与 ``length (N,)``)。
    J : array_like (N,) complex
        表面电流(η₀J_z 单位),``mom.surface_current`` 的输出。
    phi : array_like
        观察角(弧度),标量或数组。

    Returns
    -------
    ndarray (complex)
        与 ``phi`` 同形状的远场幅度。

    Notes
    -----
    成本 O(N·N_φ),一次矩阵-向量乘。**不做任何与 ρ 有关的计算** —— 它直接给出
    远场极限,调用方不需要选 ρ;"有限 ρ 的渐近够不够"另由判据 R 验。

    ⚠️ 相位 e^{+jk x̂·r'} 的符号来自 D11 的 e^{+jk·r} 入射约定 + 出射波 e^{-jkρ};
    写反会让 f 整体共轭,判据 M/E 立刻红。
    """
    k = _as_real_scalar(k, "k", positive=True)
    if not isinstance(geom, dict) or "mid" not in geom or "length" not in geom:
        raise ValueError(
            "geom 必须是含 'mid' 与 'length' 两个键的 dict(mom.circle_geometry 的返回值);"
            f"收到 {type(geom).__name__},键 = {sorted(geom) if isinstance(geom, dict) else 'n/a'}"
        )
    mid = _as_real_array(geom["mid"], "geom['mid']")
    if mid.ndim != 2 or mid.shape[1] != 2:
        raise ValueError(f"geom['mid'] 必须是 (N,2) 数组,收到 {mid.shape!r}")
    length = _as_real_array(geom["length"], "geom['length']").reshape(-1)
    if np.any(length <= 0.0):
        raise ValueError(
            f"geom['length'] 必须全为正(段长);收到最小值 {length.min()!r}。"
            "负段长会让 f 整体变号而不报错 —— 那是静默错误。"
        )
    J = np.asarray(J, dtype=complex).reshape(-1)
    if J.size != mid.shape[0] or length.size != mid.shape[0]:
        raise ValueError(
            f"J / length / geom['mid'] 长度不一致: {J.size} / {length.size} / {mid.shape[0]}"
        )
    if np.any(~np.isfinite(J)):
        raise ValueError("J 含非有限值(通常是求解失败或 Mie 截断溢出)")

    phi_arr = np.atleast_1d(_as_real_array(phi, "phi"))

    # proj[p, n] = x_n cosφ_p + y_n sinφ_p
    proj = np.vstack([np.cos(phi_arr), np.sin(phi_arr)]).T @ mid.T   # (P, N)
    S = (np.exp(1j * k * proj) * (length[None, :] * J[None, :])).sum(axis=1)
    f = _require_finite(_amplitude_factor(k) * (-0.25 * k) * S, "far_field_amplitude_from_current")
    if np.ndim(phi) == 0:
        return np.reshape(f, ())
    return f.reshape(phi_arr.shape)


def mie_far_field_amplitude(k: float, radius: float, phi, theta_i: float = 0.0,
                            amplitude: float = 1.0) -> np.ndarray:
    """Mie 级数侧的远场幅度(解析真值,不选虚拟半径)。

        f_M(φ) = √(2/(πk))·e^{+jπ/4}·Σ_n (−1)ⁿ·C_n·e^{jn(φ−θ_i)}
        C_n = −J_n(ka)/H_n⁽²⁾(ka)

    ⚠️ 角核是 ``(−1)ⁿ·C_n``。写成 ``C_n``(方向图转 180°)或 ``jⁿ·C_n``(转 90°)
    都能"看起来对" —— 见模块 docstring 的四步推导。

    ⚠️ 这个函数与 ``mom.mie_scattered_field`` **不是互相独立的真值**:两者都从同一个
    Dirichlet 边界条件定系数(C_n 相同)。独立证据是 ``far_field_amplitude_from_current``
    (来自 MoM 电流)与判据 E(能量守恒)。

    Parameters
    ----------
    k, radius : float
        波数与圆柱半径(ka > 0)。
    phi : array_like
        观察角(弧度),标量或数组。
    theta_i : float
        入射角(默认 0)。
    amplitude : float
        入射幅度(默认 1);f 随它线性缩放,由 ``scattering_width`` 负责除掉 |A|²。
    """
    from scipy.special import hankel2, jv

    k = _as_real_scalar(k, "k", positive=True)
    radius = _as_real_scalar(radius, "radius", positive=True)
    theta_i = _as_real_scalar(theta_i, "theta_i")
    amplitude = _as_real_scalar(amplitude, "amplitude")

    ka = k * radius
    n_terms = int(np.ceil(ka + 6.0 * ka ** (1.0 / 3.0))) + 60
    ns = np.arange(-n_terms, n_terms + 1)
    C = -jv(ns, ka) / hankel2(ns, ka)          # 标准 Mie 系数(不带 (-1)^n)
    weight = ((-1.0) ** ns) * C                # 远场角核 = (-1)^n C_n,见 docstring

    phi_arr = np.atleast_1d(_as_real_array(phi, "phi"))
    phase = np.exp(1j * np.outer(phi_arr - theta_i, ns))
    out = _require_finite(amplitude * _amplitude_factor(k) * (phase @ weight),
                          "mie_far_field_amplitude")
    if np.ndim(phi) == 0:
        return np.reshape(out, ())
    return out.reshape(phi_arr.shape)


def scattering_width(k: float, f_phi, amplitude: float = 1.0) -> np.ndarray:
    """二维散射宽度 ``σ_2D(φ) = |f(φ)|²/|A|²``(量纲:长度,**每弧度**约定)。

    由 ``σ_2D(φ) = lim ρ·|E_z^sca|²/|E_z^inc|²`` 与 ``f = lim √ρ e^{+jkρ}E_sca``
    直接约掉 ρ 得到(见模块 docstring;此处**不带 2π**,所以 ∫σ_2D dφ 就是总散射宽度)。

    ⚠️ 本函数过去不校验 ``f_phi``:NaN/inf 会静默传播,``amplitude=1e-200`` 会因
    |A|² 下溢而返回 inf —— 两者都直接违反它自己的契约(2026-10-12 复核员实测)。
    """
    k = _as_real_scalar(k, "k", positive=True)
    amplitude = _as_real_scalar(amplitude, "amplitude")
    if amplitude == 0.0:
        raise ValueError(f"amplitude 必须是非零有限数,收到 {amplitude!r}")
    f = np.asarray(f_phi, dtype=complex)
    if np.any(~np.isfinite(f)):
        raise ValueError("f_phi 含非有限值(NaN/inf);上游求解或 Mie 截断可能已失效")
    return _require_finite(np.abs(f) ** 2 / (abs(amplitude) ** 2), "scattering_width")
