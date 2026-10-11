"""二维 TM 散射的矩量法(EFIE + 脉冲基 + 点匹配)。

============================================================================
全局约定(与 green.py / quad.py 完全一致,见 DECISIONS D6,不得单独更改)
============================================================================
    e^{+jωt} 时间因子
    出射波 e^{-jkρ}
    (∇² + k²)G = +δ(r - r')
    G(r, r') = (j/4) H₀⁽²⁾(kR)
    u_inc = exp(+jk(x cosθ_i + y sinθ_i))

============================================================================
公式
============================================================================
TM 极化(E = E_z ẑ),PEC 边界上总场切向分量 E_z = 0。

⚠️ 这里有一个极易踩错的符号环节,必须与 green.py 的约定连起来看:
    我们锁定 (∇² + k²)G = +δ            (green.py / D6)
    ⇒ (∇² + k²)∫ J G = J,而矢量势要满足 (∇² + k²)A = -μ₀J
    ⇒ A = -μ₀ ∫ J G                    ← 负号在这里,不在别处
    ⇒ E_z^sca = -∂_t A_z = -jω A_z = +jωμ₀ ∫ J_z G
              = +jωμ₀ · (j/4) ∫ J_z H₀⁽²⁾
              = -(ωμ₀/4) ∫ J_z H₀⁽²⁾
              = -(k/4) ∫ (η₀J_z) H₀⁽²⁾

也就是说:教科书上常见的 E_z^sca = -jωμ₀ ∫ J_z G 配的是 (∇²+k²)G = -δ 那一套约定。
两套约定下"由电流求场"的整体符号相反,而物理结论(边界条件、RCS)不变 —— 但**装配式
会变号**,必须与自己的 G 约定保持一致。

把未知量取作 J := η₀ J_z,边界条件 E_z^inc + E_z^sca = 0 化成线性方程组

    Σ_n Z_{mn} J_n = V_m

    Z_{mn} = +(k/4) ∫_{段 n} H₀⁽²⁾(k|r_m - r'|) dl'      (m ≠ n)
    Z_{mm} = +(k/4) I(k, L_m/2),  I(k,a) = ∫_{-a}^{a} H₀⁽²⁾(k|t|) dt
    V_m    = exp(+jk(x_m cosθ_i + y_m sinθ_i))

由 E_z^sca = -(k/4)∫ J H₀⁽²⁾ 与 E_z^inc + E_z^sca = 0 即得 +(k/4)∫ J H₀⁽²⁾ = E_z^inc。
据此解出的 J 就是物理表面电流 J_z = H_φ,与 Mie 级数同号(由 tests/check_mom.py 判据 C 守着)。

自耦合积分 I(k,a) 直接调用 quad.self_coupling_integral(主路径为 Struve 闭式)。

============================================================================
几何离散
============================================================================
圆边界折线化成 N 条直线段:x(φ) = R(cosφ, sinφ) 的 N 等分弦。
基函数 = 段上常数(脉冲),配点 = 段中点。
注意:段长用**弦长** 2R sin(π/N) 而不是弧长 2πR/N,积分元 dl' 沿直线段。

============================================================================
独立真值(D8:判据不得拿实现自身当基准)
============================================================================
二维 PEC 圆柱的 TM 散射有 Mie 级数解析解。由
    E_z^inc = Σ_n j^n J_n(kρ) e^{jn(φ-θ_i)}
    E_z^sca = -Σ_n j^n [J_n(ka)/H_n⁽²⁾(ka)] H_n⁽²⁾(kρ) e^{jn(φ-θ_i)}
边界条件定出系数后,用 J_z = H_φ|_surface =(1/jωμ₀) ∂_ρ E_z^tot|_a 及
柱函数的 Wronskian 关系,得表面电流的闭式:

    η₀ J_z(φ) = (2 / (π k a)) Σ_n [j^n / H_n⁽²⁾(ka)] e^{jn(φ-θ_i)}

这正是本模块输出量的真值(mie_surface_current)。

============================================================================
已知限制 —— EFIE 假共振(不是本实现的 bug,是 EFIE 本身的缺陷)
============================================================================
TM 圆柱的 EFIE 在 J_n(ka) = 0 处有**假共振(spurious interior resonance)**:
此时某个模态的 J_n(ka) 趋近于零,离散矩阵的条件数在该点出现**孤立尖峰**
(不是随 ka 平滑增长的趋势),共振点附近的电流误差被放大一个量级。

实测(半径 1,quad_order 用默认 64,N = min(1600, max(64, 40·ka)) ——
**N 随 ka 变,这一点很关键,见下面 ⚠️**):

    ka       N      cond₂(Z)     说明
    19.50    780    6.96e+01     —
    19.75    790    7.02e+01     —
    19.90    796    7.03e+01     —
    19.99    799    4.52e+02     ← 尖峰**峰值**其实在这里
    20.00    800    3.76e+02     ← 仍高,但不是最高
    20.01    800    1.33e+02     —
    20.25    810    6.95e+01     —
    20.50    820    7.06e+01     —

即条件数在 ka ≈ 19.99~20.00 抬升约 5~6 倍,两侧在 ~0.05 的尺度内回落。

**⚠️ 一段被撤回的错误说明(留档,防重犯)**:早先这里写过"这个尖峰对 N 极其敏感,
第三方用固定 N、只扫 ka 复测会得到单调缓升,把效应洗掉"。**那是错的。**
当时的"固定 N"脚本实际写成 `assemble_impedance(1.0, geom)` 配
`circle_geometry(1.0, n)` —— 即 k = 1、radius = 1 ⇒ **ka ≡ 1**,
波数从头到尾没进入,"扫 ka"只扫了 N(780→820),所以得到
`6.19e+02 → 6.51e+02` 的单调缓升。真做**固定 N = 800 扫 ka**:

    ka       19.50     19.75     19.90     19.99     20.00     20.01     20.25     20.50
    cond₂    71.37     71.13     70.67     451.57    376.03    132.76    68.63     68.89

尖峰**完整存在**,对 N 的取法并不敏感(上表与本节主表逐点吻合)。
**教训:采纳外部复核给出的数字前,必须先复现它的方法。** 这一条错误结论曾被我
直接写进 `DECISIONS.md` 与 `STATE.md`,并计划写进论文"方法局限"一节。

即便如此,**论文里引用本表仍应写明 N 的取法** —— 这是可复现性的基本要求,与上面
那个被撤回的论断无关。

交叉对照:共振点附近正是 |J_n(ka)| 取极小值之处。**在 n ≤ ka 的传播模范围内**:
ka = 20 时最小 |J₁₅(20)| = 8.12e-04,而 ka = 19.75 的最小 |J_n| 是 2.19e-02(n = 11)、
ka = 20.25 是 1.23e-02(n = 6),分别大 **26.9 倍 / 15.1 倍**。
同类现象在 ka = 40(|J₂₉(40)| = 9.64e-04)处重复出现,
说明它是 EFIE 的固有性质而非数值意外。

⚠️ **取模范围必须写明**(0→1.3 复核 P3 指出):上面用的是 n ∈ [0, ⌊ka⌋]。
若改用 n ∈ [0, ⌈ka⌉+5],极小值会落到 n > ka 的**指数衰减尾部**
(ka = 19.75 → 8.01e-03 @ n = 25;ka = 20.25 → 5.60e-03 @ n = 26),
那度量的其实是尾部衰减快慢,不是传播模的近零,复核时会与 n ≤ ka 的取值
差 2.2~2.7 倍而对不上数。两种取值都支持"孤立尖峰"的结论,但必须交代清楚用的是哪种。

⚠️ **J₁₅ 在这里是严格过零,不是"近零"**(0→1.3 复检纠正):
`J₁₅(ka_zero) = 0` 的唯一根在 **ka_zero = 19.994430629816385**(二分到 1e-14),
符号由 `J₁₅(19.98) = +2.105e-03` 翻到 `J₁₅(20.00) = −8.121e-04`。
所以 ka ≈ 20 落在零点右侧 0.0056 处,`|J₁₅|` 不是"取极小"而是**离零点很近**。
先前这里写成"19 < ka < 20.5 内不变号",与事实相反,已更正。

**连带的一个实测后果**(比上面那张 cond 表更该注意):电流误差的最坏点
**不在 ka = 20,而在 ka = 19.995**(离零点最近处),实测 rel_err = **1.82e-02** ——
是 ka = 20 的 `1.475e-03` 的 **12 倍**,也是判据 G 容差 `2e-3` 的 9 倍。
判据 G 已把 19.99 / 19.995 补进采样点,并把这一段单列为"已知劣化区"。


要在共振点附近保持精度,必须改用 CFIE 或 Müller 形式(对 EFIE 加一个
磁场积分方程的组合);单独的 EFIE 在数学上就做不到。本项目的做法是
**声明并绕开**:判据 G 拆成 **G1**(邻域外,守 `2e-3`)与 **G2**(邻域内,
断言"确实劣化"且"误差随 N **单调收敛**")——后一条与 N 的取法无关,
不像绝对上界那样依赖拟合。误差表与所用的 N 都写进正文。

注意:早先这里写的是"ka = 20 处对 tol 2e-3 只剩约 1.36 倍裕度,已补入邻点
19.9 / 19.95 / 20.05"。**那段在更新 docstring 时漏删,已撤**(0→1.3 复检第 5 轮 P1):
"裕度 1.36×"本身是错的(最坏点在 19.995,不是 20),而 19.95 也已不是任何
现行采样点。
"""

from __future__ import annotations

import numpy as np

try:  # 作为包导入
    from .green import hankel0_2
    from .quad import DEFAULT_QUAD_ORDER, self_coupling_integral
except ImportError:  # 直接以 src 为模块搜索路径导入
    from green import hankel0_2
    from quad import DEFAULT_QUAD_ORDER, self_coupling_integral


__all__ = [
    "circle_geometry",
    "assemble_impedance",
    "assemble_excitation",
    "solve_dense",
    "surface_current",
    "scattered_field_on_target",
    "total_field_residual",
    "mie_surface_current",
    "mie_scattered_field",
]


# ============================================================================
# 几何
# ============================================================================

def circle_geometry(radius: float, n_segments: int) -> dict:
    """把半径 radius 的圆折线化成 n_segments 条等分弦。

    返回字典,含:
        edges    (N+1, 2)  折线顶点(首尾不重合,edges[N] 是绕回起点后的位置)
        mid      (N, 2)    各段中点,即配点
        tangent  (N, 2)    各段单位切向
        length   (N,)      各段弦长
        phi_mid  (N,)      各段中点对应的极角
        n, radius
    """
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError(f"radius 必须为正的有限数,收到 {radius!r}")
    if not isinstance(n_segments, (int, np.integer)):
        raise ValueError(f"n_segments 必须是整数,收到 {n_segments!r}")
    if n_segments < 3:
        raise ValueError(f"n_segments 至少为 3,收到 {n_segments!r}")

    phi_edges = np.linspace(0.0, 2.0 * np.pi, int(n_segments) + 1)
    edges = radius * np.column_stack([np.cos(phi_edges), np.sin(phi_edges)])
    p0, p1 = edges[:-1], edges[1:]
    seg = p1 - p0
    length = np.linalg.norm(seg, axis=1)
    return {
        "edges": edges,
        "mid": 0.5 * (p0 + p1),
        "tangent": seg / length[:, None],
        "length": length,
        "phi_mid": 0.5 * (phi_edges[:-1] + phi_edges[1:]),
        "n": int(n_segments),
        "radius": float(radius),
    }


# ============================================================================
# 装配
# ============================================================================

def assemble_impedance(k: float, geom: dict, quad_order: int = DEFAULT_QUAD_ORDER) -> np.ndarray:
    """装配阻抗矩阵 Z_{mn} = +(k/4) ∫_{段 n} H₀⁽²⁾(k|r_m - r'|) dl'(符号见模块 docstring)。

    非对角元用 quad_order 点 Gauss-Legendre:被积函数在段 n 上是光滑的
    (段是直线,场点 r_m 不在段 n 上,故 R 有正下确界),高阶求积即可。
    对角元(m = n)被积函数在 t = 0 处对数奇异,交给 quad.self_coupling_integral
    (主路径为 Struve 闭式,O(1) 全域有效)。

    逐行计算,避免一次分配 (N, N, Q, 2) 的中间数组。
    """
    if not np.isfinite(k) or k <= 0.0:
        raise ValueError(f"k 必须为正的有限数,收到 {k!r}")
    if not isinstance(quad_order, (int, np.integer)) or quad_order < 2:
        raise ValueError(f"quad_order 必须是 >= 2 的整数,收到 {quad_order!r}")
    n = geom["n"]
    mid, length, tangent = geom["mid"], geom["length"], geom["tangent"]
    half = 0.5 * length

    nodes, weights = np.polynomial.legendre.leggauss(int(quad_order))
    # 源点局部坐标 (N, Q) -> 全局坐标 (N, Q, 2)
    t_local = half[:, None] * nodes[None, :]
    src = mid[:, None, :] + t_local[:, :, None] * tangent[:, None, :]

    Z = np.empty((n, n), dtype=complex)
    for m in range(n):
        diff = src - mid[m]                       # (N, Q, 2)
        R = np.linalg.norm(diff, axis=-1)         # (N, Q)
        vals = np.empty(R.shape, dtype=complex)
        off = R > 0.0
        vals[off] = hankel0_2(k * R[off])
        vals[~off] = 0.0                          # 仅当 m == n 且节点落中点;随后覆盖
        row = 0.25 * k * half * np.sum(weights[None, :] * vals, axis=1)
        row[m] = 0.25 * k * self_coupling_integral(k, half[m])
        Z[m, :] = row
    return Z


def assemble_excitation(k: float, geom: dict, theta_i: float = 0.0,
                        amplitude: float = 1.0) -> np.ndarray:
    """V_m = A·exp(+jk(x_m cosθ_i + y_m sinθ_i)),取段中点为配点。"""
    if not np.isfinite(k) or k <= 0.0:
        raise ValueError(f"k 必须为正的有限数,收到 {k!r}")
    if not np.isfinite(theta_i):
        raise ValueError(f"theta_i 必须是有限数,收到 {theta_i!r}")
    x, y = geom["mid"][:, 0], geom["mid"][:, 1]
    return amplitude * np.exp(1j * k * (x * np.cos(theta_i) + y * np.sin(theta_i)))


def solve_dense(Z: np.ndarray, V: np.ndarray) -> tuple[np.ndarray, float]:
    """稠密 LU 求解 Z J = V。

    返回 (J, 相对残差 ||ZJ - V|| / ||V||)。残差是对求解质量的检查,
    不是对物理正确性的检查 —— 物理正确性必须靠 Mie 级数(见模块 docstring)。
    """
    J = np.linalg.solve(Z, V)
    v_norm = float(np.linalg.norm(V))
    if v_norm == 0.0:
        # V 恒零(如 amplitude=0)时 J 也恒零,残差比 0/0 无定义。
        # 显式约定为 0,免得在调用方变成一个 nan 和一条 RuntimeWarning。
        return J, 0.0
    rel_resid = float(np.linalg.norm(Z @ J - V) / v_norm)
    return J, rel_resid


def surface_current(k: float, radius: float, n_segments: int,
                    theta_i: float = 0.0, amplitude: float = 1.0,
                    quad_order: int = DEFAULT_QUAD_ORDER) -> dict:
    """一站式:建几何 → 装配 → 求解。返回 dict(geom, Z, V, J, rel_resid)。"""
    geom = circle_geometry(radius, n_segments)
    Z = assemble_impedance(k, geom, quad_order=quad_order)
    V = assemble_excitation(k, geom, theta_i=theta_i, amplitude=amplitude)
    J, rel_resid = solve_dense(Z, V)
    return {"geom": geom, "Z": Z, "V": V, "J": J, "rel_resid": rel_resid}


# ============================================================================
# 场重建与边界条件检验
# ============================================================================

def scattered_field_on_target(k: float, geom: dict, J: np.ndarray,
                              points: np.ndarray,
                              quad_order: int = DEFAULT_QUAD_ORDER) -> np.ndarray:
    """在给定场点处重算散射场 E_z^sca = -(k/4) Σ_n J_n ∫_{段n} H₀⁽²⁾(kR) dl'。

    points 形状 (P, 2)。场点若非常靠近某段,被积函数会近奇异,此处用普通
    Gauss 求积;本函数只用于边界条件残差这类诊断,不进入迭代求解。

    ⚠️ 必须用偶阶:Gauss-Legendre 的奇数阶会把 t = 0 也当作节点,当场点正好
    落在某一段上时该节点的 R = 0,被下方 vals[~off] = 0.0 直接置零,等于丢掉
    自积分的中心贡献。实测 total_field_residual(on_nodes=False) 在 quad_order
    从 64 变 65 时从 4.30e-04 劣化到 5.12e-03(12 倍)。因此这里显式拒绝奇数阶,
    而不是让它静默劣化。
    """
    if not isinstance(quad_order, (int, np.integer)) or quad_order < 2:
        raise ValueError(f"quad_order 必须是 >= 2 的整数,收到 {quad_order!r}")
    if int(quad_order) % 2 != 0:
        raise ValueError(f"quad_order 必须为偶数,收到 {quad_order!r}(奇数阶会让 t=0 成为节点)")
    mid, length, tangent = geom["mid"], geom["length"], geom["tangent"]
    nodes, weights = np.polynomial.legendre.leggauss(int(quad_order))
    half = 0.5 * length
    src = mid[:, None, :] + (half[:, None] * nodes[None, :])[:, :, None] * tangent[:, None, :]

    pts = np.atleast_2d(np.asarray(points, dtype=float))
    out = np.zeros(pts.shape[0], dtype=complex)
    for p in range(pts.shape[0]):
        diff = src - pts[p]                        # (N, Q, 2)
        R = np.linalg.norm(diff, axis=-1)
        vals = np.empty(R.shape, dtype=complex)
        off = R > 0.0
        vals[off] = hankel0_2(k * R[off])
        vals[~off] = 0.0                           # R=0 只在配点自身上;本分支不被覆盖
        integ = half * np.sum(weights[None, :] * vals, axis=1)
        out[p] = -0.25 * k * np.sum(J * integ)
    return out


def total_field_residual(k: float, geom: dict, J: np.ndarray,
                         theta_i: float = 0.0, amplitude: float = 1.0,
                         on_nodes: bool = True,
                         quad_order: int = DEFAULT_QUAD_ORDER) -> dict:
    """在非配点处检验边界条件 E_z^inc + E_z^sca ≈ 0。

    点匹配只保证配点(段中点)上方程成立,节点处是否也接近零是插值质量的
    独立证据。返回 dict(points, total, max_abs, rms_abs, scale)。
    """
    if on_nodes:
        pts = geom["edges"][:-1]                   # N 个折线顶点(与配点交错)
    else:
        pts = geom["mid"]
    E_inc = amplitude * np.exp(1j * k * (pts[:, 0] * np.cos(theta_i)
                                        + pts[:, 1] * np.sin(theta_i)))
    E_sca = scattered_field_on_target(k, geom, J, pts, quad_order=quad_order)
    total = E_inc + E_sca
    return {
        "points": pts,
        "total": total,
        "max_abs": float(np.max(np.abs(total))),
        "rms_abs": float(np.sqrt(np.mean(np.abs(total) ** 2))),
        "scale": float(np.max(np.abs(E_inc))),
    }


# ============================================================================
# Mie 级数独立真值
# ============================================================================

def mie_surface_current(k: float, radius: float, phi, theta_i: float = 0.0,
                        amplitude: float = 1.0, n_terms: int | None = None) -> np.ndarray:
    """PEC 圆柱 TM 散射的表面电流解析真值(单位与 J_n = η₀J_z 一致)。

        η₀ J_z(φ) = (2 / (π k a)) Σ_n [j^n / H_n⁽²⁾(ka)] e^{jn(φ-θ_i)}

    截断项数默认 ceil(ka + 4·ka^(1/3)) + 10。圆柱散射的级数收敛很快,但**不是**
    机器精度 —— 实测截断相对误差(默认项数 vs 2 倍项数)为:

        ka      L2 截断误差       max|ΔJ|/max|J|
        1       1.11e-16          同量级
        5       4.94e-13          —
        10      1.20e-11          —
        20      3.43e-10          2.56e-10
        100     2.69e-08          2.16e-08

    即优于 1e-12 只到 ka ≈ 10。本项目判据用到 ka ≤ 20,此时 3.4e-10 远小于
    离散误差(~1e-4 量级),故不影响结论。phi 可为标量或数组。
    """
    from scipy.special import hankel2

    ka = k * radius
    if ka <= 0.0:
        raise ValueError(f"k*radius 必须为正,收到 {ka!r}")
    if n_terms is not None and (not isinstance(n_terms, (int, np.integer)) or n_terms < 1):
        raise ValueError(f"n_terms 必须为 None 或 >= 1 的整数,收到 {n_terms!r}")
    if n_terms is None:
        n_terms = int(np.ceil(ka + 4.0 * ka ** (1.0 / 3.0))) + 10
    ns = np.arange(-n_terms, n_terms + 1)
    weight = (1j ** ns) / hankel2(ns, ka)

    phi_arr = np.atleast_1d(np.asarray(phi, dtype=float))
    phase = np.exp(1j * np.outer(phi_arr - theta_i, ns))      # (P, 2M+1)
    return amplitude * (2.0 / (np.pi * ka)) * (phase @ weight)


def mie_scattered_field(k: float, radius: float, rho, phi, theta_i: float = 0.0,
                        amplitude: float = 1.0, n_terms: int | None = None) -> np.ndarray:
    """PEC 圆柱 TM 散射的散射场解析真值(仅对 ρ ≥ a 有意义)。

        E_z^sca(ρ,φ) = -Σ_n j^n [J_n(ka)/H_n⁽²⁾(ka)] H_n⁽²⁾(kρ) e^{jn(φ-θ_i)}

    与 mie_surface_current 是两条独立路径(一个来自 ∂_ρ 边界条件,一个来自散射场
    展开本身),用于交叉校验:由 MoM 解重建的远场应与本函数同号同量级。
    rho 与 phi 支持广播。
    """
    from scipy.special import hankel2, jv

    ka = k * radius
    if ka <= 0.0:
        raise ValueError(f"k*radius 必须为正,收到 {ka!r}")
    rho_arr = np.asarray(rho, dtype=float)
    if np.any(rho_arr < radius):
        raise ValueError(
            "mie_scattered_field 的展开只在散射体外部成立,要求 rho >= radius;"
            f"收到 rho_min = {float(np.min(rho_arr))!r} < radius = {radius!r}"
        )
    if n_terms is None:
        n_terms = int(np.ceil(ka + 4.0 * ka ** (1.0 / 3.0))) + 10
    ns = np.arange(-n_terms, n_terms + 1)

    rho_b, phi_b = np.broadcast_arrays(rho_arr, np.asarray(phi, dtype=float))
    flat_rho, flat_phi = rho_b.ravel(), phi_b.ravel()
    H = hankel2(ns[None, :], k * flat_rho[:, None])                     # (P, 2M+1)
    coef = (-jv(ns, ka) / hankel2(ns, ka)) * (1j ** ns)                 # (2M+1,)
    phase = np.exp(1j * (flat_phi[:, None] - theta_i) * ns[None, :])    # (P, 2M+1)
    out = (H * phase * coef[None, :]).sum(axis=1)
    return amplitude * out.reshape(rho_b.shape)
