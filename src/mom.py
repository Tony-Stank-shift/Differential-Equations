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
    if radius <= 0.0:
        raise ValueError(f"radius 必须为正,收到 {radius!r}")
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
    if k <= 0.0:
        raise ValueError(f"k 必须为正,收到 {k!r}")
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
    x, y = geom["mid"][:, 0], geom["mid"][:, 1]
    return amplitude * np.exp(1j * k * (x * np.cos(theta_i) + y * np.sin(theta_i)))


def solve_dense(Z: np.ndarray, V: np.ndarray) -> tuple[np.ndarray, float]:
    """稠密 LU 求解 Z J = V。

    返回 (J, 相对残差 ||ZJ - V|| / ||V||)。残差是对求解质量的检查,
    不是对物理正确性的检查 —— 物理正确性必须靠 Mie 级数(见模块 docstring)。
    """
    J = np.linalg.solve(Z, V)
    rel_resid = float(np.linalg.norm(Z @ J - V) / np.linalg.norm(V))
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
    """
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
        vals[~off] = 0.0
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

    截断项数默认 ceil(ka) + 12;圆柱散射的级数收敛很快,该截断在 ka ≤ 100 内
    足以到机器精度。phi 可为标量或数组。
    """
    from scipy.special import hankel2

    ka = k * radius
    if ka <= 0.0:
        raise ValueError(f"k*radius 必须为正,收到 {ka!r}")
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
    if n_terms is None:
        n_terms = int(np.ceil(ka + 4.0 * ka ** (1.0 / 3.0))) + 10
    ns = np.arange(-n_terms, n_terms + 1)

    rho_b, phi_b = np.broadcast_arrays(np.asarray(rho, dtype=float),
                                       np.asarray(phi, dtype=float))
    flat_rho, flat_phi = rho_b.ravel(), phi_b.ravel()
    H = hankel2(ns[None, :], k * flat_rho[:, None])                     # (P, 2M+1)
    coef = (-jv(ns, ka) / hankel2(ns, ka)) * (1j ** ns)                 # (2M+1,)
    phase = np.exp(1j * (flat_phi[:, None] - theta_i) * ns[None, :])    # (P, 2M+1)
    out = (H * phase * coef[None, :]).sum(axis=1)
    return amplitude * out.reshape(rho_b.shape)
