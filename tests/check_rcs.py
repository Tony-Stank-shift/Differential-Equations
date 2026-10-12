# -*- coding: utf-8 -*-
"""1.4 自检:远场 RCS(src/rcs.py)。

本文件验的是"与 Mie 级数解析解对比达标"这条正式判据(D8 要求的独立真值),
外加远场定义自身的内部一致性、能量恒等式、有限 ρ 收敛性与输入校验。

--------------------------------------------------------------------------
独立真值的界定(承 1.3 的教训,见 DECISIONS D18)
--------------------------------------------------------------------------
能当"真值"用的只有 **Mie 解析级数**。本文件涉及两条"路径":

    f_J   来自 MoM 电流 J(经组装 + 求解,完全不经过 Mie 系数)
    f_M   来自 Mie 闭式系数 C_n = -J_n(ka)/H_n^(2)(ka)

判据 M(两条路径一致)因此是**真的独立证据** —— 与 check_mom.py 里
"mie_surface_current vs mie_scattered_field"那种同源自证不是一回事。

⚠️ 判据 E 是物理量级的独立证据:∫σ_2D dφ 必须等于 Mie 级数的总散射宽度闭式
``(4/k)Σ|C_n|²``。这条链左侧完全由电流积出、右侧完全由解析系数给出,
它们相等不是恒等式,是"同一份解必须给出同一个能量"。

⚠️ 本文件**故意不含**光学定理判据:其归一化尚未锁定(DECISIONS D20)。
不要加一条"从不红"的判据 —— 一个无论代码对错都 PASS 的判据不是判据,是装饰。

运行:
    python tests/check_rcs.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.special import hankel2, jv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mom import (  # noqa: E402
    assemble_excitation,
    assemble_impedance,
    circle_geometry,
    mie_scattered_field,
    scattered_field_on_target,
    solve_dense,
)
from rcs import (  # noqa: E402
    carrier,
    far_field_amplitude_from_current,
    mie_far_field_amplitude,
    scattering_width,
)

failures: list[str] = []


def report(name: str, value: float, tol: float, note: str = "") -> None:
    ok = np.isfinite(value) and value <= tol
    flag = "PASS" if ok else "FAIL"
    if not ok:
        failures.append(name)
    suffix = f"   {note}" if note else ""
    print(f"  [{flag}] {name:<48s} = {value:.6e}  (tol {tol:.1e}){suffix}")


def must_fail(label: str, value: float, tol: float) -> None:
    if np.isfinite(value) and value > tol:
        print(f"  [PASS] {label}: 度量 = {value:.6e} > {tol:.1e}  -> 会红")
    else:
        print(f"  [FAIL] {label}: 度量 = {value:.6e} <= {tol:.1e} --- 判据无鉴别力!")
        failures.append(label)


# ============================================================================
K = 1.0
RADIUS = 1.0
N_MAIN = 640          # 判据 M/E/R:比验收 N 更细,让"离散误差"远低于容差
N_ACC = 320           # 判据 C:项目正式验收指定的 N
NPHI = 4096
TAU = 2.0 * np.pi


def mie_coefficients(k: float, radius: float, extra_terms: int = 0):
    """标准 Mie 系数 C_n = -J_n(ka)/H_n^(2)(ka) 与其指标(独立于 rcs.py 重算)。

    ``extra_terms`` 用来做 **C3 截断自证**:把项数主动加多,看角方向图是否改变。
    """
    ka = k * radius
    n_terms = int(np.ceil(ka + 6.0 * ka ** (1.0 / 3.0))) + 60 + extra_terms
    ns = np.arange(-n_terms, n_terms + 1)
    return ns, -jv(ns, ka) / hankel2(ns, ka)


def total_scattering_width(k: float, radius: float) -> float:
    """二维总散射宽度闭式 ``(4/k)·Σ_n |C_n|²``。"""
    _, C = mie_coefficients(k, radius)
    return float(4.0 / k * np.sum(np.abs(C) ** 2))


def sphere_quadrature(phi: np.ndarray, y: np.ndarray) -> float:
    """周期函数在 [0, 2π) **均匀**网格上的积分。

    ⚠️ 不能直接用 ``np.trapezoid``:网格用 ``endpoint=False`` 时,最后一个面板
    ``[φ_{N-1}, 2π]`` 会被整段丢掉(相对误差 ≈ h·f(0)/∫f,在 N=4096 时已达 1.7e-4,
    足以让 tol=1e-8 的能量判据**假红**)。周期函数的正确写法是
        ∫_0^{2π} y dφ ≈ h·Σ_j y_j      (矩形/周期梯形,谱精度)
    """
    phi = np.asarray(phi, dtype=float)
    n = phi.size
    h = 2.0 * np.pi / n
    if n < 2 or not np.allclose(np.diff(phi), h, rtol=0.0, atol=1e-12):
        raise ValueError("本函数只支持 [0, 2π) 上的均匀网格")
    return float(np.sum(y) * h)


geom = circle_geometry(RADIUS, N_MAIN)
Z = assemble_impedance(K, geom)
V = assemble_excitation(K, geom)
J, _resid = solve_dense(Z, V)

phi = np.linspace(0.0, TAU, NPHI, endpoint=False)
f_J = far_field_amplitude_from_current(K, geom, J, phi)
f_M = mie_far_field_amplitude(K, RADIUS, phi)

# ============================================================================
print("=" * 84)
print("判据 M --- 两条独立路径(f_J 由 MoM 电流、f_M 由 Mie 闭式)必须一致")
print("=" * 84)
print("  只考察方向图不接近零点的角区(|f_M| > 0.1·max|f_M|):近零点处比值发散是")
print("  数学必然(两个小量相除),不是物理,不能拿来判错。")

floor = 0.1 * np.max(np.abs(f_M))
mask = np.abs(f_M) > floor
ratio = f_J[mask] / f_M[mask]
mean_ratio = np.mean(ratio)

# 容差 1e-3 = "两条路径的差别已降到离散化水平(实测 2.3e-5)之上 1 个量级"。
# ⚠️ 2026-10-12 复核员实测:容差取 2e-2 时,给 Mie 远场注入 0.01 rad(0.57°)的
#    系统性相位错 → **11 条判据全绿**。故收紧到 1e-3(仍留 44× 余量),让 0.01 rad
#    这类错误能红。M3 只测离散度,对纯幅度错失明(ptp=0),补 E2/S 兜。
report("M1  max|f_J/f_M - 1|", float(np.max(np.abs(ratio - 1.0))), 1e-3,
       note=f"mean = {mean_ratio.real:+.9f}{mean_ratio.imag:+.2e}j  ({mask.sum()}/{NPHI} 个角)")
report("M2  max|arg(f_J/f_M)|", float(np.max(np.abs(np.angle(ratio)))), 1e-3,
       note="Mie 角核写错(1 / j^n / (−1)^n 混淆)时该量约为 O(1)")
report("M3  ptp(|f_J|/|f_M|)", float(np.ptp(np.abs(ratio))), 1e-3)
report("M4  基线:max|f_M| 必须显著非零", 0.0 if np.max(np.abs(f_M)) > 1e-3 else np.inf, 0.0,
       note=f"max|f_M| = {np.max(np.abs(f_M)):.6e}")

# M5:纯解析路径 —— 只用 Mie 级数场(不经过 MoM 的装配/求解)在超大 kρ 处核对角核。
#     ⚠️ 这条**不是**独立真值:它与 f_M 共用 C_n。它的价值是**定位** H_n 渐近相位里
#     那个 jⁿ 记在哪 —— 角核写成 C_n 或 jⁿC_n 时这里给 O(1) 的 rel-L2。
#     真正独立于 Mie 的是 M1/M2/M3(f_J 来自电流)与判据 E。
ns5, C5 = mie_coefficients(K, RADIUS)
G5 = np.sqrt(2.0 / (np.pi * K)) * np.exp(1j * 0.25 * np.pi)
PH5 = np.exp(1j * np.outer(phi, ns5))
_rho5 = 1e5 * RADIUS
_ref5 = mie_scattered_field(K, RADIUS, _rho5, phi) * np.sqrt(_rho5) * np.exp(1j * K * _rho5)
rec5 = G5 * (PH5 @ (((-1.0) ** ns5) * C5))
report("M5  纯解析:超大 kρ 处角核 rel-L2", float(np.linalg.norm(rec5 - _ref5) / np.linalg.norm(_ref5)),
       1e-3, note=f"kρ = {K * _rho5:.0e},渐近误差 ~1/(kρ);不经 MoM")

# ---------------------------------------------------------------------------
# M6/M7/M8 是 2026-10-12 第二轮复核员指出的三个覆盖缺口:
#   M6 —— 除判据 C 外**所有判据都在 k=1 跑**,于是 "f_J 丢掉因子 k" 这类错误
#         只有 C 能红(它实测:注入该错 → 仅 C 红)。补一个 ka=5 的比值判据。
#   M7 —— `theta_i` 零覆盖:把 `phi_arr - theta_i` 改成 `+ theta_i` 后原本 42/42 全绿。
#   M8 —— C3「Mie 截断自证」原本没实现,且测试里的 n_terms 公式与 rcs.py 逐字相同,
#         参考值与实现**共享同一个截断** ⇒ 不能自证。必须主动加项数看结果是否变。
# ---------------------------------------------------------------------------

# M6:ka=5(即 k=5、radius=1),两条路径仍须一致。这一条让 k 的尺度进入判据。
_geom5 = circle_geometry(RADIUS, 640)
_K5 = 5.0 / RADIUS
_J5, _ = solve_dense(assemble_impedance(_K5, _geom5), assemble_excitation(_K5, _geom5))
_fJ5 = far_field_amplitude_from_current(_K5, _geom5, _J5, phi)
_fM5 = mie_far_field_amplitude(_K5, RADIUS, phi)
_m5 = np.abs(_fM5) > 0.1 * np.max(np.abs(_fM5))
_r5 = _fJ5[_m5] / _fM5[_m5]
report("M6  ka=5 时 max|f_J/f_M - 1|", float(np.max(np.abs(_r5 - 1.0))), 1e-2,
       note="除 C 之外的判据都在 k=1 跑;这条让 k 的尺度进入比值判据")

# M7:theta_i ≠ 0(斜入射)。MoM 侧走 assemble_excitation(theta_i=...),Mie 侧同角。
_THI7 = np.pi / 3.0
_J7, _ = solve_dense(assemble_impedance(K, geom),
                     assemble_excitation(K, geom, theta_i=_THI7))
_fJ7 = far_field_amplitude_from_current(K, geom, _J7, phi)
_fM7 = mie_far_field_amplitude(K, RADIUS, phi, theta_i=_THI7)
_m7 = np.abs(_fM7) > 0.1 * np.max(np.abs(_fM7))
_r7 = _fJ7[_m7] / _fM7[_m7]
report("M7  theta_i=π/3 时 max|f_J/f_M - 1|", float(np.max(np.abs(_r7 - 1.0))), 1e-2,
       note="把 phi-θi 写成 phi+θi 时,这条会红")
# M7b:圆柱的旋转性 —— 斜入射的方向图应当是正入射方向图平移 θi。
_shift_err = float(np.max(np.abs(_fJ7[_m7] - far_field_amplitude_from_current(K, geom, J, phi[_m7] - _THI7))))
report("M7b |f(φ;θi) - f(φ-θi;0)| 的绝对偏差(相对 max|f|)",
       _shift_err / np.max(np.abs(_fM7)), 1e-2, note="几何旋转对称性的直接检验")

# M8:C3 截断自证 —— 主动把 Mie 项数加多,角方向图必须不变。
for _ka8 in (1.0, 10.0):
    _k8 = _ka8 / RADIUS
    _w8 = np.exp(1j * np.outer(phi, mie_coefficients(_k8, RADIUS)[0]))
    _base = _w8 @ mie_coefficients(_k8, RADIUS)[1]
    _ns8, _C8 = mie_coefficients(_k8, RADIUS, extra_terms=50)
    _more = np.exp(1j * np.outer(phi, _ns8)) @ _C8
    report(f"M8  ka={_ka8:g} 默认项数 vs +50 项的 rel-L2",
           float(np.linalg.norm(_more - _base) / np.linalg.norm(_base)), 1e-12,
           note="C3(Mie 截断自证);用来排除'参考值与实现共享同一个错截断'")

# ============================================================================
print()
print("=" * 84)
print("判据 E --- 能量恒等式:∫σ_2D dφ 必须等于 Mie 总散射宽度闭式")
print("=" * 84)
print("  σ_2D(φ) = |f(φ)|²/|A|²,  ∫_0^{2π} σ_2D dφ = (4/k)Σ|C_n|²")
print("  左侧只用到电流/解析角函数,右侧只用到 Mie 系数 —— 独立。")

sigma_ref = total_scattering_width(K, RADIUS)
int_M = sphere_quadrature(phi, scattering_width(K, f_M))
int_J = sphere_quadrature(phi, scattering_width(K, f_J))

report("E1  |∫|f_M|²dφ - (4/k)Σ|C_n|²| / 闭式", abs(int_M - sigma_ref) / sigma_ref, 1e-8,
       note=f"∫ = {int_M:.10f}  闭式 = {sigma_ref:.10f}")
report("E2  |∫|f_J|²dφ - (4/k)Σ|C_n|²| / 闭式", abs(int_J - sigma_ref) / sigma_ref, 5e-3,
       note="容差 = MoM 离散误差量级,不是解析精度")

# ============================================================================
print()
print("=" * 84)
print("判据 A --- 载波因子 carrier 本身")
print("=" * 84)
print("  ⚠️ 2026-10-12 复核员实测:carrier 过去**零覆盖** —— 把它换成永远返回 42+0j 的")
print("  桩函数,全套自检仍然 29 PASS/0 FAIL。而模块 docstring 恰好预言过这个失败模式")
print("  (「f 整体差常数倍,而所有自比判据仍然全绿」)。用 H₀⁽²⁾ 的渐近直接钉它的模与相位:")
print("      H₀⁽²⁾(x) / carrier(k, x/k) → 1     (x = kρ ≫ 1,渐近误差 ~1/(8x))")

for _x in (1e3, 1e5):
    _k = 2.5
    _ratio = complex(hankel2(0, _x) / carrier(_k, _x / _k))
    report(f"A  x=kρ={_x:.0e}  |H₀⁽²⁾/carrier - 1|", abs(_ratio - 1.0),
           1e-2 if _x < 1e4 else 1e-4,
           note=f"实测 {abs(_ratio - 1.0):.2e};桩函数会让它变成 O(1)")

# ============================================================================
print()
print("=" * 84)
print("判据 S --- σ 定义的独立锚点:真实空间能流积分")
print("=" * 84)
print("  这是唯一能钉住「σ_2D 带不带 2π」的判据。")
print("  ⚠️ 准确说是:「E 守 σ 与物理闭式一致,S 才守约定本身选对了」。")
print("  第二轮复核员用反例纠正了'2π 对 E 完全不可见'的说法:只把 scattering_width 乘 2π")
print("  (σ 与闭式不一致)→ E1 立刻红(= 2π−1);只有把 σ 与闭式**一起**乘 2π(一致换")
print("  约定)时 E1/E2 才绿,此时只剩 S 红(实测 0.8408)。")
print("      总散射宽度 σ_tot = ρ∫_0^{2π}|E_sca|²dφ  必须等于  ∫σ_2D(φ)dφ")
print("  (ρ 是径向因子:二维 Poynting 通量 ∝ ρ∮|E_z|²dφ,入射强度 ∝ |E_inc|²。)")
print("  ⚠️ E 与 S 都只约束 ∫σ dφ,都不约束**角分布** —— 这是本套自检的已知边界。")

int_sigma_J = sphere_quadrature(phi, scattering_width(K, f_J))
for _rho_s in (1e3 * RADIUS, 1e4 * RADIUS):
    _pts = np.column_stack([_rho_s * np.cos(phi), _rho_s * np.sin(phi)])
    _Es = scattered_field_on_target(K, geom, J, _pts)
    _flux = _rho_s * sphere_quadrature(phi, np.abs(_Es) ** 2)
    report(f"S  ρ={_rho_s:.0e}a  |能流积分 - ∫σ_2D dφ| / 闭式",
           abs(_flux - int_sigma_J) / sigma_ref,
           1e-2 if _rho_s < 1e4 else 1e-3,
           note=f"能流 = {_flux:.9f}  ∫σ_2D = {int_sigma_J:.9f}")

# ============================================================================
print()
print("=" * 84)
print("判据 R --- f 就是 ρ→∞ 的极限(有限 ρ 场收敛到 f_J)")
print("=" * 84)

g_errors = []
for rho in (100.0 * RADIUS, 1000.0 * RADIUS):
    E = scattered_field_on_target(K, geom, J, np.array([[rho, 0.0]]))[0]
    g = E * np.sqrt(rho) * np.exp(1j * K * rho)     # 乘掉 e^{-jkρ}/√ρ
    fJ_at_0 = far_field_amplitude_from_current(K, geom, J, 0.0)
    g_errors.append(abs(g - fJ_at_0) / abs(fJ_at_0))

report("R1  ρ=1000a 时 |√ρe^{jkρ}E - f_J(0)|/|f_J(0)|", g_errors[1], 5e-3)
report("R2  e(1000a)/e(100a) 必须明显 < 1",
       float(g_errors[1] / g_errors[0]) if g_errors[0] > 0 else np.inf, 0.5,
       note=f"e(100a)={g_errors[0]:.3e}  e(1000a)={g_errors[1]:.3e}")

# ============================================================================
print()
print("=" * 84)
print("判据 C --- 正式验收:双站 RCS 相对 Mie 解析解的最大误差 < 2%")
print("=" * 84)
print("  **不用掩码**:实测在 N=320、ka≤10 时,有掩码(σ_M > 1e-3·max)与无掩码给出")
print("  完全相同的最大值 —— 即此刻「深零点放大相对误差」这件事并未发生,加掩码只会")
print("  削弱判据。等它真的发生再加,并说明理由。")
print("  ⚠️ 精度由 **N/ka(每波长段数)** 决定,不是由 N 单独决定:")
print("      每波长段数 = λ/(弦长) ≈ N/ka;误差 ~ (ka/N)²  ⇒ 2% 需要 N/ka ≳ 20")
print("      实测 N/ka = 320/64/32/21/10.7 → 3.7e-5 / 9.8e-4 / 4.8e-3 / 1.5e-2 / 6.1e-2")
print("  ⚠️⚠️ 但「N/ka ≳ 20 ⇒ 2%」只在 ka ≲ 40 成立(第二轮复核员用固定 N/ka=20 扫 ka")
print("      推翻了我的普遍性主张):ka=1 → 9.93e-3、10 → 1.22e-2、20 → 1.50e-2、")
print("      40 → 1.91e-2、**50(N=1000)→ 2.073e-2**、**80(N=1600)→ 2.573e-2** ⇒ 都超 2%。")
print("      即误差**不是** ka/N 的纯函数,还随 ka 缓慢恶化(相位误差沿周长累积)。")
print("      论文里给这个门槛时必须带适用范围,不能当普适定律。")

geom_acc = circle_geometry(RADIUS, N_ACC)
for ka_target in (1.0, 5.0, 10.0):
    kk = ka_target / RADIUS
    Zk = assemble_impedance(kk, geom_acc)
    Jk, _ = solve_dense(Zk, assemble_excitation(kk, geom_acc))
    sJ = scattering_width(kk, far_field_amplitude_from_current(kk, geom_acc, Jk, phi))
    sM = scattering_width(kk, mie_far_field_amplitude(kk, RADIUS, phi))
    err = float(np.max(np.abs(sJ - sM) / np.abs(sM)))
    report(f"C  ka={ka_target:g}, N={N_ACC} ({N_ACC / ka_target:.0f} 段/λ) 最大相对误差",
           err, 0.02)

# C-env:同一台机器上,把 N/ka 抬到 32 后 ka=20 也应回到 1e-2 以内。
#        这条用来证明上表那个"N/ka ≳ 20"的包络是真的,而不是只在 ka≤10 碰巧成立。
kk = 20.0
ge = circle_geometry(RADIUS, 640)
Jk, _ = solve_dense(assemble_impedance(kk, ge), assemble_excitation(kk, ge))
sJ = scattering_width(kk, far_field_amplitude_from_current(kk, ge, Jk, phi))
sM = scattering_width(kk, mie_far_field_amplitude(kk, RADIUS, phi))
report("C-env  ka=20, N=640 (32 段/λ) 最大相对误差",
       float(np.max(np.abs(sJ - sM) / np.abs(sM))), 1e-2)

# ============================================================================
print()
print("=" * 84)
print("判据 I --- 输入校验(坏输入必须抛 ValueError,不能静默返回垃圾)")
print("=" * 84)


_ErrState = np.seterr(all="ignore")   # 下面这批用例**故意**触发上溢/下溢;非有限结果已被 ValueError 拦下


def check_raises(label: str, fn, *args, **kwargs) -> None:
    try:
        fn(*args, **kwargs)
    except ValueError:
        print(f"  [PASS] {label}")
    except Exception as exc:                                  # noqa: BLE001
        print(f"  [FAIL] {label}: 抛了 {type(exc).__name__} 而不是 ValueError")
        failures.append(label)
    else:
        print(f"  [FAIL] {label}: 没有抛错")
        failures.append(label)


check_raises("I1  carrier(k=0)", carrier, 0.0, 1.0)
check_raises("I2  carrier(k=1, rho=-1)", carrier, 1.0, -1.0)
check_raises("I3  carrier(k=nan)", carrier, np.nan, 1.0)
_J_bad = J.copy()
_J_bad[0] = np.nan
check_raises("I4  f_J 的 J 含 NaN", far_field_amplitude_from_current, K, geom, _J_bad, phi)
check_raises("I5  f_J 的 J 长度不符", far_field_amplitude_from_current, K, geom, J[:-1], phi)
check_raises("I6  f_J 的 phi 含 inf", far_field_amplitude_from_current, K, geom, J,
             np.array([0.0, np.inf]))
check_raises("I7  f_M radius=0", mie_far_field_amplitude, K, 0.0, 0.0)
check_raises("I8  σ_2D amplitude=0", scattering_width, K, f_M, 0.0)

# --- 以下 7 条是 2026-10-12 复核员实测出的"静默垃圾出口",修完必须逐条守住 ---
print("  --- 复核员抓到的静默出口(修完逐条守住) ---")
check_raises("I9  σ_2D 的 f_phi 含 NaN", scattering_width, K, np.array([np.nan, 1.0]))
check_raises("I10 σ_2D amplitude=1e-200(|A|² 下溢)", scattering_width, K, f_M, 1e-200)
check_raises("I11 f_J 的 k=1e-320(√ 上溢)", far_field_amplitude_from_current, 1e-320, geom, J, phi)
check_raises("I12 f_J 的 geom 缺键(旧:KeyError)",
             far_field_amplitude_from_current, K, {"mid": geom["mid"]}, J, phi)
_g_bad = {"mid": geom["mid"], "length": geom["length"].copy()}
_g_bad["length"][0] = -_g_bad["length"][0]
check_raises("I13 f_J 的 geom 段长为负", far_field_amplitude_from_current, K, _g_bad, J, phi)
check_raises("I14 f_J 的 phi 为复数(旧:静默丢虚部)",
             far_field_amplitude_from_current, K, geom, J, np.array([0.0 + 1.0j]))
check_raises("I15 carrier(1e-320, 1.0)(旧:静默 inf)", carrier, 1e-320, 1.0)
check_raises("I16 f_M radius=1e-300(旧:静默 nan)", mie_far_field_amplitude, K, 1e-300, 0.0)
check_raises("I17 f_M 的 theta_i=1j(旧:静默返回有限垃圾)",
             mie_far_field_amplitude, K, RADIUS, np.array([0.0]), 1j)
check_raises("I18 carrier 的 k=1+0j(旧:TypeError 而非 ValueError)", carrier, 1 + 0j, 1.0)
check_raises("I19 f_M 的 k=1+0j(旧:TypeError 而非 ValueError)",
             mie_far_field_amplitude, 1 + 0j, RADIUS, np.array([0.0]))
check_raises("I20 σ_2D 的 amplitude=1j", scattering_width, K, f_M, 1j)

np.seterr(**_ErrState)                # 恢复 numpy 的默认浮点错误策略

# ============================================================================
print()
print("=" * 84)
print("判据 K --- 判据自身的鉴别力(故意做错,必须变红)")
print("=" * 84)
print("  一个'无论代码对错都 PASS'的判据不是判据,是装饰。")
print("  每个孪生反例先断言正确版本 PASS,再断言错误版本越过阈值。")

# 基线:正确版本必须在容差内
if (np.max(np.abs(ratio - 1.0)) <= 1e-3 and np.max(np.abs(np.angle(ratio))) <= 1e-3):
    print(f"  [PASS] K0  基线:f_J/f_M 一致(|比-1| = {np.max(np.abs(ratio - 1.0)):.2e},"
          f" |arg| = {np.max(np.abs(np.angle(ratio))):.2e})")
else:
    print("  [FAIL] K0  基线本身不一致 -> 下面的孪生反例无意义")
    failures.append("K0")

# K1:把 Mie 角核写成三种错版本,判据 M 必须红。
#     三种都保留 √(2/(πk))e^{jπ/4} 前置因子,好让"红"唯一归因于角核写错。
ns, C = mie_coefficients(K, RADIUS)
G = np.sqrt(2.0 / (np.pi * K)) * np.exp(1j * 0.25 * np.pi)
PH = np.exp(1j * np.outer(phi, ns))


def _m_metric(f_bad: np.ndarray, label: str, tol: float = 1e-3) -> None:
    m = np.abs(f_bad) > 0.1 * np.max(np.abs(f_bad))
    r = f_J[m] / f_bad[m]
    must_fail(f"{label}: max|f_J/f_M' - 1|", float(np.max(np.abs(r - 1.0))), tol)


_m_metric(G * (PH @ C), "K1a 角核写成 C_n(相对正确值旋转 π,即 180°)")
_m_metric(G * (PH @ ((1j ** ns) * C)), "K1b 角核写成 jⁿC_n(旋转 ∓π/2,即 90°)")
_m_metric(G * (PH @ (((-1j) ** ns) * C)), "K1c 角核写成 (−j)ⁿC_n")

# K4:M5 的鉴别力 —— 同一批错角核在纯解析路径上必须给出 O(1) 的 rel-L2
for _lbl, _fac in (("C_n", np.ones_like(ns5)), ("jⁿC_n", (1j) ** ns5)):
    _r5 = G5 * (PH5 @ (_fac * C5))
    must_fail(f"K4  角核={_lbl} 时 M5 的 rel-L2",
              float(np.linalg.norm(_r5 - _ref5) / np.linalg.norm(_ref5)), 1e-3)
print(f"        对照:正确角核 (−1)ⁿC_n 时 M1 = {np.max(np.abs(ratio - 1.0)):.3e}"
      f", M2 = {np.max(np.abs(np.angle(ratio))):.3e}(均应远小于 tol)")

# K2:把 f_J 的传播相位取共轭(符号写反)-> M1 必须红
f_J_conj = np.conj(f_J)
must_fail("K2  f_J 取共轭后 max|f_J/f_M - 1|",
          float(np.max(np.abs(f_J_conj[mask] / f_M[mask] - 1.0))), 1e-3)

# K5:carrier 换成桩函数 -> 判据 A 必须红(复核员实测:旧版本这里全绿)
_x5 = 1e5
must_fail("K5  carrier 换成 42+0j 桩函数后判据 A 的度量",
          abs(complex(hankel2(0, _x5) / (42.0 + 0.0j)) - 1.0), 1e-4)

# K6:K7:K8 —— 第二轮复核员指出的三个覆盖缺口各自的孪生反例
# K6:C3 截断自证必须有鉴别力 —— 把项数砍到 |n| ≤ 5,必须红
for _ka6 in (1.0, 10.0):
    _k6 = _ka6 / RADIUS
    _ns6 = np.arange(-5, 6)
    _C6 = -jv(_ns6, _k6 * RADIUS) / hankel2(_ns6, _k6 * RADIUS)
    _cut6 = np.exp(1j * np.outer(phi, _ns6)) @ _C6
    _ns6b, _C6b = mie_coefficients(_k6, RADIUS)
    _base6 = np.exp(1j * np.outer(phi, _ns6b)) @ _C6b
    must_fail(f"K6  ka={_ka6:g} 把 Mie 项数砍到 |n|≤5 时 M8 的度量",
              float(np.linalg.norm(_cut6 - _base6) / np.linalg.norm(_base6)), 1e-12)

# K7:把 Mie 侧的 (φ-θi) 写成 (φ+θi) -> M7 必须红
_fM7_bad = mie_far_field_amplitude(K, RADIUS, phi, theta_i=-_THI7)
_m7b = np.abs(_fM7_bad) > 0.1 * np.max(np.abs(_fM7_bad))
must_fail("K7  theta_i 反号后 M7 的度量",
          float(np.max(np.abs(_fJ7[_m7b] / _fM7_bad[_m7b] - 1.0))), 1e-2)

# K8:f_J 丢掉因子 k -> M6(ka=5)必须红;而它在 k=1 的 M1 上不可见
_fJ5_nok = far_field_amplitude_from_current(_K5, _geom5, _J5, phi) / _K5
must_fail("K8  f_J 丢因子 k 后 M6(ka=5)的度量",
          float(np.max(np.abs(_fJ5_nok[_m5] / _fM5[_m5] - 1.0))), 1e-2)
print("        对照:同一个错误在 k=1 时是**恒等操作**(f_J/1 ≡ f_J),所以 M1 完全看不见")
print("        —— 这正是必须补 M6(ka=5)的理由。")

# K3:把前置因子 √(2/(πk))e^{jπ/4} 拿掉 -> E1 必须红
#     |1/G|² = πk/2,所以拿掉后能量积分为 σ·(πk/2) ≠ σ。
f_no_factor = f_J / G
metric_k3 = abs(sphere_quadrature(phi, scattering_width(K, f_no_factor)) - sigma_ref) / sigma_ref
must_fail("K3  拿掉前置因子后 E1 的度量", metric_k3, 1e-8)
print(f"        对照:拿掉因子后 ∫|f|²dφ/闭式 = {1.0 + metric_k3:.6f}"
      f"(应 ≈ πk/2 = {np.pi * K / 2.0:.6f})")

# ============================================================================
print()
print("=" * 84)
if failures:
    print(f"结果: {len(failures)} 项未通过 -> {failures}")
    raise SystemExit(1)
print("结果: 全部通过")
