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


def mie_coefficients(k: float, radius: float):
    """标准 Mie 系数 C_n = -J_n(ka)/H_n^(2)(ka) 与其指标(独立于 rcs.py 重算)。"""
    ka = k * radius
    n_terms = int(np.ceil(ka + 6.0 * ka ** (1.0 / 3.0))) + 60
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
print("  这是唯一能钉住「σ_2D 带不带 2π」的判据。判据 E 的两边都写在实现约定里,")
print("  所以 2π 记账错对 E 完全不可见(2026-10-12 复核员抓到 docstring 曾写 2πρ)。")
print("      总散射宽度 σ_tot = ρ∫_0^{2π}|E_sca|²dφ  必须等于  ∫σ_2D(φ)dφ")
print("  (ρ 是径向因子:二维 Poynting 通量 ∝ ρ∮|E_z|²dφ,入射强度 ∝ |E_inc|²。)")

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


_m_metric(G * (PH @ C), "K1a 角核写成 C_n(少 π/2)")
_m_metric(G * (PH @ ((1j ** ns) * C)), "K1b 角核写成 jⁿC_n(多 π/2)")
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
