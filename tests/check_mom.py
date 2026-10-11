# -*- coding: utf-8 -*-
"""1.3 自检:矩量法稠密装配与求解(src/mom.py)。

独立真值(纪律 D8:判据不得拿实现自身当基准)
--------------------------------------------------------------------------
二维 PEC 圆柱的 TM 散射有 Mie 级数解析解,本文件用到其中两条互不依赖的路径:

    mie_surface_current   由 ∂_ρ E_z^tot|_a 的边界条件推出表面电流
    mie_scattered_field   由散射场本征展开直接给出 E_z^sca(ρ,φ)

两者都不经过 mom.py 的装配/求解代码。此外还用了两条纯几何/结构判据(不需要真值):
    · Z 是循环矩阵且对称 —— 圆柱几何的必然结果
    · 节点处 |E_z^inc + E_z^sca| 应与**精确解在同一残差上的取值**可比
      (注意:该量有"多边形几何 + 点匹配"的固有地板,不能用绝对阈值判断,
       详见判据 E 的说明)

⚠️ 本文件包含两类不寻常的判据,请勿删除:
    · 判据 J(golden value)—— 把真值的数值钉进版本控制,防"真值本身写错时
      所有自洽性判据一起变绿"
    · 判据 K(判据自身的鉴别力)—— 给关键判据配"故意做错"的孪生用例,断言
      它必须变红。一个"无论代码对错都 PASS"的判据不是判据,是装饰。
      这条纪律来自两次实际失误:1.1 的连续性判据用 |H(t-ε)-H(t+ε)| < 1e-10
      (该差本就是 O(ε),必然红),1.2 的复核脚本用同一条公式当自己的真值
      (输出全 0,假通过)。

运行:
    python tests/check_mom.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mom import (  # noqa: E402
    assemble_impedance,
    circle_geometry,
    mie_scattered_field,
    mie_surface_current,
    surface_current,
    total_field_residual,
)

failures: list[str] = []


def report(name: str, value: float, tol: float, note: str = "") -> None:
    ok = np.isfinite(value) and value <= tol
    flag = "PASS" if ok else "FAIL"
    if not ok:
        failures.append(name)
    suffix = f"   {note}" if note else ""
    print(f"  [{flag}] {name:<46s} = {value:.6e}  (tol {tol:.1e}){suffix}")


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


# ============================================================================
print("=" * 82)
print("判据 A —— 阻抗矩阵的结构性质(纯物理,不用真值)")
print("=" * 82)

K, RADIUS, N = 1.0, 5.0, 200
geom = circle_geometry(RADIUS, N)
Z = assemble_impedance(K, geom)

# A1 循环结构:Z[m,n] 只依赖 (n-m) mod N
idx = np.arange(N)
circulant_dev = float(np.max(np.abs(Z - Z[0, (idx[None, :] - idx[:, None]) % N])))
report("A1 循环结构 max|Z[m,n] - Z[0,(n-m)%N]|", circulant_dev, 1e-13)

# A2 对称性(循环 + 圆对称 ⇒ 对称)
report("A2 max|Z - Z.T|", float(np.max(np.abs(Z - Z.T))), 1e-13)

# A3 已删除(判据设计错误,保留记载以免后人重犯):
#   原设想"Im(Z_mm) < 0 对应辐射损耗,故应为负",实测 k=1,a=5 时 Im(Z_mm) = +9.1448e-02。
#   根因:Z_mm 里含自场 kR→0 奇异性贡献的自能项,它不是天线的输入阻抗,虚部符号
#   与辐射功率没有对应关系。辐射性质改由判据 F(远场符号与量级)守护;
#   定量的辐射功率校验放到实验 E1(与 Mie 级数总截面比较)。

# A4 相邻段耦合应强于对径段
opp = N // 2
report("A4 |Z_00| / |Z_0,对径| 应 > 1",
       max(1.0 - abs(Z[0, 0]) / abs(Z[0, opp]), 0.0), 0.0,
       note=f"比值为 {abs(Z[0,0])/abs(Z[0,opp]):.3f}")

# ============================================================================
print()
print("=" * 82)
print("判据 B —— 线性系统:求解残差与真值代回")
print("=" * 82)

res = surface_current(K, RADIUS, 400)
J, Zb, V, geomb = res["J"], res["Z"], res["V"], res["geom"]
Jm = mie_surface_current(K, RADIUS, geomb["phi_mid"])

report("B1 相对残差 ||ZJ-V||/||V||", res["rel_resid"], 1e-12)
report("B2 Mie 真值代回 ||Z*Jmie-V||/||V||", rel_l2(Zb @ Jm, V), 5e-4,
       note="真值满足离散方程的程度 = 离散误差本身")

# ============================================================================
print()
print("=" * 82)
print("判据 C —— 表面电流 vs Mie 级数真值")
print("=" * 82)

err_b = rel_l2(J, Jm)
report("C1 ka=5, N=400 电流相对误差", err_b, 5e-4)

res1 = surface_current(1.0, 1.0, 200)
Jm1 = mie_surface_current(1.0, 1.0, res1["geom"]["phi_mid"])
report("C2 ka=1, N=200 电流相对误差", rel_l2(res1["J"], Jm1), 5e-4)

# 逐点误差不应有系统性偏差(如端点或影子边界堆积)
per_point = np.abs(J - Jm) / np.abs(Jm).max()
report("C3 逐点最大误差 / max|Jmie|", float(per_point.max()), 2e-3)

# ============================================================================
print()
print("=" * 82)
print("判据 D —— 网格收敛阶(N 加倍,误差应降约 4 倍)")
print("=" * 82)

errs = []
for n in (100, 200, 400, 800):
    r = surface_current(K, RADIUS, n)
    jm = mie_surface_current(K, RADIUS, r["geom"]["phi_mid"])
    errs.append(rel_l2(r["J"], jm))

for i, (n, e) in enumerate(zip((100, 200, 400, 800), errs)):
    note = "" if i == 0 else f"ratio = {errs[i-1]/e:.3f}"
    print(f"  [info] N={n:<4d}  rel_err = {e:.6e}   {note}")

ratios = [errs[i] / errs[i + 1] for i in range(len(errs) - 1)]
worst = max(abs(r - 4.0) for r in ratios)
report("D1 收敛比 |ratio - 4| 最大偏离", worst, 0.35,
       note="二阶收敛 O(h^2)")

# ============================================================================
print()
print("=" * 82)
print("判据 E —— 节点边界条件残差(含多边形几何地板,判据以精确解为尺子)")
print("=" * 82)
print("  说明:节点残差有两个来源 —— (a) 解本身的电流误差,(b) 多边形几何 +")
print("        点匹配的固有地板。把**精确** Mie 电流代进同一个残差就量出了 (b)。")
print("        实测(k=5, a=5, N=400):resid(J_mom) = 1.652431e-04,")
print("                               resid(J_mie) = 2.564214e-04")
print("        —— 精确解反而**更差 1.55 倍**。所以绝对阈值毫无意义(旧判据用 2e-3,")
print("        属于靠运气过)。现改为拿精确解当尺子的相对判据,它才真有鉴别力。")

rt_mom = total_field_residual(K, geomb, J)
rt_mie = total_field_residual(K, geomb, Jm)
res_mom = rt_mom["max_abs"] / rt_mom["scale"]
res_mie = rt_mie["max_abs"] / rt_mie["scale"]

report("E1 节点残差(J_mom) / 节点残差(J_mie)", res_mom / res_mie, 1.5,
       note=f"J_mom = {res_mom:.3e}, 精确解地板 = {res_mie:.3e}")
report("E2 节点残差 / 配点电流误差", res_mom / err_b, 10.0,
       note="量级自查,非硬判据")

# ============================================================================
print()
print("=" * 82)
print("判据 F —— 符号守护(防回归:装配式与 G 约定必须一致)")
print("=" * 82)
print("  说明:由 MoM 解重建的远场必须与 Mie 解析散射场同号、同幅值、同相位。")
print("        若有人把 (∇²+k²)G = +δ 带来的负号丢掉,本判据会立刻红。")
print("        全复判据 |q-1| 同时守符号/幅值/相位;单看 Re 会放行 J×(1+1j)")
print("        (Re 依旧 PASS 而幅值差 41.4%),那一枪由判据 K 实测钉住。")

from mom import scattered_field_on_target  # 局部导入,避免顶部太长

for rho in (2.0 * RADIUS, 8.0 * RADIUS, 30.0 * RADIUS):
    e_an = mie_scattered_field(K, RADIUS, rho, 0.0)
    e_j = scattered_field_on_target(K, geomb, J, np.array([[rho, 0.0]]))[0]
    q = e_j / e_an
    report(f"F1  rho={rho:5.0f}  |E(from J)/E_mie - 1|", abs(q - 1.0), 1e-3,
           note=f"q = {q.real:+.6f}{q.imag:+.6f}j")
    report(f"F2  rho={rho:5.0f}  Re[E(from J)/E_mie] - 1", abs(q.real - 1.0), 1e-3,
           note=f"Re = {q.real:+.6f}")

# ============================================================================
print()
print("=" * 82)
print("判据 G —— ka 扫描(含电小与电中尺寸)")
print("=" * 82)

print("  分两段:共振邻域之外用 2e-3;邻域内单列,断言'确实劣化'且劣化有界。")
print("  为什么必须分开(0→1.3 复检发现):ka=19.995 处 rel_err = 1.82e-02,")
print("  是邻域外水平(~4.6e-04)的 39 倍,也远超 2e-3。那不是 bug,是 D13 记录的")
print("  EFIE 近零模效应 —— J₁₅ 的零点在 ka*=19.994430629816,19.995 离它最近。")
print("  原判据只采到 20.0,恰好错过了这个最坏点,给出'ka=20 是最坏点、裕度 1.36×'")
print("  的错误结论。掩住它比承认它危险得多。")

K_G = 1.0
NORMAL = (0.5, 1.0, 5.0, 10.0, 19.5, 19.9, 20.05, 20.5)
RESONANT = (19.99, 19.995, 20.0)


def _g(ka: float):
    n_seg = int(min(1600, max(64, 40 * ka)))
    r = surface_current(K_G, ka, n_seg)
    jm = mie_surface_current(K_G, ka, r["geom"]["phi_mid"])
    zmm = r["Z"][0, 0]
    return n_seg, rel_l2(r["J"], jm), zmm


print("  --- G1 共振邻域之外(容差 2e-3)---")
for _ka in NORMAL:
    _n, _e, _zmm = _g(_ka)
    report(f"G1 ka={_ka:<6g} N={_n:<4d} 电流相对误差", _e, 2e-3,
           note=f"Im(Z_mm)/|Z_mm| = {_zmm.imag/abs(_zmm):+.4f}")

print("  --- G2 共振邻域(已知劣化区,不追求 2e-3)---")
_res = []
for _ka in RESONANT:
    _n, _e, _zmm = _g(_ka)
    _res.append(_e)
    print(f"  [info] G2 ka={_ka:<7g} N={_n:<4d} rel_err = {_e:.6e}")
_worst = max(_res)
_lo = min(_g(k)[1] for k in (19.9, 20.05))
print(f"  [info] G2 邻域外参照水平 = {_lo:.6e}(取 ka=19.9 与 20.05)")
if _worst > 3.0 * _lo:
    print(f"  [PASS] G2a 共振邻域确实劣化: 最坏 {_worst:.6e} > 3× {_lo:.6e} = {3*_lo:.6e}")
else:
    print(f"  [FAIL] G2a 共振邻域未显著劣化({_worst:.6e} vs {_lo:.6e})"
          f" —— 近零模效应在变,需重新解释 D13")
    failures.append("G2a 共振未劣化")
report("G2b 共振邻域最坏误差上界", _worst, 5e-2,
       note="实测最坏在 ka=19.995 附近,约 1.82e-02;劣化不得失控")

# ============================================================================
print()
print("=" * 82)
print("判据 H —— 入射角(检验 theta_i 的相位约定)")
print("=" * 82)

for th in (0.0, np.pi / 4.0, np.pi / 2.0, np.pi):
    r = surface_current(K, RADIUS, 400, theta_i=th)
    jm = mie_surface_current(K, RADIUS, r["geom"]["phi_mid"], theta_i=th)
    e = rel_l2(r["J"], jm)
    report(f"H  theta_i={th:+.4f} rad 电流相对误差", e, 5e-4)

# ============================================================================
print()
print("=" * 82)
print("判据 I —— 输入校验")
print("=" * 82)

from quad import self_coupling_integral  # 判据 I 的守卫属于 quad.py,一并钉住

BAD_INPUTS = (
    ("I1  circle_geometry radius<=0", lambda: circle_geometry(-1.0, 10)),
    ("I2  circle_geometry n_segments<3", lambda: circle_geometry(1.0, 2)),
    ("I3  assemble_impedance k<=0",
     lambda: assemble_impedance(0.0, circle_geometry(1.0, 10))),
    ("I4  mie_surface_current k*radius<=0",
     lambda: mie_surface_current(0.0, 1.0, 0.0)),
    ("I5  mie_surface_current n_terms=0",
     lambda: mie_surface_current(1.0, 1.0, 0.0, n_terms=0)),
    ("I6  mie_surface_current n_terms=2.5",
     lambda: mie_surface_current(1.0, 1.0, 0.0, n_terms=2.5)),
    ("I7  mie_scattered_field rho<radius",
     lambda: mie_scattered_field(1.0, 4.0, 3.9, 0.0)),
    ("I8  self_coupling_integral half_len=0",
     lambda: self_coupling_integral(1.0, 0.0)),
    ("I9  self_coupling_integral method 未知",
     lambda: self_coupling_integral(1.0, 1.0, method="bogus")),
)

for label, fn in BAD_INPUTS:
    try:
        fn()
        print(f"  [FAIL] {label} 未抛异常")
        failures.append(label)
    except ValueError as exc:
        print(f"  [PASS] {label} -> ValueError: {exc}")
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] {label} 抛了非 ValueError: {type(exc).__name__}: {exc}")
        failures.append(label)

print(f"  共 {len(BAD_INPUTS)} 项非法输入,全部应抛 ValueError")

# ============================================================================
print()
print("=" * 82)
print("判据 J —— 黄金值(把真值钉进版本控制)")
print("=" * 82)
print("  自洽性判据有一个共同的失效模式:真值函数本身被改坏时,所有判据一起变绿。")
print("  这里把 mpmath 独立算出的值硬编码。黄金值**不证明正确性**")
print("  (那是判据 A~K 的事),它只钉住'不被告知不许变'。")
print("  ⚠️ 精度口径(0→1.3 复检指出):这些字面量**不是全部 50 位**。实测相对偏差")
print("     分布很宽 —— J4 三值与 J3 的 φ=π 是真 50 位(偏差 ~1e-22);")
print("     J1 F(0.3)=2.65e-17、J2 (1000,0.01)=4.80e-17、J3 φ=π/2=3.83e-16")
print("     则明显只算到默认精度。好在判据容差(1e-14~1e-10)对最差那一项仍有")
print("     ≥180× 裕度,不影响判定;但「50 位」这个说法是过头了,已删。")
print("  重新生成:直接把 tests/gen_golden 的逻辑手写一遍(未落盘,见 D14 的取舍);")
print("     或对任一目标量用 mpmath 提高 dps 重算,再按上表逐项标出实际精度。")

from quad import C0, C1, closed_self_integral, hankel0_antiderivative  # noqa: E402

GOLDEN_F = {                       # hankel0_antiderivative(z) = ∫_0^z H_0^(2) du
    0.3: (0.2977575802045527593366, 0.43928317583850745339),
    1.0: (0.9197304100897602393144, 0.6370693766074230975448),
    5.0: (0.7153119177847678023275, -0.1997193876223363915326),
    25.0: (0.8710149211654587516909, 0.09360792735142898867856),
}
for z, (re, im) in GOLDEN_F.items():
    got = hankel0_antiderivative(z)
    report(f"J1  F({z})", abs(complex(got) - complex(re, im)), 1e-14,
           note=f"got {got.real:+.15f}{got.imag:+.15f}j")

GOLDEN_SCI = {                     # closed_self_integral(k, a) = (2/k) F(k a)
    (1.0, 0.5): (0.979361013292090110091, 1.123590911829280563734),
    (1.0, 1.0): (1.839460820179520478629, 1.27413875321484619509),
    (5.0, 1.0): (0.286124767113907120931, -0.07988775504893455661306),
    (10.0, 0.5): (0.1430623835569535604655, -0.03994387752446727830653),
    (1000.0, 0.01): (0.002134022607913473612675, -0.0004825806366453337197304),
}
for (k, a), (re, im) in GOLDEN_SCI.items():
    got = closed_self_integral(k, a)
    ref = complex(re, im)
    report(f"J2  closed_self_integral({k:g}, {a:g})", abs(got - ref) / abs(ref), 1e-13,
           note=f"got {got.real:+.15f}{got.imag:+.15f}j")

GOLDEN_J = {                       # mie_surface_current(ka=5, radius=1)
    0.0: (0.397207438220313624033, -1.998445769919587928712),
    np.pi / 2: (0.4960018065940361619542, -0.3314070328923063837542),
    np.pi: (-0.003001732168708405463585, 0.02742767861817495482431),
}
# tol 定 1e-10(不是 1e-12):phi=π 是级数收敛最慢的点(影子边界)。
# 项数必须说准(0→1.3 复检指出):默认 n_terms = ceil(ka + 4·ka^(1/3)) + 10,
# ka=5 时 = 22 ⇒ **45 项**(n 从 -22 到 +22)。该点残差实测 2.74e-11。
# 若只取 41 项(n_max=20),残差是 1.96e-9 —— 那是另一回事,别混。
for phi, (re, im) in GOLDEN_J.items():
    got = complex(mie_surface_current(5.0, 1.0, np.atleast_1d(phi))[0])   # 该函数返回 (1,) 数组
    ref = complex(re, im)
    report(f"J3  mie_surface_current(ka=5, phi={phi:.4f})", abs(got - ref) / abs(ref), 1e-10,
           note=f"got {got.real:+.15f}{got.imag:+.15f}j")

GOLDEN_E = {                       # mie_scattered_field(k=5, radius=1, phi=0, theta_i=0)
    10.0: (0.1642230337432168375896, 0.1635660075528778505726),
    40.0: (-0.0008746386322500921853305, 0.1136202864865045483717),
    150.0: (-0.01239578687311111201526, -0.0570681920356241963052),
}
# 生成时踩过的坑(留档):第一版黄金值用 mp.hankel2(n, rho) 而漏了乘 k —— 该函数
# 要的是 k*rho(k=5 时 rho=10 对应 kρ=50,不是 10)。结果是"正确的代码被错误的黄金值
# 判红"。黄金值自身也必须复核,这正是 J 判据要交给第三方检查的原因。
for rho, (re, im) in GOLDEN_E.items():
    got = complex(mie_scattered_field(5.0, 1.0, rho, 0.0))
    ref = complex(re, im)
    report(f"J4  mie_scattered_field(rho={rho:.0f})", abs(got - ref) / abs(ref), 1e-13,
           note=f"got {got.real:+.15f}{got.imag:+.15f}j")

report("J5  C0 = -1/4 - i(1-gamma)/(2pi)",
       abs(complex(C0) - complex(-0.25, -0.06728821679274134538924)), 1e-16)
report("J6  C1 = i/(2pi)",
       abs(complex(C1) - complex(0.0, 0.1591549430918953357689)), 1e-16)

# ============================================================================
print()
print("=" * 82)
print("判据 K —— 判据自身的鉴别力(故意做错,必须变红)")
print("=" * 82)
print("  一个'无论代码对错都 PASS'的判据不是判据,是装饰。本判据给关键判据各配")
print("  一个孪生反例,断言其度量值必须越过阈值 —— 否则说明该判据没有鉴别力。")


def must_fail(label: str, value: float, tol: float) -> None:
    if np.isfinite(value) and value > tol:
        print(f"  [PASS] {label}: 度量 = {value:.6e} > {tol:.1e}  -> 会红")
    else:
        print(f"  [FAIL] {label}: 度量 = {value:.6e} <= {tol:.1e} —— 判据无鉴别力!")
        failures.append(label)


# K1/K2:B2 用的真值被做坏时,B2 必须红
must_fail("K1  B2 对 -Jmie(符号错)的响应", rel_l2(Zb @ (-Jm), V), 5e-4)
must_fail("K2  B2 对 (1+1j)*Jmie(幅值错)的响应", rel_l2(Zb @ ((1 + 1j) * Jm), V), 5e-4)

# K3:A1 的循环性被破坏时,A1 必须红
# ⚠️ 这里必须用 Zb 自己的阶数。曾经写成 % N(判据 A 的 N=200)而 Zb 是 400x400,
#    后果是**不施加任何扰动**度量就已经是 6.065e-02(比 tol 大 1e11 倍),判据恒红、
#    与扰动无关 —— 空转的判据伪装成了有效的判据。由 0→1.3 第三方复核 P1 抓出。
_Zbad = Zb.copy()
_Zbad[0, 1] += 1e-3
_idx = np.arange(Zb.shape[0])
_k3_metric = float(np.max(np.abs(_Zbad - _Zbad[0, (_idx[None, :] - _idx[:, None]) % Zb.shape[0]])))
_k3_base = float(np.max(np.abs(Zb - Zb[0, (_idx[None, :] - _idx[:, None]) % Zb.shape[0]])))
print(f"  [info] K3 基线(Zb 自身循环性偏离)= {_k3_base:.6e}  <- 必须 < tol,否则本判据空转")
if not (_k3_base < 1e-13):
    print(f"  [FAIL] K3 基线 {_k3_base:.6e} 已超 tol,判据无鉴别力")
    failures.append("K3 基线")
must_fail("K3  A1 对破坏循环性的响应", _k3_metric, 1e-13)

# K4/K4b:F1 的指标升级自证 —— 旧指标 |Re(q)-1| 会放行 q = (1+1j)·q_good
_krho = 10.0 * RADIUS
_kan = mie_scattered_field(K, RADIUS, _krho, 0.0)
_kfield = scattered_field_on_target(K, geomb, J, np.array([[_krho, 0.0]]))[0]
_q_bad = ((1 + 1j) * _kfield) / _kan
_old_metric = abs(_q_bad.real - 1.0)
_new_metric = abs(_q_bad - 1.0)
must_fail("K4  F1 全复指标对 (1+1j)E 的响应", _new_metric, 1e-3)
if _old_metric <= 1e-3:
    print(f"  [PASS] K4b 旧指标 |Re(q)-1| 对该反例 = {_old_metric:.6e} <= 1e-3,"
          f"确实会放行 —— 换指标是必要的")
else:
    print(f"  [FAIL] K4b 旧指标也红({_old_metric:.6e}),换指标的理由不成立")
    failures.append("K4b")

# K5:E1 的"尺子"被换成无信息量版本时的对照
print(f"  [info] K5 若把 E1 写成 resid(J_mom)/|J_mom| 的绝对阈值 2e-3,"
      f"实测 {res_mom:.3e} 恰好擦边通过;换尺子后为 {res_mom/res_mie:.3f},"
      f"裕度从 12x 变成 1.6x 且语义明确")

# ============================================================================
print()
print("=" * 82)
if failures:
    print(f"结果: {len(failures)} 项未通过 -> {failures}")
    raise SystemExit(1)
print("结果: 全部通过")
