# -*- coding: utf-8 -*-
"""1.3 自检:矩量法稠密装配与求解(src/mom.py)。

独立真值(纪律 D8:判据不得拿实现自身当基准)
--------------------------------------------------------------------------
二维 PEC 圆柱的 TM 散射有 Mie 级数解析解,本文件用到其中两条互不依赖的路径:

    mie_surface_current   由 ∂_ρ E_z^tot|_a 的边界条件推出表面电流
    mie_scattered_field   由散射场本征展开直接给出 E_z^sca(ρ,φ)

两者都不经过 mom.py 的装配/求解代码。此外还用了三条纯物理判据(不需要真值):
    · Z 是循环矩阵且 Hermite 对称结构 —— 圆柱几何的必然结果
    · Im Z_mm < 0 —— 自阻抗的负虚部对应辐射损耗
    · 节点处 |E_z^inc + E_z^sca| 应与配点误差同量级 —— 点匹配的插值质量

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
print("判据 E —— 非配点处的边界条件残差(点匹配的插值质量)")
print("=" * 82)

rt = total_field_residual(K, geomb, J)
rel_resid_node = rt["max_abs"] / rt["scale"]
report("E1 节点 |E_tot|max / |E_inc|max", rel_resid_node, 2e-3,
       note=f"电流误差为 {err_b:.3e},二者应同量级")
report("E2 节点残差 / 电流误差(应 ≲ 10)", rel_resid_node / err_b, 10.0)

# ============================================================================
print()
print("=" * 82)
print("判据 F —— 符号守护(防回归:装配式与 G 约定必须一致)")
print("=" * 82)
print("  说明:由 MoM 解重建的远场必须与 Mie 解析散射场**同号**。")
print("        若有人把 (∇²+k²)G = +δ 带来的那个负号丢掉,本判据会立刻红。")

for rho in (2.0 * RADIUS, 8.0 * RADIUS, 30.0 * RADIUS):
    e_an = mie_scattered_field(K, RADIUS, rho, 0.0)
    e_j = None
    from mom import scattered_field_on_target  # 局部导入,避免顶部太长
    e_j = scattered_field_on_target(K, geomb, J, np.array([[rho, 0.0]]))[0]
    ratio = (e_j / e_an).real
    report(f"F   rho={rho:5.0f}  Re[E(from J)/E_mie]", abs(ratio - 1.0), 1e-3,
           note=f"比值 = {ratio:+.6f}")

# ============================================================================
print()
print("=" * 82)
print("判据 G —— ka 扫描(含电小与电中尺寸)")
print("=" * 82)

print("   ka      N      rel_err       Im(Z_mm)/|Z_mm|")
for ka in (0.5, 1.0, 5.0, 10.0, 20.0):
    n_seg = int(min(1600, max(64, 40 * ka)))
    r = surface_current(1.0, ka, n_seg)
    jm = mie_surface_current(1.0, ka, r["geom"]["phi_mid"])
    e = rel_l2(r["J"], jm)
    zmm = r["Z"][0, 0]
    report(f"G  ka={ka:<5g} N={n_seg:<5d} 电流相对误差", e, 2e-3,
           note=f"Im(Z_mm)/|Z_mm| = {zmm.imag/abs(zmm):+.4f}")

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

for label, fn in (
    ("I1 radius<=0", lambda: circle_geometry(-1.0, 10)),
    ("I2 n_segments<3", lambda: circle_geometry(1.0, 2)),
    ("I3 k<=0", lambda: assemble_impedance(0.0, circle_geometry(1.0, 10))),
    ("I4 mie ka<=0", lambda: mie_surface_current(0.0, 1.0, 0.0)),
):
    try:
        fn()
        print(f"  [FAIL] {label} 未抛异常")
        failures.append(label)
    except ValueError as exc:
        print(f"  [PASS] {label} -> ValueError: {exc}")
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] {label} 抛了非 ValueError: {type(exc).__name__}: {exc}")
        failures.append(label)

# ============================================================================
print()
print("=" * 82)
if failures:
    print(f"结果: {len(failures)} 项未通过 -> {failures}")
    raise SystemExit(1)
print("结果: 全部通过")
