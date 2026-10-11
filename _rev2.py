# -*- coding: utf-8 -*-
"""第三方复核 脚本2:攻击判据本身 + 边界/非法输入。"""
import os
import sys
import warnings

import numpy as np
from scipy.integrate import quad as squad
from scipy.special import hankel2 as sp_h2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

import mom
import quad

K, RAD, N = 1.0, 5.0, 400
res = mom.surface_current(K, RAD, N)
J, Z, V, g = res["J"], res["Z"], res["V"], res["geom"]
Jm = mom.mie_surface_current(K, RAD, g["phi_mid"])
errJ = np.linalg.norm(J - Jm) / np.linalg.norm(Jm)


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


print("=" * 78)
print("[A] 修正版独立积分校验:scipy.integrate.quad 分实部/虚部(N=200 抽样)")
print("=" * 78)
N2 = 200
g2 = mom.circle_geometry(5.0, N2)
Z2 = mom.assemble_impedance(1.0, g2)
worst = (0.0, None)
for m in range(0, N2, 11):
    pm = g2["mid"][m]
    for n in range(N2):
        if m == n:
            continue
        midn, tann, h = g2["mid"][n], g2["tangent"][n], 0.5 * g2["length"][n]
        fr = lambda t: sp_h2(0, np.linalg.norm(pm - (midn + t * tann))).real
        fi = lambda t: sp_h2(0, np.linalg.norm(pm - (midn + t * tann))).imag
        vr, _ = squad(fr, -h, h, limit=400, epsabs=1e-15, epsrel=1e-13)
        vi, _ = squad(fi, -h, h, limit=400, epsabs=1e-15, epsrel=1e-13)
        d = abs(0.25 * (vr + 1j * vi) - Z2[m, n])
        if d > worst[0]:
            worst = (d, (m, n))
print(f"  抽样 max|dZ| = {worst[0]:.3e}  at ({worst[1][0]},{worst[1][1]})   "
      f"|Z|mean={np.abs(Z2).mean():.3e}  -> 相对 {worst[0]/np.abs(Z2).mean():.2e}")

print()
print("=" * 78)
print("[B] 判据 F 是 Re 投影,不是幅值判据 —— 构造反例")
print("=" * 78)
pt = np.array([[10.0, 0.0], [40.0, 0.0], [150.0, 0.0]])
ej = mom.scattered_field_on_target(K, g, J, pt)
for tag, scale in [("原始 J", 1.0), ("J*(1+1j)(幅值+41%,相位+45deg)", 1 + 1j),
                   ("J*1.0009(幅值+0.09%)", 1.0009), ("J*0.5(幅值减半)", 0.5)]:
    print(f"  {tag}:")
    for i, rho in enumerate((10.0, 40.0, 150.0)):
        ea = complex(mom.mie_scattered_field(K, RAD, rho, 0.0))
        q = scale * ej[i] / ea
        verdict = "PASS" if abs(q.real - 1.0) <= 1e-3 else "FAIL"
        print(f"      rho={rho:<5g} Re={q.real:+.6f} |ratio|={abs(q):.6f} -> {verdict}")

print()
print("=" * 78)
print("[C] 判据 F 的第二面:重建场是否在整个角向/径向都对(不止 phi=0)")
print("=" * 78)
for rho in (10.0, 30.0, 150.0):
    ph = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    pts = np.column_stack([rho * np.cos(ph), rho * np.sin(ph)])
    e_num = mom.scattered_field_on_target(K, g, J, pts)
    e_an = mom.mie_scattered_field(K, RAD, rho, ph)
    q = e_num / e_an
    print(f"  rho={rho:<6g} 全角向 rel L2 = {rel(e_num,e_an):.3e}   "
          f"Re(q) in [{q.real.min():.6f},{q.real.max():.6f}]  Im(q) max|.|={np.abs(q.imag).max():.2e}")

print()
print("=" * 78)
print("[D] 判据 E 到底在测什么:节点残差对 quad_order 的依赖")
print("=" * 78)
for qo in (32, 64, 128, 256, 512):
    r1 = mom.total_field_residual(K, g, J, quad_order=qo)
    r2 = mom.total_field_residual(K, g, Jm, quad_order=qo)
    print(f"  quad_order={qo:<4d}  E1(J_mom)={r1['max_abs']/r1['scale']:.6e}   "
          f"节点残差(J_mie)={r2['max_abs']/r2['scale']:.6e}   ratio={r1['max_abs']/r2['max_abs']:.3f}")
print("  ^ 若 J_mie 行的残差随 quad_order 单调下降,而 J_mom 行收敛到某常数,说明 E1 被求积误差污染")
print()
mid = g["mid"]
for qo in (64, 256, 1024):
    es = mom.scattered_field_on_target(K, g, J, mid, quad_order=qo)
    print(f"  quad_order={qo:<5d} 配点处 |(-E_sca)-(Z@J)| max = {np.max(np.abs(-es - Z@J)):.3e}"
          f"   (理论上界: 应为 0,因为 Z@J=V 且 -E_sca|配点 就是 Z@J)")
print("  ^ 配点处场点落在段中点 => 核在 R=0 被置 0(mom.py:217-218),对角贡献整体缺失")

print()
print("=" * 78)
print("[E] 判据 H:theta_i 是否真被用上 + 符号/镜像盲区")
print("=" * 78)
print("  E-1 直接构造 theta 相位取反的 bug,看 H 会不会红:")
for th in (0.0, 0.3, np.pi / 2, np.pi):
    Vbug = mom.assemble_excitation(K, g, theta_i=-th)     # 模拟 e^{-jk(...)} 的错号
    Jbug, _ = mom.solve_dense(Z, Vbug)
    Jmth = mom.mie_surface_current(K, RAD, g["phi_mid"], theta_i=th)
    print(f"    theta={th:+.4f}  rel_err(错号J, 正确Mie) = {rel(Jbug,Jmth):.6e}   "
          f"(正确实现 {rel(mom.surface_current(K,RAD,N,theta_i=th)['J'], Jmth):.6e})")
print("  E-2 激励幅值取反(全局符号错)会不会被 H 抓到:")
for th in (0.0, 0.3):
    Vneg = -mom.assemble_excitation(K, g, theta_i=th)
    Jn, _ = mom.solve_dense(Z, Vneg)
    Jmth = mom.mie_surface_current(K, RAD, g["phi_mid"], theta_i=th)
    print(f"    theta={th:+.4f}  rel_err(-V) = {rel(Jn,Jmth):.6e}")
print("  E-3 旋转等价性检验:theta 从 0 加到 2pi/N 的整数倍,解只是循环移位")
for th in (0.0, 2 * np.pi / N, 4 * np.pi / N):
    r = mom.surface_current(K, RAD, N, theta_i=th)
    print(f"    theta={th:.8f}  rel_err={rel(r['J'], mom.mie_surface_current(K,RAD,g['phi_mid'],theta_i=th)):.15e}")

print()
print("=" * 78)
print("[F] 判据 B2 的可欺骗性:SVD 最小奇异向量方向上的扰动")
print("=" * 78)
u, s, vh = np.linalg.svd(Z)
print(f"  N=400 ka=5: cond(Z)={s[0]/s[-1]:.4e}  sigma_max={s[0]:.3e} sigma_min={s[-1]:.3e}")
print(f"  B2 定义:||Z*Jmie - V||/||V|| = {rel(Z@Jm, V):.6e}")
vmin = np.conj(vh[-1])
for amp in (1e-1, 1.0, 10.0, 100.0):
    Jbad = Jm + amp * vmin
    b2 = rel(Z @ Jbad, V) if np.linalg.norm(V) else float("nan")
    b2 = float(np.linalg.norm(Z @ Jbad - V) / np.linalg.norm(V))
    c = rel(Jbad, Jm)
    print(f"  扰动 {amp:>6g}*v_min: B2={b2:.3e} (判据 tol 5e-4 -> {'PASS' if b2<=5e-4 else 'FAIL'})   "
          f"但电流相对误差 C1 变为 {c:.3e} (tol 5e-4 -> {'PASS' if c<=5e-4 else 'FAIL'})")
print("  ^ 说明 B2 单独不足以守住 Mie 真值(它只守'真值是否满足离散方程')")
print("  另一路:给 Mie 真值乘一个常数因子 c,B2 的响应:")
for c in (1 + 1e-4, 1 + 1e-2, 1 + 0.5):
    print(f"    c={c:<8g} B2={float(np.linalg.norm(Z@(c*Jm)-V)/np.linalg.norm(V)):.6e}")

print()
print("=" * 78)
print("[G] 输入校验覆盖度(判据 I 只有 4 项)")
print("=" * 78)
def tryit(label, fn):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = fn()
        shape = getattr(out, "shape", None)
        print(f"  [NO-RAISE] {label:<52s} -> {type(out).__name__} {shape if shape is not None else ''}")
    except Exception as e:
        print(f"  [raised  ] {label:<52s} -> {type(e).__name__}: {str(e)[:70]}")

tryit("circle_geometry(nan, 10)", lambda: mom.circle_geometry(float("nan"), 10))
tryit("circle_geometry(inf, 10)", lambda: mom.circle_geometry(float("inf"), 10))
tryit("circle_geometry(1.0, 3.5) 非整数", lambda: mom.circle_geometry(1.0, 3.5)["n"])
tryit("circle_geometry(1.0, nan)", lambda: mom.circle_geometry(1.0, float("nan")))
tryit("circle_geometry(1.0, -1)", lambda: mom.circle_geometry(1.0, -1))
tryit("circle_geometry(1.0, 0)", lambda: mom.circle_geometry(1.0, 0))
tryit("assemble_impedance(nan, geom)", lambda: mom.assemble_impedance(float("nan"), mom.circle_geometry(1.0, 8)))
tryit("assemble_impedance(inf, geom)", lambda: mom.assemble_impedance(float("inf"), mom.circle_geometry(1.0, 8)))
tryit("assemble_impedance(k, geom, quad_order=0)", lambda: mom.assemble_impedance(1.0, mom.circle_geometry(1.0, 8), quad_order=0))
tryit("assemble_impedance(k, geom, quad_order=1)", lambda: mom.assemble_impedance(1.0, mom.circle_geometry(1.0, 8), quad_order=1))
tryit("assemble_excitation(nan k)", lambda: mom.assemble_excitation(float("nan"), mom.circle_geometry(1.0, 8)))
tryit("assemble_excitation(theta_i=nan)", lambda: mom.assemble_excitation(1.0, mom.circle_geometry(1.0, 8), theta_i=float("nan")))
tryit("surface_current(amp=0)", lambda: mom.surface_current(1.0, 1.0, 8, amplitude=0.0)["J"])
tryit("mie_surface_current(nan)", lambda: mom.mie_surface_current(float("nan"), 1.0, 0.0))
tryit("mie_scattered_field(rho<radius)", lambda: mom.mie_scattered_field(1.0, 5.0, 1.0, 0.0))
tryit("solve_dense 奇异矩阵", lambda: mom.solve_dense(np.zeros((3, 3)), np.ones(3)))
tryit("scattered_field_on_target 空点集", lambda: mom.scattered_field_on_target(1.0, mom.circle_geometry(1.0, 8), np.ones(8), np.zeros((0, 2))))

print()
print("  quad_order=0 的后果(静默错误,不抛异常):")
g8 = mom.circle_geometry(5.0, 8)
Za = mom.assemble_impedance(1.0, g8, quad_order=64)
Zb = mom.assemble_impedance(1.0, g8, quad_order=0)
print(f"    max|Z(64)-Z(0)| = {np.max(np.abs(Za-Zb)):.3e}   ||Z(0)||max={np.max(np.abs(Zb)):.3e}")
print(f"    Z(0) 的完全非对角元是否全为 0: {np.allclose(Zb - np.diag(np.diag(Zb)), 0)}")

print()
print("  amplitude=0 的后果:")
try:
    r0 = mom.surface_current(1.0, 1.0, 8, amplitude=0.0)
    print(f"    V 全零? {np.all(r0['V']==0)}   J 全零? {np.all(r0['J']==0)}   rel_resid={r0['rel_resid']}")
except Exception as e:
    print(f"    异常 {type(e).__name__}: {e}")
