# -*- coding: utf-8 -*-
"""第三方复核 脚本1:独立重算(全用 mpmath / scipy.integrate.quad,不调 quad.py 主路径)。"""
import os
import sys
import time

import numpy as np
import mpmath as mp
from scipy.integrate import quad as squad
from scipy.special import hankel2 as sp_hankel2, jv as sp_jv

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

import mom
import quad

mp.mp.dps = 30


# ---------------------------------------------------------------- 独立 Mie
def mie_J_ref(k, a, phi, theta_i=0.0, amp=1.0, M=None):
    """独立实现:逐点 mpmath 求和,不用 numpy 的 hankel2 广播。"""
    ka = mp.mpf(k) * mp.mpf(a)
    if M is None:
        M = int(mp.ceil(ka + 4 * ka ** (mp.mpf(1) / 3))) + 10
    out = []
    for p in np.atleast_1d(phi):
        s = mp.mpc(0)
        for n in range(-M, M + 1):
            s += (mp.mpc(0, 1) ** n) / mp.hankel2(n, ka) * mp.exp(mp.mpc(0, 1) * n * (mp.mpf(p) - mp.mpf(theta_i)))
        out.append(amp * 2 / (mp.pi * ka) * s)
    return out


def mie_Esca_ref(k, a, rho, phi, theta_i=0.0, amp=1.0, M=None):
    ka = mp.mpf(k) * mp.mpf(a)
    if M is None:
        M = int(mp.ceil(ka + 4 * ka ** (mp.mpf(1) / 3))) + 10
    s = mp.mpc(0)
    for n in range(-M, M + 1):
        s += (-(mp.mpc(0, 1) ** n)) * (mp.besselj(n, ka) / mp.hankel2(n, ka)) \
             * mp.hankel2(n, mp.mpf(k) * mp.mpf(rho)) * mp.exp(mp.mpc(0, 1) * n * (mp.mpf(phi) - mp.mpf(theta_i)))
    return amp * s


def mp_F(z):
    """独立算 F(z) = int_0^z H0^(2)(u) du,大 z 时按 2pi 分段。"""
    z = mp.mpf(z)
    f = lambda u: mp.hankel2(0, u)
    if z < 40:
        return mp.quad(f, [0, z])
    n = int(mp.floor(z / (2 * mp.pi)))
    pts = [2 * mp.pi * i for i in range(1, n + 1)] + [z]
    return mp.quad(f, [0] + pts)


print("=" * 78)
print("[1] 自耦合积分 I(k,a)=int_{-a}^{a}H0^(2)(k|t|)dt —— 基准: mpmath(30位)")
print("=" * 78)
print(f"{'k*a':>10} {'quad.self_coupling_integral':>34} {'rel|dI|/|I|':>14}  {'t':>5}")
bad1 = []
for ka in [1e-8, 1e-6, 1e-4, 1e-2, 0.3, 0.5, 0.5000001, 0.75, 1.0, 5.0, 20.0, 50.0, 100.0]:
    t0 = time.time()
    ref = 2 * mp_F(mp.mpf(ka))
    got = quad.self_coupling_integral(1.0, ka)
    rel = abs(mp.mpc(got) - ref) / abs(ref)
    print(f"{ka:>10g} {str(complex(got)):>34} {mp.nstr(rel,4):>14}  {time.time()-t0:5.1f}")
    if rel > mp.mpf('1e-11'):
        bad1.append((ka, mp.nstr(rel, 4)))

print("\n  大 k*a(分段 mpmath 积分较慢)专门再测:")
for ka in [200.0, 500.0]:
    t0 = time.time()
    ref = 2 * mp_F(mp.mpf(ka))
    got = quad.self_coupling_integral(1.0, ka)
    rel = abs(mp.mpc(got) - ref) / abs(ref)
    print(f"    ka={ka:<8g} relerr={mp.nstr(rel,4):<12} ({time.time()-t0:.0f}s)")
    if rel > mp.mpf('1e-11'):
        bad1.append((ka, mp.nstr(rel, 4)))

print("\n  hankel0_2 两分支在不同 x 上的一致性(mpmath 基准):")
maxrel = 0.0
for x in [1e-8, 1e-4, 1e-2, 0.1, 0.3, 0.499, 0.4999, 0.5, 0.5001, 0.6, 1.0, 5.0, 30.0]:
    got = mom.hankel0_2(x)
    ref = mp.hankel2(0, mp.mpf(x))
    rel = abs(mp.mpc(got) - ref) / abs(ref)
    maxrel = max(maxrel, float(rel))
    print(f"    x={x:<9g} relerr={mp.nstr(rel,4)}")
print(f"    -> max rel err over the branch boundary = {maxrel:.3e}")

print()
print("=" * 78)
print("[2] 全矩阵 Z 独立重算")
print("=" * 78)

# (2a) N=16 全矩阵,mpmath 逐元
N16 = 16
g16 = mom.circle_geometry(5.0, N16)
Z16 = mom.assemble_impedance(1.0, g16)
t0 = time.time()
Zref = np.zeros((N16, N16), dtype=complex)
for m in range(N16):
    for n in range(N16):
        mid_n = g16["mid"][n]; tan_n = g16["tangent"][n]; h = 0.5 * g16["length"][n]
        pm = g16["mid"][m]

        def integ(t):
            R = mp.sqrt(sum((mp.mpf(pm[i]) - (mp.mpf(mid_n[i]) + mp.mpf(t) * mp.mpf(tan_n[i]))) ** 2 for i in range(2)))
            return mp.hankel2(0, R)

        if m == n:
            pts = [-h, 0, h]
        else:
            pts = [-h, h]
        val = mp.quad(integ, pts)
        Zref[m, n] = 0.25 * val
print(f"  N=16 全矩阵 mpmath 逐元({time.time()-t0:.1f}s) max|dZ|/max|Z| = "
      f"{np.max(np.abs(Z16 - Zref))/np.max(np.abs(Zref)):.3e}   max abs diff = {np.max(np.abs(Z16-Zref)):.3e}")

# (2b) N=200 非对角元,scipy.integrate.quad
N = 200
g = mom.circle_geometry(5.0, N)
Z = mom.assemble_impedance(1.0, g)
worst = (0.0, None)
t0 = time.time()
for m in range(0, N, 7):
    pm = g["mid"][m]
    for n in range(N):
        if m == n:
            continue
        mid_n = g["mid"][n]; tan_n = g["tangent"][n]; h = 0.5 * g["length"][n]
        f = lambda t: sp_hankel2(0, np.linalg.norm(pm - (mid_n + t * tan_n)))
        val, err = squad(f, -h, h, limit=200, epsabs=1e-14, epsrel=1e-12)
        d = abs(0.25 * val - Z[m, n])
        if d > worst[0]:
            worst = (d, (m, n, val, err))
print(f"  N=200 非对角元抽样({time.time()-t0:.1f}s) max|dZ| = {worst[0]:.3e}  (element {worst[1][0]},{worst[1][1]})")
print(f"    quad 自带误差估计 = {worst[1][3]:.2e}; |Z| 典型量级 = {np.abs(Z).mean():.3e}")

print()
print("=" * 78)
print("[3] Mie 真值独立重算(mpmath 逐点求和)")
print("=" * 78)
K, RAD, N400 = 1.0, 5.0, 400
g400 = mom.circle_geometry(RAD, N400)
phi = g400["phi_mid"]
t0 = time.time()
Jm_mom = mom.mie_surface_current(K, RAD, phi)
Jm_ref = np.array([complex(v) for v in mie_J_ref(K, RAD, phi)], dtype=complex)
rel = np.linalg.norm(Jm_mom - Jm_ref) / np.linalg.norm(Jm_ref)
print(f"  mie_surface_current vs mpmath(逐点, N=400, {time.time()-t0:.1f}s): rel L2 = {rel:.3e}")
for rho in (10.0, 40.0, 150.0):
    e_mom = complex(mom.mie_scattered_field(K, RAD, rho, 0.0))
    e_ref = complex(mie_Esca_ref(K, RAD, rho, 0.0))
    print(f"  mie_scattered_field rho={rho:<6g}: mom={e_mom:+.12e}  mp={e_ref:+.12e}  rel={abs(e_mom-e_ref)/abs(e_ref):.3e}")

# 交叉:mie_scattered_field 与 mie_surface_current 的 Wronskian/边界一致性
print("  交叉检验(两条解析路径必须自洽): E_sca(a,phi) 应 = -E_inc(a,phi)")
gtest = mom.circle_geometry(5.0, 400)
ph = np.linspace(0, 2 * np.pi, 37)[:-1]
Einc = np.exp(1j * 1.0 * 5.0 * np.cos(ph))
Esca = mom.mie_scattered_field(1.0, 5.0, 5.0, ph)
print(f"    max|Esca+Einc|/max|Einc| = {np.max(np.abs(Esca+Einc))/np.max(np.abs(Einc)):.3e}")

print()
print("=" * 78)
print("[4] 作者判据关键数字独立重算")
print("=" * 78)
idx = np.arange(N)
A1 = np.max(np.abs(Z - Z[0, (idx[None, :] - idx[:, None]) % N]))
A2 = np.max(np.abs(Z - Z.T))
opp = N // 2
A4 = abs(Z[0, 0]) / abs(Z[0, opp])
print(f"  A1 = {A1:.6e}   (作者 2.115851e-15)")
print(f"  A2 = {A2:.6e}   (作者 2.314815e-15)")
print(f"  A4 = {A4:.6f}   (作者 10.050)")

res = mom.surface_current(K, RAD, N400)
J, Zb, V = res["J"], res["Z"], res["V"]
Jm = mom.mie_surface_current(K, RAD, res["geom"]["phi_mid"])
print(f"  B1 = {res['rel_resid']:.6e}   (作者 5.490031e-16)")
print(f"  B2 = {np.linalg.norm(Zb@Jm - V)/np.linalg.norm(V):.6e}   (作者 1.706524e-04)")
print(f"  C1 = {np.linalg.norm(J-Jm)/np.linalg.norm(Jm):.6e}   (作者 1.963859e-04)")

# F: 完整复数比值,不只看实部
pt = np.array([[10.0, 0.0], [40.0, 0.0], [150.0, 0.0]])
ej = mom.scattered_field_on_target(K, res["geom"], J, pt)
print("  F 的完整复数比值(作者只取 .real):")
for i, rho in enumerate((10.0, 40.0, 150.0)):
    ea = complex(mom.mie_scattered_field(K, RAD, rho, 0.0))
    q = ej[i] / ea
    print(f"    rho={rho:<6g} ratio={q:+.9f}  |ratio|={abs(q):.9f}  Re={q.real:.9f}  Im={q.imag:+.3e}")

print()
print("=" * 78)
print("[5] 判据 F 能否识别符号错:手工把重建场反号")
print("=" * 78)
for i, rho in enumerate((10.0, 40.0, 150.0)):
    ea = complex(mom.mie_scattered_field(K, RAD, rho, 0.0))
    q = -ej[i] / ea
    flag = "PASS(判据失效!)" if abs(q.real - 1.0) <= 1e-3 else "FAIL(判据正确报警)"
    print(f"    rho={rho:<6g} 反号后 Re ratio = {q.real:+.6f} -> {flag}")

print()
print("=" * 78)
print("[6] 判据 E 的成分分析:节点残差到底来自电流误差还是求积误差")
print("=" * 78)
rt = mom.total_field_residual(K, res["geom"], J)
print(f"  E1(用 MoM 解 J)         = {rt['max_abs']/rt['scale']:.6e}   max|E_tot|={rt['max_abs']:.6e}")
rt_mie = mom.total_field_residual(K, res["geom"], Jm)
print(f"  节点残差(用精确 Mie 电流)= {rt_mie['max_abs']/rt_mie['scale']:.6e}   max|E_tot|={rt_mie['max_abs']:.6e}")
print("   -> 若第二行是机器精度,说明 E1 反映的是 J 的误差;若不是,说明被求积/几何误差污染")
# on_nodes=False 路径
try:
    rt_c = mom.total_field_residual(K, res["geom"], J, on_nodes=False)
    print(f"  on_nodes=False(配点处,含 R=0 对角项) = {rt_c['max_abs']/rt_c['scale']:.6e}")
    print("    注:该路径在 R=0 处把核置 0(scattered_field_on_target 第 217-218 行),对角贡献被丢掉")
except Exception as e:
    print(f"  on_nodes=False 抛异常: {type(e).__name__}: {e}")
# 用配点处正确的重建(collocation 自洽性)
mid = res["geom"]["mid"]
Einc_mid = np.exp(1j * K * mid[:, 0])
E_sca_mid = mom.scattered_field_on_target(K, res["geom"], J, mid)
print(f"  scattered_field_on_target 在配点处 vs 直接 Z@J: "
      f"max|(-E_sca_mid) - (Zb@J)| = {np.max(np.abs(-E_sca_mid - Zb@J)):.3e}")

print()
print("=" * 78)
print("[7] 判据 H 的解释:theta_i 到底有没有被用上")
print("=" * 78)
print("  7a) 四个 theta_i 是否都是 2*pi/N 的整数倍(N=400 -> 步长 0.0157080):")
for th in (0.0, np.pi / 4, np.pi / 2, np.pi):
    q = th / (2 * np.pi / 400)
    print(f"    theta={th:.6f}  = {q:.6f} * (2pi/N)   整数倍? {abs(q-round(q))<1e-12}")
print("  7b) 非整数倍 theta 的误差(应当不同,且随 theta 变化):")
for th in (0.0, 0.003, 0.005, 0.00785, 0.007854*3, np.pi/4 + 0.007):
    r = mom.surface_current(K, RAD, 400, theta_i=th)
    jm = mom.mie_surface_current(K, RAD, r["geom"]["phi_mid"], theta_i=th)
    print(f"    theta={th:.8f}  rel_err={np.linalg.norm(r['J']-jm)/np.linalg.norm(jm):.12e}")
print("  7c) 手算一个配点的 V_m,核对 theta_i 真的进入激励:")
g3 = mom.circle_geometry(5.0, 12)
V3 = mom.assemble_excitation(1.0, g3, theta_i=0.7)
xm, ym = g3["mid"][3]
manual = np.exp(1j * 1.0 * (xm * np.cos(0.7) + ym * np.sin(0.7)))
print(f"    V_m(手算) = {manual:+.15e}   V_m(代码) = {V3[3]:+.15e}   diff={abs(manual-V3[3]):.2e}")
print(f"    V(theta=0.7)[3] vs V(theta=0)[3] 之差 = {abs(V3[3]-mom.assemble_excitation(1.0,g3,theta_i=0.0)[3]):.3e}  (非零 => 被用上)")
print("  7d) theta -> -theta 是否可被 H 判据区分(镜像对称):")
for th in (0.3, -0.3, 0.05, -0.05):
    r = mom.surface_current(K, RAD, 400, theta_i=th)
    jm = mom.mie_surface_current(K, RAD, r["geom"]["phi_mid"], theta_i=th)
    print(f"    theta={th:+.4f}  rel_err={np.linalg.norm(r['J']-jm)/np.linalg.norm(jm):.12e}   ||J||={np.linalg.norm(r['J']):.12e}")
