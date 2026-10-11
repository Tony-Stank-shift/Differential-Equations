# -*- coding: utf-8 -*-
"""第三方复核 脚本3:参数空间扫描 + 失效边界 + 判据 D/H 的机理验证。"""
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

import mom

K, RAD = 1.0, 5.0


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def err_of(J, phi_mid, k, a, th=0.0):
    return rel(J, mom.mie_surface_current(k, a, phi_mid, theta_i=th))


print("=" * 78)
print("[1] 判据 D:O(h^2) 是不是常数因子的巧合?扩展到 N=1600 + 非均匀网格")
print("=" * 78)
print("  先验证 quad_order=16 与 64 在 N=400 上是否等价(以便大 N 提速):")
g400 = mom.circle_geometry(RAD, 400)
Za = mom.assemble_impedance(K, g400, quad_order=64)
Zb = mom.assemble_impedance(K, g400, quad_order=16)
print(f"    max|Z(64)-Z(16)|/max|Z| = {np.max(np.abs(Za-Zb))/np.max(np.abs(Za)):.3e}")
Ja, _ = mom.solve_dense(Za, mom.assemble_excitation(K, g400))
Jb, _ = mom.solve_dense(Zb, mom.assemble_excitation(K, g400))
print(f"    rel(J(64)-J(16)) = {rel(Jb,Ja):.3e}")

prev = None
print(f"  {'N':>6} {'rel_err':>14} {'ratio':>8} {'h':>10} {'h=2Rsin(pi/N)':>12}")
for n in (50, 100, 200, 400, 800, 1600):
    qo = 64 if n <= 400 else 16
    r = mom.surface_current(K, RAD, n, quad_order=qo)
    e = err_of(r["J"], r["geom"]["phi_mid"], K, RAD)
    h = 2 * RAD * np.sin(np.pi / n)
    print(f"  {n:>6} {e:>14.6e} {(prev/e if prev else 0):>8.4f} {h:>10.3e} {h:>12.3e}")
    prev = e
print("  ^ ratio 单调趋近 4 => 真二阶;若是常数因子巧合不会连续 5 次都 ~4")

print()
print("  非均匀(分级)网格 —— 代码本身只支持圆等分,这里自己造 geom 字典喂给装配:")
def geom_from_phi(R, phi_edges):
    edges = R * np.column_stack([np.cos(phi_edges), np.sin(phi_edges)])
    p0, p1 = edges[:-1], edges[1:]
    seg = p1 - p0
    L = np.linalg.norm(seg, axis=1)
    return {"edges": edges, "mid": 0.5 * (p0 + p1), "tangent": seg / L[:, None],
            "length": L, "phi_mid": 0.5 * (phi_edges[:-1] + phi_edges[1:]),
            "n": int(len(L)), "radius": float(R)}

beta = 0.6
def graded_phi(N):
    t = np.linspace(0.0, 1.0, N + 1)
    return 2 * np.pi * (t + beta * np.sin(2 * np.pi * t) / (2 * np.pi))

prev = None
print(f"  {'N':>6} {'rel_err':>14} {'ratio':>8} {'h_max':>10} {'h_min':>10}")
for n in (50, 100, 200, 400, 800):
    ph = graded_phi(n)
    gm = geom_from_phi(RAD, ph)
    Zn = mom.assemble_impedance(K, gm, quad_order=64)
    Vn = mom.assemble_excitation(K, gm)
    Jn, _ = mom.solve_dense(Zn, Vn)
    e = err_of(Jn, gm["phi_mid"], K, RAD)
    print(f"  {n:>6} {e:>14.6e} {(prev/e if prev else 0):>8.4f} "
          f"{gm['length'].max():>10.3e} {gm['length'].min():>10.3e}")
    prev = e
print("  ^ 分级网格同样 ~4 倍 => O(h^2) 不是规则网格的巧合")

print()
print("  几何项 vs 格式项的量级分离:把 Mie 真值的半径换成弦中点半径 R*cos(pi/N)")
for n in (100, 200, 400, 800):
    r = mom.surface_current(K, RAD, n)
    e_R = err_of(r["J"], r["geom"]["phi_mid"], K, RAD)
    e_Re = err_of(r["J"], r["geom"]["phi_mid"], K, RAD * np.cos(np.pi / n))
    shift = rel(mom.mie_surface_current(K, RAD * np.cos(np.pi / n), r["geom"]["phi_mid"]),
                mom.mie_surface_current(K, RAD, r["geom"]["phi_mid"]))
    print(f"    N={n:<5d} err(Mie@R)={e_R:.4e}  err(Mie@Rcos)={e_Re:.4e}  "
          f"||Jm(Rcos)-Jm(R)||/||Jm(R)||={shift:.4e}")
print("  ^ 若 err(Mie@Rcos) 显著更小,说明几何半径偏差是主因;否则格式误差为主")

print()
print("=" * 78)
print("[2] k*Delta 扫描:每段电长度多大时方法失效(固定 ka=5, a=5, k=1)")
print("=" * 78)
print(f"  {'N':>5} {'k*Delta':>10} {'Delta/lambda':>13} {'rel_err':>12} {'|Im Z_mm/Z_mm|':>16} {'cond(Z)':>10}")
for n in (4, 6, 8, 10, 12, 16, 20, 30, 40, 60, 100, 200, 400):
    r = mom.surface_current(K, RAD, n)
    d = np.mean(r["geom"]["length"])
    kd = K * d
    e = err_of(r["J"], r["geom"]["phi_mid"], K, RAD)
    zmm = r["Z"][0, 0]
    try:
        c = np.linalg.cond(r["Z"])
    except Exception:
        c = float("nan")
    print(f"  {n:>5} {kd:>10.4f} {d/(2*np.pi/K):>13.4f} {e:>12.3e} "
          f"{zmm.imag/abs(zmm):>16.4f} {c:>10.3e}")
print()
print("  同一扫描改成固定 a=1, ka=1(更小电尺寸),看阈值是否随 ka 变:")
for n in (8, 12, 16, 24, 32, 48, 64, 128):
    r = mom.surface_current(1.0, 1.0, n)
    d = np.mean(r["geom"]["length"])
    e = err_of(r["J"], r["geom"]["phi_mid"], 1.0, 1.0)
    print(f"    N={n:<5d} k*Delta={1.0*d:>8.4f}  rel_err={e:>12.3e}")

print()
print("=" * 78)
print("[3] Mie 级数截断:默认 n_terms = ceil(ka+4ka^(1/3))+10,与 2M 对比")
print("=" * 78)
print(f"  {'ka':>7} {'M(默认)':>8} {'rel(surface M vs 2M)':>22} {'rel(Esca M vs 2M)':>20}")
for ka_ in (0.5, 1.0, 5.0, 20.0, 100.0, 500.0, 1000.0):
    a_ = ka_
    M = int(np.ceil(ka_ + 4 * ka_ ** (1 / 3))) + 10
    phi_s = np.linspace(0, 2 * np.pi, 37)[:-1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        J1 = mom.mie_surface_current(1.0, a_, phi_s, n_terms=M)
        J2 = mom.mie_surface_current(1.0, a_, phi_s, n_terms=2 * M)
        E1 = mom.mie_scattered_field(1.0, a_, 20.0 * a_, phi_s, n_terms=M)
        E2 = mom.mie_scattered_field(1.0, a_, 20.0 * a_, phi_s, n_terms=2 * M)
    print(f"  {ka_:>7g} {M:>8d} {rel(J1,J2):>22.3e} {rel(E1,E2):>20.3e}")
print("  ^ 若 rel 随 ka 上升/爆掉,说明默认截断在大 ka 不够")

print()
print("=" * 78)
print("[4] 电小尺寸 ka=1e-3: 病态?条件数多少?解对不对?")
print("=" * 78)
for ka_ in (1e-1, 1e-2, 1e-3, 1e-4, 1e-6):
    for n in (64, 256):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = mom.surface_current(1.0, ka_, n)
            e = err_of(r["J"], r["geom"]["phi_mid"], 1.0, ka_)
            c = np.linalg.cond(r["Z"])
            zmm = r["Z"][0, 0]
        print(f"  ka={ka_:<8g} N={n:<5d} cond(Z)={c:>10.3e}  rel_err={e:>12.3e}  "
              f"|J|max={np.abs(r['J']).max():>10.3e}  |V|max={np.abs(r['V']).max():.3f}  "
              f"Im(Zmm)/|Zmm|={zmm.imag/abs(zmm):+.4f}")

print()
print("=" * 78)
print("[5] 极端 N:N=3,4,5,6,1000")
print("=" * 78)
for n in (3, 4, 5, 6, 7, 1000):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            qo = 64 if n < 1000 else 16
            r = mom.surface_current(K, RAD, n, quad_order=qo)
            e = err_of(r["J"], r["geom"]["phi_mid"], K, RAD)
            c = np.linalg.cond(r["Z"])
            print(f"  N={n:<5d} cond={c:>10.3e}  rel_err={e:>12.3e}  resid={r['rel_resid']:.2e}")
        except Exception as ex:
            print(f"  N={n:<5d} 异常 {type(ex).__name__}: {ex}")

print()
print("=" * 78)
print("[6] 判据 H 的机理:Z 是循环矩阵 => DFT 对角化 => 误差范数对 theta 严格无关")
print("=" * 78)
N = 400
g = mom.circle_geometry(RAD, N)
Z = mom.assemble_impedance(K, g)
phi = g["phi_mid"]
print("  6a) 把 MoM 解与 Mie 真值都做 DFT,看比值是否只依赖谐波序号 p(与 theta 无关):")
for th in (0.0, 0.3, 0.77):
    V = mom.assemble_excitation(K, g, theta_i=th)
    J, _ = mom.solve_dense(Z, V)
    Jm = mom.mie_surface_current(K, RAD, phi, theta_i=th)
    Jh, Jmh = np.fft.fft(J) / N, np.fft.fft(Jm) / N
    e = rel(J, Jm)
    print(f"    theta={th:<6g} rel_err={e:.15e}")
    if th == 0.0:
        P = 8
        idxs = list(range(-P, P + 1))
        r = [Jh[p % N] / Jmh[p % N] for p in range(-P, 0)] + [Jh[p] / Jmh[p] for p in range(0, P + 1)]
        print("      p 与 (J_mom/J_mie) 的 DFT 比值:", " ".join(f"{p}:{v.real:.6f}" for p, v in zip(idxs, r)))
print("  6b) 把激励换成'错误的 theta 映射'(theta->theta-pi/2),H 是否变红:")
for th in (0.0, 0.3, np.pi / 2, np.pi):
    Vbad = mom.assemble_excitation(K, g, theta_i=th - np.pi / 2)
    Jbad, _ = mom.solve_dense(Z, Vbad)
    Jm = mom.mie_surface_current(K, RAD, phi, theta_i=th)
    print(f"    theta={th:+.4f}  rel_err(错映射)={rel(Jbad, Jm):.6e}   (tol 5e-4)")
print("  6c) 激励整体乘常数(只错幅值)会不会被抓:")
for c in (1.0001, 1.01, 1.5, -1.0):
    Jc, _ = mom.solve_dense(Z, c * mom.assemble_excitation(K, g))
    Jm = mom.mie_surface_current(K, RAD, phi)
    print(f"    c={c:<8g} rel_err={rel(Jc, Jm):.6e}")

print()
print("=" * 78)
print("[7] 对称入射角(pi/2, pi)的独立手算校验")
print("=" * 78)
g12 = mom.circle_geometry(5.0, 12)
for th in (0.0, np.pi / 4, np.pi / 2, np.pi, -np.pi / 2):
    V = mom.assemble_excitation(K, g12, theta_i=th)
    xm, ym = g12["mid"][:, 0], g12["mid"][:, 1]
    Vman = np.exp(1j * K * (xm * np.cos(th) + ym * np.sin(th)))
    r = mom.surface_current(K, RAD, 400, theta_i=th)
    Jm = mom.mie_surface_current(K, RAD, r["geom"]["phi_mid"], theta_i=th)
    print(f"  theta={th:+.6f}  V 手算-代码 max|diff|={np.max(np.abs(V-Vman)):.2e}   "
          f"rel_err={rel(r['J'],Jm):.6e}  resid={r['rel_resid']:.2e}")

print()
print("=" * 78)
print("[8] quad_order 敏感性(装配求积是否收敛)与 amplitude=0")
print("=" * 78)
for n, qo in ((400, 8), (400, 16), (400, 32), (400, 64), (400, 128), (800, 16), (800, 64)):
    r = mom.surface_current(K, RAD, n, quad_order=qo)
    e = err_of(r["J"], r["geom"]["phi_mid"], K, RAD)
    print(f"  N={n:<5d} quad_order={qo:<4d} rel_err={e:.6e}")
r0 = mom.surface_current(1.0, 1.0, 8, amplitude=0.0)
print(f"  amplitude=0: V 全零={np.all(r0['V']==0)}  J 全零={np.all(r0['J']==0)}  "
      f"rel_resid={r0['rel_resid']}  (0/0 => nan?)")

print()
print("=" * 78)
print("[9] 判据 E 的收敛:节点残差随 N 是否 O(h^2)(说明它是离散化地板而非插值质量)")
print("=" * 78)
Jm = None
for n in (50, 100, 200, 400):
    r = mom.surface_current(K, RAD, n)
    jm = mom.mie_surface_current(K, RAD, r["geom"]["phi_mid"])
    rt = mom.total_field_residual(K, r["geom"], r["J"])
    rt2 = mom.total_field_residual(K, r["geom"], jm)
    e = rel(r["J"], jm)
    print(f"  N={n:<5d} E1(J_mom)={rt['max_abs']/rt['scale']:.4e}  "
          f"E1(J_mie)={rt2['max_abs']/rt2['scale']:.4e}  电流误差={e:.4e}  "
          f"E2={rt['max_abs']/rt['scale']/e:.3f}")
