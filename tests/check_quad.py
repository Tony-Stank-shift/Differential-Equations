"""1.2 自检:奇异积分模块 quad.py。

验收判据(来自 WBS 1.2,并已按第三方复核的发现扩充):
    A. ``self_coupling_integral``(主路径)与 **mpmath 高精度数值积分真值**在
       8 组 (k, a) 上相对误差 < 1e-12,覆盖 ka 从 0.05 到 5
    A2. **大 ka 守护**(复核问题①):ka = 10 / 52 / 256 / 1000 / 1e4 时主路径
        (闭式)对 mpmath 高精度 Struve 实现 < 1e-12
    A3. **双实现交叉验证**:闭式(特殊函数)与级数(幂级数)在 ka <= 0.5 上一致
    B. ``log_term_integral`` / ``poly2_term_integral`` 两个闭式主部积分与 mpmath
       数值积分一致(< 1e-12)
    B3. 二阶系数 C0 / C1 的正确性:余项必须真的是 O(x^4 ln x) ——
       若 C0/C1 写错,|R2-1| 会退回 O(x^2)
    C. 分段路径(备用)在 ka = 5 时的稳定性
    C2. **回归**(复核问题①):分段路径在 ka 大时确实失效,而主路径不失效 ——
       把复核员的发现固化成测试,防止有人把分段路径改回主路径
    D. 两个错误做法的危害量化:
        D1 ``naive_self_coupling``(只拆对数主部,把余项当 1)
        D2 ``direct_gauss``(直接对奇异核求积)
    E. 形状/边界/微积分一致性(dI/da = 2 H_0^{(2)}(ka))
    F. **备用路径的真实断言**(复核问题②):ka <= 0.5 时 < 1e-12;并在 ka = 5
       时把已知劣化如实打印出来,不留"文档推荐但无人守护"的路径
    G. **浮点边界**(复核问题③):a = 0.5/k 及其 1e-15 邻域不抛无谓 ValueError

⚠️ 与 1.1 同一纪律(见 DECISIONS D8):判据必须用**独立真值**(mpmath)。
   mpmath 缺失时判据 A 记 FAIL,不静默跳过。

运行:
    python tests/check_quad.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from green import hankel0_2  # noqa: E402
from quad import (  # noqa: E402
    C0,
    C1,
    DEFAULT_QUAD_ORDER,
    analytic_part_integral,
    closed_self_integral,
    direct_gauss,
    gauss_legendre,
    hankel0_antiderivative,
    log_term_integral,
    naive_self_coupling,
    poly2_term_integral,
    regular_part,
    self_coupling_integral,
    series_self_integral,
    split_self_integral,
)

try:
    import mpmath as mp

    mp.mp.dps = 40
    HAVE_MPMATH = True
except ImportError:  # pragma: no cover
    HAVE_MPMATH = False

TOL_A = 1e-12
TOL_B = 1e-12
#: 复核员实测分段路径首次越过 1e-12 的 ka 约在 52
SPLIT_BREAK_KA = 52.0
failures: list[str] = []


def rel_err(a, b) -> float:
    a = complex(a)
    b = complex(b)
    denom = abs(b) if abs(b) > 0.0 else 1.0
    return abs(a - b) / denom


def report(name: str, err: float, tol: float) -> None:
    ok = err < tol
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<50s} rel_err = {err:.3e}  (tol {tol:.0e})")
    if not ok:
        failures.append(name)


def mp_self_coupling(k: float, a: float) -> complex:
    """mpmath 真值:∫_{-a}^{a} H_0^{(2)}(k|t|) dt(把 0 作为分段点)。

    ⚠️ ka 很大时 mp.quad 自身收敛困难;那时改用 ``mp_struve_closed``(见判据 A2)。
    """
    K = mp.mpf(repr(float(k)))
    A = mp.mpf(repr(float(a)))
    return complex(mp.quad(lambda t: mp.hankel2(0, K * abs(t)), [-A, 0, A]))


def mp_struve_closed(k: float, a: float) -> complex:
    """mpmath 高精度 Struve 闭式真值(判据 A2 用,可到任意大 ka)。

        F(z) = z H_0^{(2)}(z) + (pi z / 2)[ H_1^{(2)}(z) StrauveH_0(z)
                                              - H_0^{(2)}(z) StrauveH_1(z) ]
    """
    K = mp.mpf(repr(float(k)))
    z = K * mp.mpf(repr(float(a)))
    F = z * mp.hankel2(0, z) + (mp.pi * z / 2) * (
        mp.hankel2(1, z) * mp.struveh(0, z) - mp.hankel2(0, z) * mp.struveh(1, z)
    )
    return complex(2 / K * F)


def mp_log_term(k: float, a: float) -> complex:
    """mpmath 真值:∫_{-a}^{a} S(k|t|) dt。"""
    K = mp.mpf(repr(float(k)))
    A = mp.mpf(repr(float(a)))
    g = mp.euler

    def f(t):
        x = K * abs(t)
        return mp.mpf(0) if x == 0 else -(2j / mp.pi) * (mp.log(x / 2) + g)

    return complex(mp.quad(f, [-A, 0, A]))


def mp_poly2_term(k: float, a: float) -> complex:
    """mpmath 真值:∫_{-a}^{a} x^2 [C0 + C1 ln(x/2)] dt, x = k|t|。"""
    K = mp.mpf(repr(float(k)))
    A = mp.mpf(repr(float(a)))
    c0 = mp.mpf(str(C0.real)) + 1j * mp.mpf(str(C0.imag))
    c1 = mp.mpf(str(C1.real)) + 1j * mp.mpf(str(C1.imag))

    def f(t):
        x = K * abs(t)
        if x == 0:
            return mp.mpf(0)
        return x**2 * (c0 + c1 * mp.log(x / 2))

    return complex(mp.quad(f, [-A, 0, A]))


print("=" * 86)
print("1.2 奇异积分自检")
print("=" * 86)
print(f"  mpmath 可用: {HAVE_MPMATH}" + (f" (dps={mp.mp.dps})" if HAVE_MPMATH else ""))
print(f"  C0 = {C0:.12g}   C1 = {C1:.12g}")

#: ka 从 0.05 到 5 —— 覆盖复核员复算过的全部原算例
CASES = [
    (1.0, 0.05),
    (1.0, 0.1),
    (1.0, 0.5),
    (1.0, 1.0),
    (2.0, 0.3),
    (5.0, 0.1),
    (0.5, 2.0),
    (10.0, 0.5),
]

#: 大 ka 守护(复核问题①) —— 主路径必须全域有效
BIG_CASES = [(1.0, 10.0), (1.0, 52.0), (1.0, 256.0), (1.0, 1000.0), (1.0, 1e4)]

# ======================================================= [1] 判据 A
print("\n[1] 判据 A —— self_coupling_integral(主路径)vs mpmath 高精度积分")
if HAVE_MPMATH:
    worst_a = 0.0
    for k, a in CASES:
        got = self_coupling_integral(k, a)
        ref = mp_self_coupling(k, a)
        e = rel_err(got, ref)
        worst_a = max(worst_a, e)
        path = "级数" if k * a <= 0.5 else "闭式"
        print(
            f"         k={k:<5g} a={a:<5g} ka={k * a:<7.3g} [{path}]  "
            f"ours={got.real:+.12f}{got.imag:+.12f}i  rel_err={e:.3e}"
        )
    report(f"{len(CASES)} 组 (k, a) 上的最大相对误差", worst_a, TOL_A)
else:
    print("  [FAIL] 未安装 mpmath —— 判据 A 无法执行")
    failures.append("判据 A 未执行")

# ======================================================= [2] 判据 A2
print("\n[2] 判据 A2 —— 大 ka 守护(复核问题①):主路径 vs mpmath 高精度 Struve 实现")
print("         mp.quad 在高 ka 收敛困难,故这里用 mpmath 的 hankel2/struveh 高精度复算")
if HAVE_MPMATH:
    worst_a2 = 0.0
    for k, a in BIG_CASES:
        got = self_coupling_integral(k, a)
        ref = mp_struve_closed(k, a)
        e = rel_err(got, ref)
        worst_a2 = max(worst_a2, e)
        print(
            f"         ka={k * a:<9.4g} ours={got.real:+.12g}{got.imag:+.12g}i  "
            f"rel_err={e:.3e}"
        )
    report("ka = 10 / 52 / 256 / 1000 / 1e4", worst_a2, TOL_A)

    # 交叉印证:公式本身在小 ka 处必须同时命中 mp.quad(独立积分)
    e_formula = max(
        rel_err(mp_struve_closed(k, a), mp_self_coupling(k, a))
        for k, a in [(1.0, 0.5), (2.0, 0.3), (10.0, 0.5)]
    )
    report("闭式公式本身 vs mp.quad(小 ka,验证公式而非实现)", e_formula, 1e-13)
else:
    report("大 ka 守护", float("nan"), TOL_A)

# ======================================================= [3] 判据 A3
print("\n[3] 判据 A3 —— 双实现交叉验证:闭式(特殊函数)vs 级数(幂级数)")
if HAVE_MPMATH:
    worst_a3 = 0.0
    for x in (0.01, 0.05, 0.1, 0.2, 0.35, 0.5):
        e = rel_err(series_self_integral(1.0, x), closed_self_integral(1.0, x))
        worst_a3 = max(worst_a3, e)
        print(f"         ka={x:<6g} 闭式 rel_err={e:.3e}")
    report("ka = 0.01 ... 0.5 两条独立实现的一致性", worst_a3, TOL_A)

    # 判据 A2(旧编号)的级数单点验证
    worst_ser = max(
        rel_err(series_self_integral(1.0, x), mp_self_coupling(1.0, x))
        for x in (0.01, 0.05, 0.1, 0.2, 0.35, 0.5)
    )
    report("series_self_integral 单独 vs mpmath", worst_ser, TOL_A)

    try:
        series_self_integral(1.0, 0.6)
        print("  [FAIL] ka > 0.5 未抛 ValueError")
        failures.append("series range guard")
    except ValueError:
        print("  [PASS] ka > 0.5 抛出 ValueError(不会静默给出坏结果)")

    try:
        self_coupling_integral(1.0, 0.6, method="series")
        print("  [FAIL] method='series' 超出范围未抛 ValueError")
        failures.append("series method guard")
    except ValueError:
        print("  [PASS] method='series' 超出范围抛 ValueError")

    vals = hankel0_antiderivative(np.array([0.0, 0.5, 2.0]))
    print(f"         hankel0_antiderivative 向量化形状 = {vals.shape},F(0) = {vals[0]}")
    if vals.shape != (3,) or vals[0] != 0.0:
        print("  [FAIL] hankel0_antiderivative 形状或 F(0) 不对")
        failures.append("antiderivative vectorization")
else:
    report("双实现交叉验证", float("nan"), TOL_A)

# ======================================================= [4] 判据 B
print("\n[4] 判据 B —— 两个闭式主部积分 vs mpmath")
if HAVE_MPMATH:
    worst_l = max(rel_err(log_term_integral(k, a), mp_log_term(k, a)) for k, a in CASES)
    report("log_term_integral(对数主部)", worst_l, TOL_B)
    worst_p = max(rel_err(poly2_term_integral(k, a), mp_poly2_term(k, a)) for k, a in CASES)
    report("poly2_term_integral(二阶主部)", worst_p, TOL_B)
    worst_s = max(
        rel_err(analytic_part_integral(k, a), mp_log_term(k, a) + mp_poly2_term(k, a))
        for k, a in CASES
    )
    report("analytic_part_integral(两者之和)", worst_s, TOL_B)
else:
    report("闭式主部积分", float("nan"), TOL_B)

# ======================================================= [5] 判据 B3
print("\n[5] 判据 B3 —— 二阶系数 C0 / C1 的正确性(余项必须是 O(x^4 ln x))")
print("         若 C0/C1 写错,|R2-1| 会退回 O(x^2),最后一列会发散")
gate_b3 = True
prev_quad_ratio = None
for x in (0.4, 0.2, 0.1, 0.05, 0.02):
    r2 = complex(np.asarray(regular_part(1.0, x)).ravel()[0])
    dev = abs(r2 - 1.0)
    scale = x**4 * abs(float(np.log(x / 2.0)))
    ratio = dev / scale if scale > 0 else float("nan")
    quad_ratio = dev / (x * x)
    print(
        f"         x={x:<6g} |R2-1| = {dev:.4e}   x^4|ln(x/2)| = {scale:.4e}   "
        f"比值 = {ratio:8.4f}   |R2-1|/x^2 = {quad_ratio:.4e}"
    )
    # 判据一:与 x^4|ln(x/2)| 的比值保持 O(1)(不能随 x -> 0 发散)
    if not (1e-3 < ratio < 1.0):
        gate_b3 = False
    # 判据二:|R2-1|/x^2 必须随 x 减小而单调下降(真含 O(x^2) 项时会趋于常数)
    if prev_quad_ratio is not None and quad_ratio >= prev_quad_ratio:
        gate_b3 = False
    prev_quad_ratio = quad_ratio
if gate_b3:
    print("  [PASS] 比值有界且 |R2-1|/x^2 单调趋零 -> 余项确为 O(x^4 ln x),C0/C1 正确")
else:
    print("  [FAIL] 余项仍含 O(x^2) 项 —— C0 或 C1 有误")
    failures.append("poly2 coefficients")

# ======================================================= [6] 判据 C
print("\n[6] 判据 C —— 分段路径(备用)的稳定性(ka = 5,改变外段 Gauss 阶数)")
if HAVE_MPMATH:
    k_c, a_c = 10.0, 0.5  # ka = 5,delta = 0.05,外段占绝大部分
    ref_c = mp_self_coupling(k_c, a_c)
    vals = []
    for order in (16, 32, 64, 128):
        e = rel_err(split_self_integral(k_c, a_c, order=order), ref_c)
        vals.append(e)
        print(f"         order={order:<4d} rel_err={e:.3e}")
    print(f"         各阶数结果相对散布 = {max(vals) - min(vals):.3e}")
    print("         外段 [delta, a] 在 ka=5 时容纳约 4.5 弧度相位,16 阶不够是正常的")
    print("         收敛行为;32 阶起已到机器精度,128 阶的回升是浮点噪声而非缺陷")
    if max(vals[1:]) < 1e-12:
        print("  [PASS] 阶数 >= 32 时结果 < 1e-12")
    else:
        print("  [FAIL] 分段结果依赖阶数")
        failures.append("segment stability")
else:
    report("分段稳定性", float("nan"), TOL_A)

# ======================================================= [7] 判据 C2(回归)
print("\n[7] 判据 C2 —— 回归(复核问题①):分段路径在 ka 大时失效,主路径不失效")
print("         这条测试的意义:防止有人把分段路径改回主路径")
if HAVE_MPMATH:
    print("         ka       split(order=64)      closed(主路径)      mpmath 高精度")
    split_worst_big = 0.0
    closed_worst_big = 0.0
    for k, a in BIG_CASES:
        ref = mp_struve_closed(k, a)
        e_split = rel_err(split_self_integral(k, a, order=64), ref)
        e_closed = rel_err(closed_self_integral(k, a), ref)
        split_worst_big = max(split_worst_big, e_split)
        closed_worst_big = max(closed_worst_big, e_closed)
        print(
            f"         {k * a:<9.4g} {e_split:<20.3e} {e_closed:<20.3e}"
        )
    if split_worst_big > TOL_A:
        print(
            f"  [PASS] 分段路径在大 ka 处确实失效(最差 {split_worst_big:.3e}),"
            "与第三方复核一致;主路径同期 < 1e-12"
        )
    else:
        print("  [FAIL] 分段路径在大 ka 处居然没失效 —— 请复核这条回归的意义")
        failures.append("split-breakage regression")
    if closed_worst_big >= TOL_A:
        print("  [FAIL] 主路径在大 ka 处越线")
        failures.append("closed path at large ka")
else:
    report("split-breakage 回归", float("nan"), TOL_A)

# ======================================================= [8] 判据 D
print("\n[8] 判据 D —— 错误做法的危害量化(证明必须处理奇异性)")
if HAVE_MPMATH:
    print("         D1 只拆对数主部、把余项当常数 1(naive_self_coupling):")
    e_naive = 0.0
    for k, a in CASES:
        e = rel_err(naive_self_coupling(k, a), mp_self_coupling(k, a))
        e_naive = max(e_naive, e)
        print(f"            k={k:<5g} a={a:<5g} ka={k * a:<7.3g} rel_err={e:.3e}")
    print(f"          → 最差 {e_naive:.3e}(远大于 {TOL_A:.0e},故本模块不使用该近似)")
    if e_naive < TOL_A:
        print("  [FAIL] 该近似居然也满足判据 A —— 与门槛结论不符")
        failures.append("naive approximation unexpectedly good")
    else:
        print("  [PASS] 错误做法的误差被如实暴露")

    print("         D2 直接对奇异核求积(direct_gauss,不拆主部):")
    for k, a in [(1.0, 0.05), (1.0, 0.5), (1.0, 1.0)]:
        ref = mp_self_coupling(k, a)
        got = direct_gauss(k, a)
        print(
            f"            k={k:<4g} a={a:<5g} ours={got.real:+.10f}{got.imag:+.10f}i "
            f"ref={ref.real:+.10f}{ref.imag:+.10f}i rel_err={rel_err(got, ref):.3e}"
        )
    print("          → 对数奇异让 t=0 附近的节点采到发散值,误差停在 1e-2~1e-3 量级;")
    print("            提高阶数只能挤掉光滑部分的误差,奇异部分永远啃不动")

# ======================================================= [9] 判据 E
print("\n[9] 判据 E —— 形状、边界与微积分一致性")
nodes, weights = gauss_legendre(8)
print(f"         order=8: sum(weights) = {np.sum(weights):.15f} (应为 2)")
sym = float(np.max(np.abs(nodes + nodes[::-1])))
print(f"                  节点对称性 max|t_i + t_(n+1-i)| = {sym:.3e}")

t_arr = np.array([[-0.3, 0.0], [0.1, 0.25]])
reg = regular_part(2.0, t_arr)
print(f"         regular_part 形状 {reg.shape} (输入 {t_arr.shape})")
if reg.shape != t_arr.shape:
    print("  [FAIL] regular_part 形状不保持")
    failures.append("regular_part shape")

r0 = complex(np.asarray(regular_part(2.0, 0.0)).ravel()[0])
print(f"         regular_part(k, 0) = {r0} (应恰为 1+0j)")
if abs(r0 - 1.0) > 0.0:
    print("  [FAIL] t=0 未返回精确的极限值 1")
    failures.append("regular_part at zero")

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    self_coupling_integral(1.0, 0.1)
    self_coupling_integral(10.0, 0.5)
    self_coupling_integral(1.0, 1000.0)  # 大 ka 正当用法,不应告警
    regular_part(1.0, np.array([-0.1, 0.0, 0.1]))
    clean = [str(c.message) for c in caught]
print(f"         正当用法产生的告警数 = {len(clean)}")
if not clean:
    print("  [PASS] 正当用法无告警")
else:
    print(f"  [FAIL] 正当用法误告警: {clean}")
    failures.append("false alarm")

for bad in (0.0, -0.1):
    try:
        self_coupling_integral(1.0, bad)
        print(f"  [FAIL] half_len={bad} 未抛异常")
        failures.append("half_len guard")
    except ValueError:
        pass
print("  [PASS] half_len <= 0 抛出 ValueError")

for bad_k in (0.0, -1.0):
    try:
        self_coupling_integral(bad_k, 1.0)
        print(f"  [FAIL] k={bad_k} 未抛异常")
        failures.append("k guard")
    except ValueError:
        pass
print("  [PASS] k <= 0 抛出 ValueError")

try:
    self_coupling_integral(1.0, 1.0, method="nonsense")
    print("  [FAIL] 未知 method 未抛异常")
    failures.append("method guard")
except ValueError:
    print("  [PASS] 未知 method 抛出 ValueError")

if HAVE_MPMATH:
    k_d, a_d, h = 1.5, 0.4, 1e-6
    num = (closed_self_integral(k_d, a_d + h) - closed_self_integral(k_d, a_d - h)) / (
        2 * h
    )
    ana = 2.0 * complex(hankel0_2(k_d * a_d))
    report("dI/da = 2 H_0^(2)(ka) (微积分一致性,主路径)", rel_err(num, ana), 1e-8)

    # 三条路径在同一点上给同一答案(ka = 0.4,三条都走得通)
    k_x, a_x = 1.0, 0.4
    ref_x = mp_self_coupling(k_x, a_x)
    final = self_coupling_integral(k_x, a_x, method="closed")
    ser = self_coupling_integral(k_x, a_x, method="series")
    spl = self_coupling_integral(k_x, a_x, method="split")
    print(
        f"         三条路径(k={k_x}, a={a_x}):闭式 rel_err={rel_err(final, ref_x):.3e}  "
        f"级数 rel_err={rel_err(ser, ref_x):.3e}  分段 rel_err={rel_err(spl, ref_x):.3e}"
    )

# ======================================================= [10] 判据 F
print("\n[10] 判据 F —— 备用路径的真实断言(复核问题②)+ 已知劣化如实记录")
if HAVE_MPMATH:
    print("         (备用路径 = analytic_part_integral + GL(regular_part))")
    worst_f_ok = 0.0
    for k, a in [(1.0, 0.05), (1.0, 0.1), (1.0, 0.3), (2.0, 0.25), (1.0, 0.5)]:
        nd, wt = gauss_legendre(DEFAULT_QUAD_ORDER)
        via = analytic_part_integral(k, a) + a * complex(np.sum(wt * regular_part(k, a * nd)))
        ref = mp_self_coupling(k, a)
        e = rel_err(via, ref)
        worst_f_ok = max(worst_f_ok, e)
        print(f"         ka={k * a:<6.3g} order=64 rel_err={e:.3e}")
    report("备用路径在 ka <= 0.5 上(有断言)", worst_f_ok, TOL_A)

    print("         已知劣化(不设门,只如实记录 —— docstring 已写明适用范围):")
    for k, a in [(1.0, 1.0), (1.0, 5.0)]:
        ref = mp_self_coupling(k, a)
        for order in (64, 512):
            nd, wt = gauss_legendre(order)
            via = analytic_part_integral(k, a) + a * complex(
                np.sum(wt * regular_part(k, a * nd))
            )
            print(
                f"            ka={k * a:<5g} order={order:<4d} rel_err={rel_err(via, ref):.3e}"
            )
    print("         → 误差全在 ∫R2 的 GL 上(主部解析式 ~1e-16):R2-1 = O(t^4 ln|t|)")
    print("           的分支点在区间内部,只能代数收敛 n^-5,不是端点奇性")
else:
    report("备用路径断言", float("nan"), TOL_A)

# ======================================================= [11] 判据 G
print("\n[11] 判据 G —— 浮点边界 a = 0.5/k 及其邻域(复核问题③)")
gate_g = True
for k in (1.0, 2.0, 0.5, 1e-6, 1e6):
    for label, scale in (
        ("a = 0.5/k", 1.0),
        ("a*(1+1e-15)", 1.0 + 1e-15),
        ("a*(1+1e-16)", 1.0 + 1e-16),
    ):
        a = 0.5 / k * scale
        for method in ("auto", "closed", "split"):
            try:
                self_coupling_integral(k, a, method=method)
            except ValueError as exc:
                gate_g = False
                print(f"  [FAIL] k={k:g} {label} method={method} -> ValueError: {exc}")
if gate_g:
    print("  [PASS] 5 个 k × 3 个边界缩放 × 3 个 method = 45 次调用,无无谓 ValueError")
else:
    failures.append("float boundary at a = 0.5/k")

print("\n" + "=" * 86)
if failures:
    print(f"结果: {len(failures)} 项未通过 -> {failures}")
    raise SystemExit(1)
print("结果: 全部通过")
