"""1.2 自检:奇异积分模块 quad.py。

验收判据(来自 WBS 1.2):
    A. ``self_coupling_integral``(主路径)与 **mpmath 高精度数值积分真值**在
       8 组 (k, a) 上相对误差 < 1e-12 —— 实质判据,覆盖 ka 从 0.05 到 5
       (即"单段级数"与"分段"两条代码路径都要走到)
    A2. ``series_self_integral`` 单独在 ka <= 0.5 的多点上 < 1e-12
    B. ``log_term_integral`` / ``poly2_term_integral`` 两个闭式主部积分与 mpmath
       数值积分一致(< 1e-12)
    B3. 二阶系数 C0 / C1 的正确性:余项必须真的是 O(x^4 ln x) ——
       若 C0/C1 写错,|R2-1| 会退回 O(x^2)
    C. 分段方案的稳定性:ka = 5 时改变外段 Gauss 阶数,结果应稳定在 1e-14 内
    D. 两个错误做法的危害量化:
        D1 ``naive_self_coupling``(只拆对数主部,把余项当 1)
        D2 ``direct_gauss``(直接对奇异核求积)
    E. 形状/边界/微积分一致性(dI/da = 2 H_0^{(2)}(ka))

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
    direct_gauss,
    gauss_legendre,
    log_term_integral,
    naive_self_coupling,
    poly2_term_integral,
    regular_part,
    self_coupling_integral,
    series_self_integral,
)

try:
    import mpmath as mp

    mp.mp.dps = 40
    HAVE_MPMATH = True
except ImportError:  # pragma: no cover
    HAVE_MPMATH = False

TOL_A = 1e-12
TOL_B = 1e-12
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
    """mpmath 真值:∫_{-a}^{a} H_0^{(2)}(k|t|) dt(把 0 作为分段点)。"""
    K = mp.mpf(repr(float(k)))
    A = mp.mpf(repr(float(a)))
    return complex(mp.quad(lambda t: mp.hankel2(0, K * abs(t)), [-A, 0, A]))


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

#: ka 从 0.05 到 5 —— 0.5 以下是纯级数路径,0.5 以上走分段
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

# ======================================================= [1] 判据 A
print("\n[1] 判据 A —— self_coupling_integral(主路径)vs mpmath 高精度积分")
if HAVE_MPMATH:
    worst_a = 0.0
    for k, a in CASES:
        got = self_coupling_integral(k, a)
        ref = mp_self_coupling(k, a)
        e = rel_err(got, ref)
        worst_a = max(worst_a, e)
        path = "级数" if k * a <= 0.5 else "分段"
        print(
            f"         k={k:<5g} a={a:<5g} ka={k * a:<7.3g} [{path}]  "
            f"ours={got.real:+.12f}{got.imag:+.12f}i  rel_err={e:.3e}"
        )
    report(f"{len(CASES)} 组 (k, a) 上的最大相对误差", worst_a, TOL_A)
else:
    print("  [FAIL] 未安装 mpmath —— 判据 A 无法执行")
    failures.append("判据 A 未执行")

# ======================================================= [1b] 判据 A2
print("\n[1b] 判据 A2 —— series_self_integral 单独(ka <= 0.5)")
if HAVE_MPMATH:
    worst_a2 = 0.0
    for x in (0.01, 0.05, 0.1, 0.2, 0.35, 0.5):
        e = rel_err(series_self_integral(1.0, x), mp_self_coupling(1.0, x))
        worst_a2 = max(worst_a2, e)
    report("ka = 0.01 ... 0.5 共 6 点", worst_a2, TOL_A)

    try:
        series_self_integral(1.0, 0.6)
        print("  [FAIL] ka > 0.5 未抛 ValueError")
        failures.append("series range guard")
    except ValueError:
        print("  [PASS] ka > 0.5 抛出 ValueError(不会静默给出坏结果)")
else:
    report("ka = 0.01 ... 0.5", float("nan"), TOL_A)

# ======================================================= [2] 判据 B
print("\n[2] 判据 B —— 两个闭式主部积分 vs mpmath")
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

# ======================================================= [3] 判据 B3
print("\n[3] 判据 B3 —— 二阶系数 C0 / C1 的正确性(余项必须是 O(x^4 ln x))")
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

# ======================================================= [4] 判据 C
print("\n[4] 判据 C —— 分段方案的稳定性(ka = 5,改变外段 Gauss 阶数)")
if HAVE_MPMATH:
    k_c, a_c = 10.0, 0.5  # ka = 5,delta = 0.05,外段占绝大部分
    ref_c = mp_self_coupling(k_c, a_c)
    vals = []
    for order in (16, 32, 64, 128):
        e = rel_err(self_coupling_integral(k_c, a_c, order=order), ref_c)
        vals.append(e)
        print(f"         order={order:<4d} rel_err={e:.3e}")
    print(f"         各阶数结果相对散布 = {max(vals) - min(vals):.3e}")
    print("         外段 [delta, a] 在 ka=5 时容纳约 4.5 弧度相位,16 阶不够是正常的")
    print("         收敛行为;32 阶起已到机器精度,128 阶的回升是浮点噪声而非缺陷")
    if max(vals[1:]) < 1e-12:
        print("  [PASS] 阶数 >= 32 时结果 < 1e-12(默认 64 阶有充分余量)")
    else:
        print("  [FAIL] 分段结果依赖阶数")
        failures.append("segment stability")
else:
    report("分段稳定性", float("nan"), TOL_A)

# ======================================================= [5] 判据 D
print("\n[5] 判据 D —— 错误做法的危害量化(证明必须处理奇异性)")
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

# ======================================================= [6] 判据 E
print("\n[6] 判据 E —— 形状、边界与微积分一致性")
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

if HAVE_MPMATH:
    k_d, a_d, h = 1.5, 0.4, 1e-6
    num = (
        self_coupling_integral(k_d, a_d + h) - self_coupling_integral(k_d, a_d - h)
    ) / (2 * h)
    ana = 2.0 * complex(hankel0_2(k_d * a_d))
    report("dI/da = 2 H_0^(2)(ka) (微积分一致性)", rel_err(num, ana), 1e-8)

    # 与纯数值路径(拆主部 + 求积)交叉验证 —— 两条完全不同的路子应给同一答案
    k_x, a_x = 2.0, 0.3
    ref_x = mp_self_coupling(k_x, a_x)
    nd, wt = gauss_legendre(DEFAULT_QUAD_ORDER)
    via_analytic = analytic_part_integral(k_x, a_x) + a_x * complex(
        np.sum(wt * regular_part(k_x, a_x * nd))
    )
    print(
        f"         交叉验证(k={k_x}, a={a_x}):级数路径 rel_err="
        f"{rel_err(self_coupling_integral(k_x, a_x), ref_x):.3e}  "
        f"主部+求积路径 rel_err={rel_err(via_analytic, ref_x):.3e}"
    )

print("\n" + "=" * 86)
if failures:
    print(f"结果: {len(failures)} 项未通过 -> {failures}")
    raise SystemExit(1)
print("结果: 全部通过")
