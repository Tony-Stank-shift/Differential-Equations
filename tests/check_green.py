"""1.1 自检:格林函数模块。

验收判据(来自 WBS 1.1。经第三方复核后拆成两条 —— 理由见下):
    A. 级数分支 ``hankel0_2_small`` 在 kR ∈ [1e-12, 0.5] 上与 **mpmath 60 位
       高精度真值** 的相对误差 < 1e-10 —— 这是实质判据
    B. 自动切换包装 ``hankel0_2`` 在 kR ∈ [1e-6, 10] 上与 scipy 的相对误差
       < 1e-10 —— 这是集成检查

为什么必须拆(第三方复核发现):
    原判据只有 B。但 ``green.py`` 在 |x| >= 0.5 时直接调用 scipy,所以 B 在
    [0.5, 10](约占对数区间长度的 2/3)上 ``np.array_equal(got, ref)`` 恒为
    True —— 拿被验证对象自己做基准,是同义反复。真正的独立验证由 A 承担。

另含:
    [3] 阈值处拼接一致性 + 差分商 Δh 缩放检验(强于单点差分商)
    [4] 形状与向量化
    [5] 格林函数因子 + 对数奇异主部的余项极限 **及其适用门槛**
    [6] 误用守卫(hankel0_2_small 超范围、log_singular_part 的 R=0)

运行:
    python tests/check_green.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.special import hankel2 as scipy_ref

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from green import (  # noqa: E402
    SMALL_THRESHOLD,
    green_2d,
    hankel0_2,
    hankel0_2_small,
    log_singular_part,
)

try:
    import mpmath as mp

    mp.mp.dps = 60
    HAVE_MPMATH = True
except ImportError:  # pragma: no cover
    HAVE_MPMATH = False

TOL = 1e-10
failures: list[str] = []


def rel_err(a, b):
    a = np.asarray(a, dtype=complex)
    b = np.asarray(b, dtype=complex)
    denom = np.abs(b)
    denom = np.where(denom == 0.0, 1.0, denom)
    return np.abs(a - b) / denom


def report(name: str, err: float, tol: float = TOL) -> None:
    ok = err < tol
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name:<56s} max_rel_err = {err:.3e}  (tol {tol:.0e})")
    if not ok:
        failures.append(name)


def mpmath_hankel2(x) -> complex:
    """mpmath 60 位精度的 H_0^{(2)}(x) —— 独立于 scipy 与 green.py 的真值。"""
    return complex(mp.hankel2(0, mp.mpf(repr(float(x)))))


print("=" * 82)
print("1.1 格林函数自检")
print("=" * 82)
print(f"  mpmath 可用: {HAVE_MPMATH}" + (f" (dps={mp.mp.dps})" if HAVE_MPMATH else ""))

# ======================================================= [1] 判据 A:级数分支
print("\n[1] 判据 A —— 级数分支 hankel0_2_small vs mpmath 60dps 真值")
z_series = np.logspace(-12, np.log10(SMALL_THRESHOLD), 800)  # 末点即阈值 0.5
got_s = np.array([complex(hankel0_2_small(z)) for z in z_series])

if HAVE_MPMATH:
    ref_mp = np.array([mpmath_hankel2(z) for z in z_series])
    report(
        f"kR in [1e-12, {SMALL_THRESHOLD}]  ({len(z_series)} 点)",
        rel_err(got_s, ref_mp).max(),
    )

    z_tiny = np.array([1e-12, 1e-10, 1e-8, 1e-6])
    ours_t = np.array([complex(hankel0_2_small(z)) for z in z_tiny])
    mp_t = np.array([mpmath_hankel2(z) for z in z_tiny])
    sc_t = scipy_ref(0, z_tiny)
    print("  [info] 极端小参数三方对照 (ours / mpmath / scipy):")
    for z, o, m_, s in zip(z_tiny, ours_t, mp_t, sc_t):
        print(f"         kR={z:.0e}  ours ={o.real:+.14f}{o.imag:+.14f}i")
        print(f"                    mp   ={m_.real:+.14f}{m_.imag:+.14f}i")
        print(f"                    scipy={s.real:+.14f}{s.imag:+.14f}i")
    report(
        "scipy 在 [1e-12,0.5] 上相对 mpmath 的误差(基准资格)",
        rel_err(sc_t, mp_t).max(),
    )
else:
    print("  [SKIP] 未安装 mpmath —— 判据 A 无法执行,本脚本将失去独立验证")
    failures.append("判据 A 未执行")

print(f"  [info] 同点位 vs scipy 相对误差 = {float(rel_err(got_s, scipy_ref(0, z_series)).max()):.3e}  (辅助,不作判据)")

# ======================================================= [2] 判据 B:全区间
print("\n[2] 判据 B —— 自动切换的 hankel0_2 在整个 [1e-6, 10]")
z_all = np.concatenate(
    [
        np.logspace(-6, np.log10(SMALL_THRESHOLD) - 0.02, 200),
        np.linspace(SMALL_THRESHOLD - 0.05, SMALL_THRESHOLD + 0.05, 200),
        np.linspace(SMALL_THRESHOLD + 0.05, 10.0, 400),
    ]
)
got_all = np.array([complex(hankel0_2(z)) for z in z_all])
ref_all = scipy_ref(0, z_all)
report(f"kR in [1e-6, 10]  ({len(z_all)} 点)", rel_err(got_all, ref_all).max())

mask_big = np.abs(z_all) >= SMALL_THRESHOLD
identical = bool(np.array_equal(got_all[mask_big], ref_all[mask_big]))
print(f"  [info] |x| >= 阈值的子段与 scipy 逐位相同: {identical}")
if identical:
    print("         ⚠️ 该子段是同义反复(实现即直接调用 scipy),证据力全部来自判据 A")

if HAVE_MPMATH:
    sel = np.linspace(0, len(z_all) - 1, 60).astype(int)
    report(
        "[判据 A 延伸] 全区间抽样 60 点 vs mpmath(含 |x|>=0.5 段)",
        rel_err(got_all[sel], np.array([mpmath_hankel2(z) for z in z_all[sel]])).max(),
    )

# ======================================================= [3] 拼接一致性
print("\n[3] 两条分支的拼接一致性")
t = SMALL_THRESHOLD
if HAVE_MPMATH:
    report(
        f"级数分支在阈值 t={t} 处 vs mpmath",
        float(rel_err(complex(hankel0_2_small(t)), mpmath_hankel2(t))),
    )

deltas = [1e-4, 1e-5, 1e-6, 1e-7]
ratios = []
for dh in deltas:
    d = abs(complex(hankel0_2(t + dh)) - complex(hankel0_2(t - dh)))
    ratios.append(d / (2.0 * dh))
mean_r = float(np.mean(ratios))
spread = float((max(ratios) - min(ratios)) / mean_r)
print("  [info] 差分商 Δh 缩放检验:C = " + ", ".join(f"{r:.6f}" for r in ratios))
print(f"         均值 = {mean_r:.6f},相对散布 = {spread:.3e}")
print("         若拼接处存在阶跃 β,C ≈ β/Δh + const 会随 Δh 减半而翻倍(散布 O(1));")
print("         无阶跃则 C 为常数,散布应停在机器精度量级。")
if spread < 1e-3 and 0.1 < mean_r < 10.0:
    print("  [PASS] C 为有限常数,无拼接跳变")
else:
    print(f"  [FAIL] 差分商随 Δh 漂移(散布 {spread:.3e}),疑似拼接跳变")
    failures.append("continuity slope scaling")

# ======================================================= [4] 形状与向量化
print("\n[4] 形状与向量化")
try:
    z2 = np.linspace(0.1, 5.0, 12).reshape(3, 4)
    out2 = hankel0_2(z2)
    assert out2.shape == (3, 4), f"shape {out2.shape} != (3, 4)"
    report("数组输入返回同形状 (3,4)", float(rel_err(out2, scipy_ref(0, z2)).max()))
    for label, val in (("标量", 1.3), ("0 维数组", np.array(1.3)), ("列表(长度1)", [1.3])):
        r_ = hankel0_2(val)
        v_ = complex(np.asarray(r_).ravel()[0])
        print(f"  [info] {label}: 返回 {type(r_).__name__},值 {v_:.6e}")
except Exception as exc:  # pragma: no cover
    print(f"  [FAIL] 形状测试异常: {exc}")
    failures.append("shape")

# ======================================================= [5] 格林函数
print("\n[5] 格林函数")
k = 2.0
r = np.array([0.0, 0.0])
rp = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 0.25]])
g = green_2d(k, r, rp)
R = np.linalg.norm(rp, axis=1)
report("green_2d vs (j/4)H0^(2) 直接计算", float(rel_err(g, 0.25j * scipy_ref(0, k * R)).max()))

z_probe = 1.7
factor = complex(
    green_2d(1.0, np.array([0.0, 0.0]), np.array([z_probe, 0.0]))
) / complex(hankel0_2(z_probe))
print(f"  [info] G / H0^(2)(kR) = {factor.real:.3e}{factor.imag:+.3e}j   (应恰为 0+0.25j)")
if abs(factor - 0.25j) < 1e-15:
    print("  [PASS] 因子精确等于 (j/4)")
else:
    print("  [FAIL] 因子不等于 (j/4)")
    failures.append("green_2d factor")

# ======================================================= [5b] 余项适用门槛
print("\n[5b] 对数奇异主部:余项极限与 **适用门槛**")
print("       |H0^(2)(kR) - log_singular_part(k,R) - 1|")
gate_ok = True
for kr in (1e-8, 1e-6, 1e-4, 1e-2, 0.05, 0.3, 0.5):
    resid = complex(hankel0_2(kr)) - complex(log_singular_part(1.0, kr))
    dev = abs(resid - 1.0)
    flag = "  <-- 可当常数 1 用" if kr <= 0.05 else "  <-- 不可当 1 用"
    print(f"       kR={kr:<9.0e} 偏离 = {dev:.3e}{flag}")
    if kr <= 0.05 and dev > 5e-3:
        gate_ok = False
if gate_ok:
    print("  [PASS] kR <= 0.05 段偏离 < 5e-3,门槛结论与代码文档一致")
else:
    print("  [FAIL] 门槛结论与实测不符")
    failures.append("log_singular gate")

lhs = complex(log_singular_part(2.0, 1e-9))
rhs = complex(log_singular_part(1.0, 2e-9))
report("齐次性:只依赖 kR", abs(lhs - rhs) / abs(rhs))

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    v0 = log_singular_part(1.0, 0.0)
    warned = any("log_singular_part" in str(c.message) for c in caught)
print(f"  [info] log_singular_part(k, 0) = {v0}   告警: {warned}")
if warned:
    print("  [PASS] R=0 有显式告警(不再静默污染对角线)")
else:
    print("  [FAIL] R=0 无告警")
    failures.append("log_singular R=0 warn")

# ======================================================= [6] 误用守卫
print("\n[6] 误用守卫")
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    hankel0_2_small(30.0)
    guarded = any("hankel0_2_small" in str(c.message) for c in caught)
print(f"  [info] hankel0_2_small(30.0) 告警: {guarded}")
if guarded:
    print("  [PASS] 超范围使用级数分支会告警(不再静默错)")
else:
    print("  [FAIL] 超范围使用级数分支无告警")
    failures.append("series guard")

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    g0 = np.asarray(green_2d(2.0, np.array([0.0, 0.0]), np.array([0.0, 0.0]))).ravel()[0]
print(f"  [info] green_2d 在 R=0 返回 {g0}(预期 nan,已在 docstring 声明)")

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    hankel0_2_small(0.5)
    hankel0_2(np.array([1e-9, 0.3, 0.5, 3.0]))
    log_singular_part(2.0, 0.3)
    clean = [str(c.message) for c in caught]
print(f"  [info] 正当用法产生的告警数 = {len(clean)}")
if not clean:
    print("  [PASS] 正当用法无告警")
else:
    print(f"  [FAIL] 正当用法误告警: {clean}")
    failures.append("false alarm")

print("\n" + "=" * 82)
if failures:
    print(f"结果: {len(failures)} 项未通过 -> {failures}")
    raise SystemExit(1)
print("结果: 全部通过")
