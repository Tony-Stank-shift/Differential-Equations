"""1.1 自检:格林函数模块。

验收判据(来自 WBS 1.1):
    与 scipy.special.hankel2 对比,kR 在 [1e-6, 10] 上相对误差 < 1e-10。

本脚本做三件事:
    1. 小参数级数分支 hankel0_2_small 与 scipy 基准逐点对比
    2. 自动切换的 hankel0_2 在 [1e-6, 10] 全区间上与 scipy 对比
    3. 检查切换点附近是否连续(防止两条分支拼接出跳变)

运行:
    python tests/check_green.py
"""
from __future__ import annotations

import sys
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
    print(f"  [{status}] {name:<52s} max_rel_err = {err:.3e}  (tol {tol:.0e})")
    if not ok:
        failures.append(name)


print("=" * 78)
print("1.1 格林函数自检")
print("=" * 78)

# ---------------------------------------------------------------- 1. 级数分支
print("\n[1] 小参数级数分支 hankel0_2_small vs scipy 基准")
z_small = np.logspace(-6, np.log10(SMALL_THRESHOLD), 300)
got = np.array([complex(hankel0_2_small(z)) for z in z_small])
ref = scipy_ref(0, z_small)
report(f"kR in [1e-6, {SMALL_THRESHOLD}]  ({len(z_small)} points)", rel_err(got, ref).max())

# 更极端:R 小到 1e-12 也要能算(只有级数分支能扛住,scipy 会丢精度)
z_tiny = np.array([1e-12, 1e-10, 1e-8, 1e-6])
got_tiny = np.array([complex(hankel0_2_small(z)) for z in z_tiny])
ref_tiny = scipy_ref(0, z_tiny)
print(f"  [info] 极端小参数 (scipy 在此区间可能已丢精度):")
for z, g, r in zip(z_tiny, got_tiny, ref_tiny):
    print(f"         kR={z:.0e}   ours={g.real:+.12f}{g.imag:+.12f}i   scipy={r.real:+.12f}{r.imag:+.12f}i")

# ------------------------------------------------- 2. 全区间与 scipy 的一致性
print("\n[2] 自动切换的 hankel0_2 在整个 [1e-6, 10] 区间")
# 刻意取非整数幂的密度,并在阈值两侧加密
z_all = np.concatenate([
    np.logspace(-6, np.log10(SMALL_THRESHOLD) - 0.02, 200),
    np.linspace(SMALL_THRESHOLD - 0.05, SMALL_THRESHOLD + 0.05, 200),
    np.linspace(SMALL_THRESHOLD + 0.05, 10.0, 400),
])
got_all = np.array([complex(hankel0_2(z)) for z in z_all])
ref_all = scipy_ref(0, z_all)
report(f"kR in [1e-6, 10]  ({len(z_all)} points)", rel_err(got_all, ref_all).max())

# ------------------------------------------------- 3. 两条分支的拼接一致性
print("\n[3] 两条分支的拼接一致性")
t = SMALL_THRESHOLD
report(
    f"级数分支 vs scipy 在阈值 t={t} 处",
    float(rel_err(complex(hankel0_2_small(t)), complex(scipy_ref(0, t)))),
)

# 差分商:若拼接处存在跳变,差分商会异常放大(数值上表现为 O(1/eps))
eps = 1e-7
slope = abs(complex(hankel0_2(t + eps)) - complex(hankel0_2(t - eps))) / (2 * eps)
print(f"  [info] 阈值处差分商 |dH/dx| = {slope:.6f}  (应为有限常数)")
if 0.1 < slope < 10.0:
    print("  [PASS] 差分商为有限常数,无拼接跳变")
else:
    print("  [FAIL] 差分商异常,疑似拼接跳变")
    failures.append("continuity slope")

# ------------------------------------------------------- 4. 向量化与形状一致
print("\n[4] 形状与向量化")
try:
    z2 = np.linspace(0.1, 5.0, 12).reshape(3, 4)
    out2 = hankel0_2(z2)
    assert out2.shape == (3, 4), f"shape {out2.shape} != (3, 4)"
    report("数组输入返回同形状 (3,4)", float(rel_err(out2, scipy_ref(0, z2)).max()))
except Exception as exc:  # pragma: no cover
    print(f"  [FAIL] 形状测试异常: {exc}")
    failures.append("shape")

# ------------------------------------------------------------ 5. 格林函数本身
print("\n[5] 格林函数与对数奇异主部")
k = 2.0
r = np.array([0.0, 0.0])
rp = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 0.25]])
g = green_2d(k, r, rp)
R = np.linalg.norm(rp, axis=1)
g_ref = 0.25j * scipy_ref(0, k * R)
report("green_2d vs (i/4)H0^(2) 直接计算", float(rel_err(g, g_ref).max()))

# 小 R 时,H_0^{(2)} 减去对数奇异部分后应趋于 1
R_tiny = 1e-9
g_val = complex(green_2d(k, r, np.array([R_tiny, 0.0])))
sing = complex(log_singular_part(k, R_tiny))
residual = g_val - 0.25j * sing
print(f"  [info] R=1e-9:  G = {g_val:.6e}")
print(f"               (i/4)*log_singular = {0.25j * sing:.6e}")
print(f"               余项 G - (i/4)*sing = {residual:.6e}   (应趋于 (i/4)*1)")
report("R->0 时余项趋于 (i/4)", abs(residual - 0.25j), tol=1e-4)

print("\n" + "=" * 78)
if failures:
    print(f"结果: {len(failures)} 项未通过 -> {failures}")
    raise SystemExit(1)
print("结果: 全部通过")
