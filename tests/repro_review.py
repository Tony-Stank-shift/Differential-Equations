# -*- coding: utf-8 -*-
"""复现第三方复核(1.2)指出的两处失效 —— 存档用,不参与主自检。

复核员(独立 subagent,2026-10)在审 ``quad.py`` 时给出三条:

  ① [high]   默认 order=64 的分段路径在 ka ~ 52 起静默越过 1e-12,ka = 1000 时
              相对误差 5.53(完全错)。**当时这是主路径。**
  ② [medium] "拆主部 + 求积"的备用路径在 ka > 0.5 就掉出 1e-12(ka = 5 -> 8.6e-8)。
  ③ [info]   ∫_0^z H_0^{(2)}(u) du 有 **Struve 闭式** —— 作者原先断言"无闭式",错。

处置(见 DECISIONS D9 / D10):
  - 主路径已换成 Struve 闭式;原分段路径降级为 ``split_self_integral`` 并保留
    ``order`` 参数,以便本脚本复现 ①。
  - 备用路径保留(1.3 的近奇异积分要用),docstring 写明适用范围,自检里加了
    真实断言 —— 不留"文档推荐但无人守护"的路径。
  - "无闭式"的断言在 DECISIONS D9 里已被就地更正。

用法:
    python tests/repro_review.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from scipy.special import hankel2 as H2  # noqa: E402
from scipy.special import struve  # noqa: E402

import quad  # noqa: E402


try:
    import mpmath as mp

    mp.mp.dps = 40
    HAVE_MPMATH = True
except ImportError:  # pragma: no cover
    HAVE_MPMATH = False


def F_scipy(z):
    """``∫_0^z H_0^{(2)}(u) du`` 的 scipy 实现(与 ``quad.closed_self_integral`` 同源)。

    仅用于说明"同源实现比对是同义反复"——**不要拿它当验收真值**。
    """
    return z * H2(0, z) + (np.pi * z / 2.0) * (
        H2(1, z) * struve(0, z) - H2(0, z) * struve(1, z)
    )


def F(z):
    """``∫_0^z H_0^{(2)}(u) du`` 的 Struve 闭式 —— **mpmath 40 位独立真值**。

    刻意不用 scipy:同一台机器上 scipy 实现与 ``quad`` 的实现是同一段公式,
    比它们等于自比(纪律 D8)。mpmath 走的是自己的特殊函数库。
    """
    if not HAVE_MPMATH:
        raise RuntimeError("本脚本需要 mpmath 才能给出独立真值")
    zz = mp.mpf(repr(float(z)))
    return complex(
        zz * mp.hankel2(0, zz)
        + (mp.pi * zz / 2)
        * (mp.hankel2(1, zz) * mp.struveh(0, zz) - mp.hankel2(0, zz) * mp.struveh(1, zz))
    )


def I(k, a):
    """``∫_{-a}^{a} H_0^{(2)}(k|t|) dt`` 的独立真值。"""
    return 2.0 / k * F(k * a)


def main() -> int:
    print("=" * 78)
    print("复核最小反例复现(存档:记录问题已被定位与处置,不是当前实现的验收)")
    print("=" * 78)

    print("\n[① 历史主路径 order=64 的失效阈值]  split_self_integral, k = 1")
    print(f"  {'ka':>8} {'rel err':>12}")
    for a in (32.0, 40.0, 52.0, 56.0, 64.0, 128.0, 256.0, 1000.0):
        got = quad.split_self_integral(1.0, a, order=64)
        ref = I(1.0, a)
        print(f"  {a:>8.0f} {abs(got - ref) / abs(ref):>12.2e}")

    print("\n[① 处置后:主路径 closed_self_integral 在同点位]")
    print(f"  {'ka':>8} {'rel err':>12}")
    for a in (52.0, 256.0, 1000.0, 1e4):
        got = quad.closed_self_integral(1.0, a)
        ref = I(1.0, a)
        print(f"  {a:>8.0f} {abs(got - ref) / abs(ref):>12.2e}")

    print("\n[② 备用路径的 ka 依赖]  analytic_part_integral + GL(regular_part)")
    print(f"  {'k':>6} {'a':>6} {'ka':>8} {'order':>6} {'rel err':>12}")
    for k, a in ((2.0, 0.3), (1.0, 1.0), (1.0, 5.0)):
        ref = I(k, a)
        for order in (64, 512):
            n, w = quad.gauss_legendre(order)
            via = quad.analytic_part_integral(k, a) + a * np.sum(
                w * quad.regular_part(k, a * n)
            )
            print(
                f"  {k:>6g} {a:>6g} {k*a:>8g} {order:>6d} "
                f"{abs(via - ref) / abs(ref):>12.2e}"
            )

    print("\n[③ 被证伪的贝塞尔闭式]  I  ?=  2a * H_1^{(2)}(ka)")
    for k, a in ((2.0, 0.3), (10.0, 0.5)):
        ref = I(k, a)
        wrong = 2.0 * a * H2(1, k * a)
        print(
            f"  k={k:<5g} a={a:<5g}  2a*H1^(2) rel err = "
            f"{abs(wrong - ref) / abs(ref):.2e}   <- 错,需要 Struve"
        )

    print("\n[附:边界 k*a 恰为 0.5 附近不再抛无谓 ValueError]")
    for k in (1.0, 2.0, 0.5, 1e-6):
        a = 0.5 / k
        try:
            v = quad.self_coupling_integral(k, a)
            note = "ok"
        except ValueError as exc:
            v, note = float("nan"), f"ValueError: {exc}"
        print(f"  k={k:<8g} a=0.5/k={a:<12.6g} -> {note}")
        for scale in (1.0 + 1e-15, 1.0 + 1e-16):
            try:
                quad.self_coupling_integral(k, a * scale)
                note = "ok"
            except ValueError as exc:
                note = f"ValueError: {exc}"
            print(f"      a*(1+{scale - 1:.0e}) -> {note}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
