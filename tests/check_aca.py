# -*- coding: utf-8 -*-
"""3.2 自检:部分主元 ACA(src/aca.py)。

本文件验五件事:

    ① **原验收**:数值秩-5 矩阵能否被 ACA 精确恢复(残差到机器精度),且"4 步不够";
    ② **数学边界**:k 步 ACA 的近似秩 ≤ k ⇒ 对数值秩-5 且 σ₅/σ₁ ≥ 1e-3 的矩阵,
       3 步达到 1e-10 是不可能的(Eckart–Young 下界);同时给出"只按代数秩 5"
       可以被满足的反例 —— 说明判据必须钉住数值秩;
    ③ **契约**:每层 u/v 必须等于"用之前的层独立重算的残差行/列 ÷ 主元";
       每步残差必须等于独立重算的 ||A − U_k V_k||_F/||A||_F;`pivot_rows` 必须
       真的走"argmax|u| + 回退"这条规则(三个 fixture 的精确值);
    ④ **导出量**:`rel_error` 必须与独立 ref 一致(不能自己证自己)、`residual[0]=1`、
       `n_fallbacks` 精确计数、`pivot_rows` 精确值、`u[i_k,k]=1`(含回退路径);
    ⑤ **边界与退化**:零/空/单行/单列/对角/首行全零/非连续视图/dtype/确定性/
       start_row/NaN 输入/tol=0/max_rank=0。

计数口径(D24):有鉴别力的判据 + 恒 PASS 的基线分开数;鉴别力由
``tests/mut_check_aca.py`` 的反例审计证明(每个变异体都必须把本文件打红)。

⚠️ 2026-11-24 第 1 轮复核(needs_revision)的 8 条 high 全部修入本版:
   `rel_error` 自证、`start_row` 零覆盖、选行规则未钉、单调性假命题、
   新验收的 4 步绝对阈值、台账未更新、13 个逃逸变异体、回退导出量未覆盖。

运行:
    python tests/check_aca.py        # 期望 exit 0
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from aca import ACAFactor, aca_lowrank  # noqa: E402

failures: list[str] = []
COUNTS = {"disc": 0, "base": 0}


def ok(label: str, cond: bool, detail: str = "", *, kind: str = "disc") -> None:
    COUNTS[kind] += 1
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}{('   ' + detail) if detail else ''}")
    if not cond:
        failures.append(label)


def report(name: str, value: float, tol: float, note: str = "", *, kind: str = "disc") -> None:
    """相对误差/范数类量:必须 **0 ≤ value ≤ tol**（负值也是坏的,复核第 10 条）。"""
    good = np.isfinite(value) and 0.0 <= value <= tol
    COUNTS[kind] += 1
    print(f"  [{'PASS' if good else 'FAIL'}] {name:<48s} = {value:.6e}  (0 ≤ · ≤ {tol:.1e})"
          f"{('   ' + note) if note else ''}")
    if not good:
        failures.append(name)


# ============================================================================
# 构造与独立量
# ============================================================================
def rank_k(m: int, n: int, k: int, *, seed: int = 0, complex_: bool = False) -> np.ndarray:
    """精确秩-k 的随机矩阵,按 Frobenius 归一 ⇒ 绝对残差 = 相对残差。"""
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((m, k))
    v = rng.standard_normal((k, n))
    if complex_:
        u = u + 1j * rng.standard_normal((m, k))
        v = v + 1j * rng.standard_normal((k, n))
    a = u @ v
    return a / np.linalg.norm(a)


def with_singular_values(m: int, n: int, s: np.ndarray, *, seed: int = 0) -> np.ndarray:
    """随机酉因子 + 指定奇异值(归一化后返回)。用来构造"代数秩 k 但数值秩 k-2"的反例。"""
    rng = np.random.default_rng(seed)
    q1, _ = np.linalg.qr(rng.standard_normal((m, m)))
    q2, _ = np.linalg.qr(rng.standard_normal((n, n)))
    a = (q1[:, : len(s)] * np.asarray(s)) @ q2[:, : len(s)].T
    return a / np.linalg.norm(a)


def ey_bound(a: np.ndarray, k: int) -> float:
    """Eckart–Young:任何秩 ≤ k 的近似,其 Frobenius 误差 ≥ sqrt(Σ_{i>k} σ_i²)。"""
    s = np.linalg.svd(a, compute_uv=False)
    return float(np.sqrt(np.sum(s[k:] ** 2)) / np.linalg.norm(a))


def ref_error(a: np.ndarray, fac: ACAFactor) -> float:
    """**独立**相对误差:只用 fac.u/fac.v 与 A,不调用 fac.rel_error。"""
    den = float(np.linalg.norm(a))
    return 0.0 if den == 0.0 else float(np.linalg.norm(a - fac.u @ fac.v) / den)


def contract_error(fac: ACAFactor, a: np.ndarray) -> dict:
    """独立重算每一层:u_k = R_k[:,j_k]/δ_k,v_k = R_k[i_k,:],δ_k = R_k[i_k,j_k]。"""
    m, n = a.shape
    u_prev = np.zeros((m, 0), dtype=a.dtype)
    v_prev = np.zeros((0, n), dtype=a.dtype)
    err_u = err_v = 0.0
    pivot_ok = True
    identity_ok = True
    for k in range(fac.rank):
        i, j = int(fac.pivot_rows[k]), int(fac.pivot_cols[k])
        row = a[i, :] - (u_prev[i, :] @ v_prev if k else 0.0)
        col = a[:, j] - (u_prev @ v_prev[:, j] if k else 0.0)
        delta = row[j]
        pivot_ok = pivot_ok and bool(abs(delta) + 1e-15 >= np.max(np.abs(row)))
        identity_ok = identity_ok and bool(abs(fac.u[i, k] - 1.0) <= 1e-12)
        den_v = max(float(np.linalg.norm(row)), 1e-300)
        den_u = max(float(np.linalg.norm(col / delta)), 1e-300)
        err_v = max(err_v, float(np.linalg.norm(fac.v[k, :] - row) / den_v))
        err_u = max(err_u, float(np.linalg.norm(fac.u[:, k] - col / delta) / den_u))
        u_prev = np.concatenate([u_prev, fac.u[:, [k]]], axis=1)
        v_prev = np.concatenate([v_prev, fac.v[[k], :]], axis=0)
    return {"u": err_u, "v": err_v, "pivot": pivot_ok, "identity": identity_ok}


def pivot_rule_violations(fac: ACAFactor, a: np.ndarray, rel_tol: float = 1e-10) -> list[tuple]:
    """检查第 k+1 步的选行是否 = argmax|u_k|;若不是,**那一行必须残差为零**(回退)。"""
    m, n = a.shape
    norm = float(np.linalg.norm(a))
    u_prev = np.zeros((m, 0), dtype=a.dtype)
    v_prev = np.zeros((0, n), dtype=a.dtype)
    bad: list[tuple] = []
    for k in range(fac.rank - 1):
        u_prev = np.concatenate([u_prev, fac.u[:, [k]]], axis=1)
        v_prev = np.concatenate([v_prev, fac.v[[k], :]], axis=0)
        expected = int(np.argmax(np.abs(fac.u[:, k])))
        if int(fac.pivot_rows[k + 1]) == expected:
            continue
        row = a[expected, :] - (u_prev[expected, :] @ v_prev if k + 1 else 0.0)
        if float(np.max(np.abs(row))) > rel_tol * norm:
            bad.append((k, expected, int(fac.pivot_rows[k + 1])))
    return bad


# ============================================================================
# ① 原验收:数值秩-5 精确恢复;"4 步不够"
# ============================================================================
def section_recovery() -> None:
    print("\n[① 数值秩-5 精确恢复(台账 3.2 的意图)]")
    a = rank_k(60, 60, 5, seed=1, complex_=True)
    fac = aca_lowrank(a, tol=1e-10)
    ok("rank = 5", fac.rank == 5, f"实际 {fac.rank}")
    ok("converged = True", fac.converged)
    report("独立重算 ||A − U V|| / ||A||", ref_error(a, fac), 1e-12)
    report("residual[-1]", float(fac.residual[-1]), 1e-12)
    ok("residual[0] = 1(非零矩阵的约定)", float(fac.residual[0]) == 1.0,
       f"{fac.residual[0]!r}")
    ok("residual 长度 = rank + 1", len(fac.residual) == fac.rank + 1,
       f"{len(fac.residual)} vs {fac.rank + 1}")

    # 4 步:必须还差得远
    f4 = aca_lowrank(a, tol=1e-10, max_rank=4)
    ok("4 步未收敛", not f4.converged)
    ok("4 步残差 ≥ 0.5·EY₄(不保证最优,但不能差一个量级)",
       float(f4.residual[-1]) >= 0.5 * ey_bound(a, 4),
       f"{f4.residual[-1]:.3e} vs 0.5·EY₄={0.5 * ey_bound(a, 4):.3e}")

    # 多 seed:判据不能只对 seed=1 成立
    print("  —— 多 seed 交叉(seeds 0–4,60×60 complex) ——")
    worst5 = worst4 = 0.0
    for seed in range(5):
        b = rank_k(60, 60, 5, seed=seed, complex_=True)
        g = aca_lowrank(b, tol=1e-10)
        g4 = aca_lowrank(b, tol=0.0, max_rank=4)
        worst5 = max(worst5, float(g.residual[-1]))
        worst4 = max(worst4, float(eyes_ratio(g4, b)))
        ok(f"seed={seed}: 5 步收敛、rank=5、残差 ≤1e-12",
           g.rank == 5 and g.converged and ref_error(b, g) <= 1e-12)
    report("多 seed:5 步残差最大值", worst5, 1e-12)
    report("多 seed:4 步残差 / EY₄ 的最大比值", worst4, 10.0,
           note="ACA 不保证 EY 最优;比值 1–3 是正常的")


def eyes_ratio(fac: ACAFactor, a: np.ndarray) -> float:
    """残差 / (EY 下界 + 1e-300) —— 量纲无关的"离最优多远"。"""
    return float(fac.residual[-1]) / (ey_bound(a, fac.rank) + 1e-300)


# ============================================================================
# ② 数学边界 + "只按代数秩 5"的反例
# ============================================================================
def section_impossible() -> None:
    print("\n[② 原判据「3 步内 < 1e-10」的判决:Eckart–Young + 秩界]")
    a = rank_k(80, 50, 5, seed=2, complex_=True)
    f3 = aca_lowrank(a, tol=0.0, max_rank=3)
    bound = ey_bound(a, 3)
    r3 = ref_error(a, f3)          # **独立重算**,不用 f3.residual 自证
    report("|f3.residual[-1] − 独立重算 r3|", abs(float(f3.residual[-1]) - r3), 1e-12)
    ok("3 步近似的秩 = 3", f3.rank == 3, f"{f3.rank}")
    ok("3 步 r3 ≥ EY₃(数学下界)", r3 >= bound - 1e-12, f"{r3:.6e} vs {bound:.6e}")
    ok("3 步 r3 > 1e-3 ⇒ 原判据在数值秩-5 上不可满足", r3 > 1e-3, f"{r3:.3e}")
    ok("EY₃ 本身 > 1e-3(本项目生成器,不是单个矩阵的偶然)",
       all(ey_bound(rank_k(60, 60, 5, seed=s, complex_=True), 3) > 1e-3 for s in range(5)))
    print(f"         (EY₃ = {bound:.4f};ACA r3 = {r3:.4f};比值 {r3 / bound:.2f})")

    # 反例:代数秩 5,但 σ₄=σ₅=1e-12 ⇒ **数值秩 = 3**,3 步可以到 1e-12。
    # 这条说明"3 步 <1e-10 不可能"必须限定"数值秩 5 且 σ₅/σ₁ ≥ 1e-3"。
    s = np.array([1.0, 1.0, 1.0, 1e-12, 1e-12])
    b = with_singular_values(60, 60, s, seed=3)
    fb = aca_lowrank(b, tol=1e-10)
    ok("代数秩 5 但 σ₄=σ₅=1e-12:ACA 判为数值秩 3", fb.rank == 3, f"{fb.rank}")
    ok("该反例下 3 步残差 < 1e-10(原句按字面可被满足)", ref_error(b, fb) < 1e-10,
       f"{ref_error(b, fb):.2e}")

    # 孪生对照:秩-2 矩阵 1 步不够、2 步精确
    a2 = rank_k(30, 30, 2, seed=4)
    g1 = aca_lowrank(a2, tol=0.0, max_rank=1)
    g2 = aca_lowrank(a2, tol=1e-10)
    ok("秩-2:1 步残差 > 1e-3", ref_error(a2, g1) > 1e-3, f"{ref_error(a2, g1):.3e}")
    report("秩-2:2 步残差", ref_error(a2, g2), 1e-12)


# ============================================================================
# ③ 契约 + 选行规则 + 回退导出量
# ============================================================================
def section_contract() -> None:
    print("\n[③ 契约:逐层重算 + 选行规则 + 回退导出量]")
    fixtures = [("40x35 complex", rank_k(40, 35, 4, seed=5, complex_=True), 1e-12),
                ("25x70 real", rank_k(25, 70, 3, seed=6), 1e-12)]
    for name, a, tol in fixtures:
        fac = aca_lowrank(a, tol=tol)
        ce = contract_error(fac, a)
        report(f"{name}: u 逐层相对误差", ce["u"], 1e-12)
        report(f"{name}: v 逐层相对误差", ce["v"], 1e-12)
        ok(f"{name}: 每步主元 = 该残差行的最大元", ce["pivot"])
        ok(f"{name}: u[i_k, k] = 1", ce["identity"])
        ok(f"{name}: 选行规则(argmax|u|,否则必须是残差为零的回退)",
           not pivot_rule_violations(fac, a), f"{pivot_rule_violations(fac, a)}")

        # residual 曲线每一步独立复算
        worst = 0.0
        u_p = np.zeros((a.shape[0], 0), dtype=a.dtype)
        v_p = np.zeros((0, a.shape[1]), dtype=a.dtype)
        for step in range(fac.rank):
            u_p = np.concatenate([u_p, fac.u[:, [step]]], axis=1)
            v_p = np.concatenate([v_p, fac.v[[step], :]], axis=0)
            rec = float(np.linalg.norm(a - u_p @ v_p) / np.linalg.norm(a))
            worst = max(worst, abs(rec - float(fac.residual[step + 1])))
        report(f"{name}: residual 曲线独立复算", worst, 1e-12)
        ok(f"{name}: pivots 无重复(同一列不会二次成为主元)",
           len(set(fac.pivot_cols.tolist())) == fac.rank)

    # ---- 三个 fixture 的 pivot_rows / n_fallbacks 精确值(钉住 |u| 回退顺序) ----
    print("  —— 回退路径:pivot_rows / n_fallbacks 精确值 ——")
    a60 = rank_k(60, 60, 5, seed=1, complex_=True)
    f60 = aca_lowrank(a60, tol=1e-10)
    ok("60×60: pivot_rows 精确", f60.pivot_rows.tolist() == [0, 6, 35, 58, 40],
       f"{f60.pivot_rows.tolist()}")
    ok("60×60: n_fallbacks = 1", f60.n_fallbacks == 1, f"{f60.n_fallbacks}")

    d = np.diag([0.5, 0.1, 0.01, 1e-3, 1e-5]).astype(float)
    fd = aca_lowrank(d, tol=1e-13)
    ce_d = contract_error(fd, d)
    ok("对角秩-5:精确恢复", ref_error(d, fd) < 1e-13, f"{ref_error(d, fd):.2e}")
    ok("对角秩-5: pivot_rows 精确", fd.pivot_rows.tolist() == [0, 1, 2, 3, 4],
       f"{fd.pivot_rows.tolist()}")
    ok("对角秩-5: n_fallbacks = 4", fd.n_fallbacks == 4, f"{fd.n_fallbacks}")
    ok("对角(回退路径): u[i_k,k] = 1", ce_d["identity"])
    ok("对角(回退路径): 逐层 u/v 误差为 0", ce_d["u"] <= 1e-12 and ce_d["v"] <= 1e-12)

    a0 = rank_k(10, 10, 3, seed=8)
    a0[0, :] = 0.0
    f0 = aca_lowrank(a0, tol=1e-13)
    ce_0 = contract_error(f0, a0)
    ok("首行全零:精确恢复", ref_error(a0, f0) < 1e-13, f"{ref_error(a0, f0):.2e}")
    ok("首行全零: pivot_rows 精确", f0.pivot_rows.tolist() == [1, 2, 3],
       f"{f0.pivot_rows.tolist()}")
    ok("首行全零: n_fallbacks = 3", f0.n_fallbacks == 3, f"{f0.n_fallbacks}")
    ok("首行全零(回退路径): u[i_k,k] = 1", ce_0["identity"])


# ============================================================================
# ④ 导出量:rel_error 自证 / start_row / residual 语义
# ============================================================================
def section_exports() -> None:
    print("\n[④ 导出量互证]")
    for name, a, seed in (("60×60 complex", rank_k(60, 60, 5, seed=1, complex_=True), 1),
                          ("200×200 complex", rank_k(200, 200, 8, seed=14, complex_=True), 14)):
        fac = aca_lowrank(a, tol=1e-10)
        report(f"{name}: |rel_error − 独立 ref|",
               abs(fac.rel_error(a) - ref_error(a, fac)), 1e-15)

    # start_row:公开参数,必须真的被采用
    a = rank_k(10, 10, 3, seed=9, complex_=True)
    for sr in (0, 5, 9):
        f = aca_lowrank(a, tol=1e-13, start_row=sr)
        ok(f"start_row={sr}: 首个 pivot_row = {sr}", int(f.pivot_rows[0]) == sr,
           f"{f.pivot_rows[0]}")
        report(f"start_row={sr}: 仍精确恢复", ref_error(a, f), 1e-13)
    for bad in (10, 11, -1):
        try:
            aca_lowrank(a, start_row=bad)
            good = False
        except ValueError:
            good = True
        ok(f"start_row={bad} -> ValueError", good)

    # residual[0] 与 max_rank=0 的语义
    f0 = aca_lowrank(a, tol=1e-10, max_rank=0)
    ok("max_rank=0(非零矩阵): rank 0、未收敛、residual = [1.0]",
       f0.rank == 0 and not f0.converged and np.array_equal(f0.residual, np.array([1.0])),
       f"{f0.residual}")
    # tol=0:只有残差恰好 0 才算收敛
    fz = aca_lowrank(a, tol=0.0)
    ok("tol=0(秩-3 矩阵): 数值秩耗尽而非 tol 收敛", not fz.converged,
       f"converged={fz.converged}, r={fz.residual[-1]:.2e}")
    ok("tol=0: 最终残差仍到机器精度", float(fz.residual[-1]) <= 1e-12)


# ============================================================================
# ⑤ 边界与退化
# ============================================================================
def section_edges() -> None:
    print("\n[⑤ 边界与退化]")
    z = np.zeros((6, 9))
    fz = aca_lowrank(z)
    ok("零矩阵: rank 0 且 converged", fz.rank == 0 and fz.converged)
    ok("零矩阵: reconstruct 全零", not np.any(fz.reconstruct()))
    ok("零矩阵: residual = [0]", np.array_equal(fz.residual, np.array([0.0])),
       f"{fz.residual}")
    ok("零矩阵: rel_error = 0(不是 0/0 的 nan)", fz.rel_error(z) == 0.0,
       f"{fz.rel_error(z)!r}")

    ok("空矩阵 0x5: rank 0", aca_lowrank(np.zeros((0, 5))).rank == 0)
    ok("空矩阵 5x0: rank 0", aca_lowrank(np.zeros((5, 0))).rank == 0)

    a1 = rank_k(1, 7, 1, seed=10)
    f1 = aca_lowrank(a1)
    ok("1x7 秩-1: rank 1 且精确", f1.rank == 1 and ref_error(a1, f1) < 1e-13)
    b1 = rank_k(7, 1, 1, seed=11)
    ok("7x1 秩-1: rank 1 且精确", aca_lowrank(b1).rank == 1
       and ref_error(b1, aca_lowrank(b1)) < 1e-13)

    # 首列全零
    a = rank_k(10, 10, 3, seed=12)
    a[:, 0] = 0.0
    ok("首列全零:仍精确恢复", ref_error(a, aca_lowrank(a, tol=1e-13)) < 1e-13)

    # 非连续视图
    big = rank_k(20, 20, 2, seed=13)
    view = big[::2, ::2]
    ok("非连续视图:可用且精确",
       ref_error(view, aca_lowrank(view, tol=1e-13)) < 1e-13)

    # dtype
    fi = aca_lowrank(np.arange(12).reshape(3, 4).astype(int))
    ok("整数输入 -> float64", fi.u.dtype == np.float64, f"{fi.u.dtype}")
    fc = aca_lowrank(rank_k(8, 8, 2, seed=15, complex_=True))
    ok("复数输入 -> complex128", fc.u.dtype == np.complex128, f"{fc.u.dtype}")

    # 确定性
    a = rank_k(16, 16, 3, seed=16, complex_=True)
    x, y = aca_lowrank(a), aca_lowrank(a)
    ok("确定性:两次调用逐位相同",
       np.array_equal(x.u, y.u) and np.array_equal(x.v, y.v)
       and np.array_equal(x.pivot_rows, y.pivot_rows)
       and np.array_equal(x.residual, y.residual))

    # NaN 输入必须拒绝(静默返回 rank 0 会把坏数据洗成"正常结果")
    try:
        aca_lowrank(np.array([[np.nan, 1.0], [0.0, 1.0]]))
        good = False
    except ValueError:
        good = True
    ok("NaN 输入 -> ValueError", good)


# ============================================================================
# ⑥ 单调性反例 + 规模 sanity
# ============================================================================
def section_counterexample_and_scale() -> None:
    print("\n[⑥ 单调性反例(文档里已声明不是不变量) + 规模 sanity]")
    a2 = np.array([[0.01, 0.01], [1.0, -1.0]])
    f2 = aca_lowrank(a2, tol=1e-12)
    ok("2×2 反例:残差**不**单调(1.0 → 1.414)", float(f2.residual[1]) > float(f2.residual[0]),
       f"{f2.residual}")
    report("2×2 反例:仍然精确恢复", ref_error(a2, f2), 1e-12)

    a = rank_k(200, 200, 8, seed=17, complex_=True)
    f = aca_lowrank(a, tol=1e-10)
    ok("200×200 秩-8: rank = 8", f.rank == 8, f"rank {f.rank}")
    report("200×200 秩-8: 独立相对误差", ref_error(a, f), 1e-12)
    ok("U/V 元素数 = r(m+n)", f.u.size + f.v.size == f.rank * (200 + 200),
       f"{f.u.size + f.v.size}")


def section_validation() -> None:
    print("\n[⑦ 输入校验]")
    cases = [
        ("一维输入 -> ValueError", lambda: aca_lowrank(np.zeros(5))),
        ("非数值 dtype -> ValueError", lambda: aca_lowrank(np.array([["a", "b"]]))),
        ("tol<0 -> ValueError", lambda: aca_lowrank(np.zeros((3, 3)), tol=-1.0)),
        ("tol=inf -> ValueError", lambda: aca_lowrank(np.zeros((3, 3)), tol=float("inf"))),
        ("max_rank<0 -> ValueError", lambda: aca_lowrank(np.zeros((3, 3)), max_rank=-1)),
    ]
    for label, fn in cases:
        try:
            fn()
            good = False
        except ValueError:
            good = True
        ok(label, good)


def main() -> int:
    print("=" * 78)
    print("3.2 自检:部分主元 ACA(src/aca.py)")
    print("=" * 78)
    section_recovery()
    section_impossible()
    section_contract()
    section_exports()
    section_edges()
    section_counterexample_and_scale()
    section_validation()

    print("\n" + "=" * 78)
    total = COUNTS["disc"] + COUNTS["base"]
    if failures:
        print(f"结果:FAIL({len(failures)} 项)-> {failures}")
        print(f"计数:{total} = {COUNTS['disc']} 有鉴别力 + {COUNTS['base']} 基线")
        return 1
    print(f"结果:PASS —— {total} 项 = {COUNTS['disc']} 有鉴别力 + {COUNTS['base']} 基线")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
