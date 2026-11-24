# -*- coding: utf-8 -*-
"""3.2 的**鉴别力审计**(元测试):把 src/aca.py 逐个改坏,check_aca.py 必须变红。

为什么这条要进仓库:
- 3.1 的复核(2026-10-19)实测发现旧自检里有 8 条分支删掉后仍全绿(D14);
- 3.2 的第 1 轮复核(2026-11-24)又抓到 **13 个逃逸变异体**:`rel_error` 桩成 0、
  `start_row` 忽略、部分主元选行换成 0、回退顺序/计数/pivot_rows 全可改、
  `residual[0]` 改 1.5、`max_rank=0` 的 residual 改 0 —— 59 项自检全部照绿。
  本文件把那次的攻击固化成可复跑的命令;现在每一个都必须红。

判据:
    · 每个变异体都 **exit != 0**;
    · 若指定 `expect_label`,该标签必须出现在 `[FAIL]` 行里(不许靠 traceback 蒙混);
    · 未变异的基线必须 exit 0。

运行:
    python tests/mut_check_aca.py        # 期望 exit 0
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "aca.py"
TEST = ROOT / "tests" / "check_aca.py"
WORK = ROOT / "_scratch" / "mutcheck_aca"


def _prepare(case: str, patch=None) -> Path:
    d = WORK / case
    if d.exists():
        shutil.rmtree(d)
    (d / "src").mkdir(parents=True)
    (d / "tests").mkdir()
    text = SRC.read_text(encoding="utf-8")
    if patch is not None:
        patched = patch(text)
        assert patched != text, f"{case}: 变异没生效(原文不匹配)"
        text = patched
    (d / "src" / "aca.py").write_text(text, encoding="utf-8")
    shutil.copy2(TEST, d / "tests" / "check_aca.py")
    return d


def _run(d: Path) -> tuple[int, list[str]]:
    cp = subprocess.run([sys.executable, str(d / "tests" / "check_aca.py")],
                        capture_output=True, text=True, encoding="utf-8",
                        errors="replace",
                        env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    fails = [ln.strip() for ln in cp.stdout.splitlines() if "[FAIL]" in ln]
    if cp.returncode != 0 and not fails:
        tail = cp.stderr.strip().splitlines()
        fails = ["<无 [FAIL] 行,疑似 traceback>:" + (tail[-1][:140] if tail else "<无输出>")]
    return cp.returncode, fails


def _rep(text: str, old: str, new: str) -> str:
    assert old in text, f"找不到待替换代码:\n{old[:160]}"
    return text.replace(old, new, 1)


U_APPEND = "        u = np.concatenate([u, (col / delta).reshape(m, 1)], axis=1)"
V_APPEND = "        v = np.concatenate([v, row.reshape(1, n)], axis=0)"
RETURN = "    return ACAFactor(\n        u=u, v=v,"
RESID = "        residual.append(float(np.linalg.norm(a - u @ v)) / norm_a)"
STOP = "        if residual[-1] <= tol:\n            converged = True\n            break"
TOLCHK = "    if not np.isfinite(tol) or tol < 0:\n        raise ValueError(f\"tol 必须是有限非负数,收到 {tol!r}\")"
REL_ERR = "        return float(np.linalg.norm(a - self.reconstruct()) / den)"
START_ROW = "    i = int(start_row) if m else 0"
ARGMAX_ROW = "        i = int(np.argmax(np.abs(u[:, -1])))"
FB_RET = "        n_fallbacks=n_fallbacks)"
PIV_APPEND = "        pivot_rows.append(i_used)"
ORDER = "    order = np.argsort(-np.abs(u[:, -1]), kind=\"stable\") if u.shape[1] else np.arange(m)"
RESID0 = "    residual = [1.0]"

CASES: list[tuple[str, object, str | None]] = [
    # ---- A. 因子被换坏(输出形状对、内容错) ----
    ("MUT_zero_factors", lambda t: _rep(t, RETURN, RETURN.replace("u=u, v=v,", "u=u * 0, v=v * 0,")),
     "U V"),
    ("MUT_drop_last_factor",
     lambda t: _rep(t, RETURN, RETURN.replace("u=u, v=v,", "u=u[:, :-1], v=v[:-1, :],")),
     "U V"),
    ("MUT_stale_row_not_residual",
     lambda t: _rep(t, V_APPEND, V_APPEND.replace("row.reshape(1, n)", "a[i_used, :].reshape(1, n)")),
     "v 逐层相对误差"),
    ("MUT_u_wrong_scale",
     lambda t: _rep(t, U_APPEND, U_APPEND.replace("(col / delta)", "(col / (2.0 * delta))")),
     "u 逐层相对误差"),

    # ---- B. 残差/停止被换坏 ----
    ("MUT_residual_zero", lambda t: _rep(t, RESID, "        residual.append(0.0)  # MUTANT"),
     "U V"),
    ("MUT_residual_max_norm",
     lambda t: _rep(t, RESID,
                    "        residual.append(float(np.max(np.abs(a - u @ v))) / norm_a)"),
     "residual 曲线独立复算"),
    ("MUT_never_converge", lambda t: _rep(t, STOP, "        pass  # MUTANT"),
     "converged = True"),

    # ---- C. 输入校验/退化分支被删 ----
    ("MUT_tol_validation_gone", lambda t: _rep(t, TOLCHK, "    pass  # MUTANT"),
     "tol<0 -> ValueError"),
    ("MUT_zero_matrix_residual",
     lambda t: _rep(t, "        resid = np.array([0.0 if norm_a == 0.0 else 1.0])",
                    "        resid = np.array([1.0])  # MUTANT"),
     "零矩阵: residual = [0]"),
    ("MUT_no_pivot_fallback", lambda t: _rep(t, "    for i in order:", "    for i in []:  # MUTANT"),
     "对角秩-5:精确恢复"),
    ("MUT_complex_cast_to_real",
     lambda t: _rep(t, "    dtype = np.result_type(a.dtype, np.float64)",
                    "    dtype = np.float64  # MUTANT"),
     "复数输入 -> complex128"),

    # ---- D. 第 1 轮复核抓到的 8 条 high(2026-11-24 补) ----
    ("MUT_rel_error_zero", lambda t: _rep(t, REL_ERR, "        return 0.0  # MUTANT"),
     "|rel_error − 独立 ref|"),
    ("MUT_rel_error_negative", lambda t: _rep(t, REL_ERR, "        return -1e-16  # MUTANT"),
     "|rel_error − 独立 ref|"),
    ("MUT_start_row_ignored", lambda t: _rep(t, START_ROW, "    i = 0  # MUTANT"),
     "start_row=5: 首个 pivot_row = 5"),
    ("MUT_partial_pivot_row0", lambda t: _rep(t, ARGMAX_ROW, "        i = 0  # MUTANT"),
     "n_fallbacks = 1"),
    ("MUT_nfallback_always", lambda t: _rep(t, FB_RET, "        n_fallbacks=n_fallbacks + 4)"),
     "n_fallbacks = 1"),
    ("MUT_pivot_rows_requested", lambda t: _rep(t, PIV_APPEND, "        pivot_rows.append(i)  # MUTANT"),
     "对角秩-5: pivot_rows 精确"),
    ("MUT_fallback_order", lambda t: _rep(t, ORDER, "    order = np.arange(m)  # MUTANT"),
     "pivot_rows 精确"),
    ("MUT_residual0_1p5", lambda t: _rep(t, RESID0, "    residual = [1.5]  # MUTANT"),
     "residual[0] = 1"),
    ("MUT_maxrank0_residual0",
     lambda t: _rep(t, "        resid = np.array([0.0 if norm_a == 0.0 else 1.0])",
                    "        resid = np.array([0.0])  # MUTANT"),
     "max_rank=0(非零矩阵)"),
]


def main() -> int:
    WORK.mkdir(parents=True, exist_ok=True)

    base = _prepare("_BASELINE")
    rc, fails = _run(base)
    print(f"{'基线(未变异)':<34s} exit={rc} 失败项={fails[:3]}")
    if rc != 0:
        print("!! 基线都不过,先修 check_aca.py")
        return 1

    bad: list[str] = []
    for name, patch, label in CASES:
        d = _prepare(name, patch)
        rc, fails = _run(d)
        caught = rc != 0
        hit = (label is None) or any(label in ln for ln in fails)
        print(f"{name:<34s} exit={rc} {'抓到' if caught and hit else '逃逸'}  {fails[:2]}")
        if not (caught and hit):
            bad.append(f"{name}(exit={rc}, 期望标签={label}, 实际={fails[:2]})")

    print()
    if bad:
        print(f"结果: {len(bad)} 个变异体逃逸 -> {bad}")
        return 1
    print(f"结果: 全部通过({len(CASES)} 个变异体全部被抓到;基线 exit 0)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
