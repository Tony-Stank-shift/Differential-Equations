"""一键复现:按顺序运行全部实验,结果写入 results/,图写入 figures/。

用法:
    .venv/Scripts/python run_all.py          # 运行全部实验
    .venv/Scripts/python run_all.py e1 e4    # 只运行指定实验
"""
from __future__ import annotations

import importlib
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
# 实验模块住在 experiments/ 下,而 importlib 只认 sys.path —— 2026-10-13 补上。
# 缺这一行时 `importlib.import_module("exp02_dense_scaling")` 必然 ModuleNotFoundError,
# 会被上面的 except 当成"模块尚未实现"而静默 SKIP(即:真模块存在也会被报成不存在)。
sys.path.insert(0, str(ROOT / "experiments"))

# 实验编号 -> (模块名, 说明)
EXPERIMENTS: dict[str, tuple[str, str]] = {
    "e1": ("exp01_mie_validation", "MoM vs Mie series (correctness)"),
    "e2": ("exp02_dense_scaling", "dense solve time/memory vs N"),
    "e3": ("exp03_aca_rank", "ACA block rank characteristics"),
    "e4": ("exp04_hmatrix_scaling", "H-matrix assembly/solve vs N"),
    "e5": ("exp05_ka_sweep", "block rank vs electrical size ka"),
    "e6": ("exp06_nn_endtoend", "fixed rank / ACA / NN end-to-end"),
}


def main(argv: list[str]) -> int:
    wanted = argv[1:] or list(EXPERIMENTS)
    unknown = [k for k in wanted if k not in EXPERIMENTS]
    if unknown:
        print(f"unknown experiment id: {unknown}")
        print(f"available: {list(EXPERIMENTS)}")
        return 2

    print(f"running: {wanted}")
    print(f"start:   {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("-" * 60)

    not_done: list[str] = []
    for key in wanted:
        module_name, desc = EXPERIMENTS[key]
        print(f"[{key}] {desc}")
        t0 = time.perf_counter()
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            print(f"    SKIP: module {module_name} not implemented yet ({exc})")
            not_done.append(key)
            continue
        run = getattr(module, "run", None)
        if run is None:
            print(f"    SKIP: {module_name} has no run()")
            not_done.append(key)
            continue
        run()
        print(f"    done in {time.perf_counter() - t0:.1f} s")

    print("-" * 60)
    if not_done:
        print(f"not completed: {not_done}")
        return 1
    print("all experiments completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
