# -*- coding: utf-8 -*-
"""打印当前可用内存,以及关键 N 的内存守卫判定(**只读,不跑任何实验**)。

用途:关掉占内存的程序后,先跑这一行确认够不够再开跑。
    python experiments/check_mem.py
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

spec = importlib.util.spec_from_file_location(
    "exp02", ROOT / "experiments" / "exp02_dense_scaling.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

ms = m.system_memory()
G = 2 ** 30
print(f"总物理 {ms['total_phys_b'] / G:.2f} GiB | 可用 {ms['avail_phys_b'] / G:.2f} GiB"
      f" | 占用 {ms['memory_load_pct']}%")
print("守卫:need + 1.0 GiB ≤ 0.85×可用 且 need ≤ 0.5×总量")
for n in (11600, 12000, 14000):
    a = m.memory_admit(n)
    need_free = (a["need_b"] + m.SAFETY_MARGIN_B) / 0.85 / G
    print(f"  N={n:<6}: need {a['need_b'] / G:.2f} GiB -> "
          f"{'✅ 通过' if a['admit'] else '❌ 不通过'}"
          f"(需要可用 ≥ {need_free:.2f} GiB,当前 {a['free_b'] / G:.2f} GiB)")
