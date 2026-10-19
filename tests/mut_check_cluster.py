# -*- coding: utf-8 -*-
"""3.1 的**鉴别力审计**(元测试):把 src/cluster.py 或 check_cluster.py 逐个改坏,
自检必须变红。

为什么这条也要进仓库(而不是只在复核员手里跑一次):
    第三方复核(2026-10-19)实测:旧版 `check_cluster.py` 里删掉 nesting / order /
    leaf_size / depth 公式 / levels_ok / count_ok 任一支,**仍然 33 项全绿**;
    而把 `indices` 换成 `np.arange(start, stop)`、把 `order_by_angle` 换成常数旋转,
    也全绿。也就是说那些判据当时**没有专门的反例**(D14),`indices` / `order_by_angle`
    这两个导出量更是**零覆盖**(D22)。
    本脚本把那次的攻击固化成可复跑的命令 —— 以后任何人删掉一条分支,这里必须红。

判据(每一条都要满足):
    · 每个变异体都 **exit != 0**;
    · 若指定了 `expect_label`,该标签必须出现在 `[FAIL]` 行里(不许靠 traceback 蒙混);
    · 未变异的基线必须 exit 0(否则"全红"也算通过,等于没测)。

不变量:本脚本只**复制**到 `_scratch/mutcheck/` 下改副本,不碰 proj/src 与 proj/tests。

运行:
    python tests/mut_check_cluster.py        # 期望 exit 0
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
TEST = ROOT / "tests" / "check_cluster.py"
WORK = ROOT / "_scratch" / "mutcheck"

SRC_FILES = ("cluster.py", "mom.py", "quad.py", "green.py")


def _prepare(case: str, *, src_patch=None, test_patch=None) -> Path:
    """建一个最小可运行的副本目录 `_scratch/mutcheck/<case>/{src,tests}`。"""
    d = WORK / case
    if d.exists():
        shutil.rmtree(d)
    (d / "src").mkdir(parents=True)
    (d / "tests").mkdir()
    for name in SRC_FILES:
        shutil.copy2(SRC / name, d / "src" / name)
    if src_patch is not None:
        f = d / "src" / "cluster.py"
        text = f.read_text(encoding="utf-8")
        patched = src_patch(text)
        assert patched != text, f"{case}: src 变异没生效(原文不匹配)"
        f.write_text(patched, encoding="utf-8")
    text = TEST.read_text(encoding="utf-8")
    if test_patch is not None:
        patched = test_patch(text)
        assert patched != text, f"{case}: test 变异没生效(原文不匹配)"
        text = patched
    (d / "tests" / "check_cluster.py").write_text(text, encoding="utf-8")
    return d


def _run(d: Path) -> tuple[int, list[str]]:
    cp = subprocess.run([sys.executable, str(d / "tests" / "check_cluster.py")],
                        capture_output=True, text=True, encoding="utf-8",
                        errors="replace",
                        env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
    fails = [ln.strip() for ln in cp.stdout.splitlines() if "[FAIL]" in ln]
    if cp.returncode != 0 and not fails:
        fails = ["<无 [FAIL] 行,疑似 traceback>:" + cp.stderr.strip().splitlines()[-1][:120]
                 if cp.stderr.strip() else "<无输出>"]
    return cp.returncode, fails


def _del(text: str, old: str, new: str) -> str:
    assert old in text, f"找不到待删除的代码块:\n{old[:120]}"
    return text.replace(old, new, 1)


# ============================================================================
# 变异清单
# ============================================================================
#: (名字, 作用对象, 变异函数, 期望被抓到的标签)
CASES: list[tuple[str, str, object, str | None]] = [
    # ---- A. 导出量实现被换成坏实现(D22) ----
    ("MUT_indices_returns_positions", "src",
     lambda t: _del(t,
                    "        return self.order[cluster.start:cluster.stop]",
                    "        return np.arange(cluster.start, cluster.stop)  # MUTANT"),
     None),                      # 几何检查会大面积红;不指定单一标签
    ("MUT_order_by_angle_no_sort", "src",
     lambda t: _del(t,
                    "    pts = _as_points(points)\n"
                    "    center = pts.mean(axis=0)\n"
                    "    ang = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])\n"
                    "    return np.argsort(ang, kind=\"stable\")",
                    "    pts = _as_points(points)\n"
                    "    return (np.arange(pts.shape[0]) + 7) % pts.shape[0]  # MUTANT"),
     "D4"),                      # 打乱输入恢复边界顺序的独立锚点必须红

    # ---- B. 检查器分支被删掉(D14) ----
    ("MUT_drop_partition", "test",
     lambda t: _del(t, "        if not (tile and mapping_ok):\n            out.append(\"partition\")",
                    "        pass  # MUTANT"), "K1"),
    ("MUT_drop_nesting", "test",
     lambda t: _del(t, "            out.append(\"nesting\")\n            break",
                    "            break  # MUTANT"), "K6"),
    ("MUT_drop_size", "test",
     lambda t: _del(t, "        if max(sizes) > leaf_size or max(sizes) - min(sizes) > 1 or sizes != expected:\n"
                       "            out.append(\"size\")",
                    "        pass  # MUTANT"), "K2"),
    ("MUT_drop_depth", "test",
     lambda t: _del(t, "    if not (tree.depth == d_ref and measured == tree.depth\n"
                       "            and all(c.level in (tree.depth, tree.depth - 1) for c in leaves)):\n"
                       "        out.append(\"depth\")",
                    "    pass  # MUTANT"), "K4"),
    ("MUT_drop_field", "test",
     lambda t: _del(t, "    if (tree.n != n or tree.leaf_size != leaf_size\n"
                       "            or tree.n_leaves != len(leaves) or tree.root.level != 0):\n"
                       "        out.append(\"field\")",
                    "    pass  # MUTANT"), "K7"),
    ("MUT_drop_malformed", "test",
     lambda t: _del(t, "    if bad is not None:\n        return [\"malformed\"]",
                    "    pass  # MUTANT"), None),
    ("MUT_drop_geometry", "test",
     lambda t: _del(t, "    if not (worst <= 1e-12 * scale):\n        out.append(\"geometry\")",
                    "    pass  # MUTANT"), "K5"),
    ("MUT_drop_angles", "test",
     lambda t: _del(t, "            if gap.size and np.max(np.abs(np.abs(gap) - 2.0 * np.pi / n)) > 1e-9:\n"
                       "                out.append(\"angles\")",
                    "            pass  # MUTANT"), "K3"),
]


def main() -> int:
    WORK.mkdir(parents=True, exist_ok=True)

    # 基线:未变异必须 exit 0,否则下面的"变红"没有意义。
    base = _prepare("_BASELINE")
    rc, fails = _run(base)
    print(f"{'基线(未变异)':<34s} exit={rc} 失败项={fails[:3]}")
    if rc != 0:
        print("!! 基线都不过,先修 check_cluster.py")
        return 1

    bad: list[str] = []
    for name, target, patch, label in CASES:
        d = _prepare(name,
                     src_patch=patch if target == "src" else None,
                     test_patch=patch if target == "test" else None)
        rc, fails = _run(d)
        caught = rc != 0
        hit_label = (label is None) or any(label in ln for ln in fails)
        verdict = "抓到" if (caught and hit_label) else "逃逸"
        print(f"{name:<34s} exit={rc} {verdict}  {fails[:2]}")
        if not (caught and hit_label):
            bad.append(f"{name}(exit={rc}, 期望标签={label}, 实际失败={fails[:2]})")

    print()
    if bad:
        print(f"结果: {len(bad)} 个变异体逃逸 -> {bad}")
        return 1
    print(f"结果: 全部通过({len(CASES)} 个变异体全部被抓到;基线 exit 0)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
