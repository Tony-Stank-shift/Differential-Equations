# -*- coding: utf-8 -*-
"""3.1 自检:二叉树聚类(src/cluster.py)。

本文件验三件事,顺序就是论文里要主张的顺序:

    ① **结构**:分割(不重不漏)、连续性(父 = 两子首尾相接)、均匀性(叶子大小 =
       ⌊N/k⌋/⌈N/k⌉、前 r 块各多 1)、深度(= ⌈log2 k⌉,k = ⌈N/leaf⌉);
    ② **契约**:`indices`/`leaf_indices` 必须真的做"几何位置 → 原始段号"的映射
       (含跨切口回绕)、`order` 必须原样采用调用方给的置换、`radius` 必须真的是
       包围半径的上界(不是随便一个正数);
    ③ **鉴别力**(D14)与**计数口径**(D24):拿 9 棵**故意做坏**的树喂同一个检查器,
       逐条分支都必须能红;报告时把"有鉴别力的判据"与"恒 PASS 的基线"分开数。

⚠️ 这份文件被第三方复核打回过一次(needs_revision,2026-10-19)。那次的四个 high:
    · `indices` 零判据 —— 换成 `np.arange(start,stop)` 后 33 项全绿;
    · `order_by_angle` 可被桩成"任意旋转"而全绿;
    · K 组只覆盖了 partition/geometry 两支,删掉 nesting/size/depth 仍全绿;
    · 声称"深度有两条独立口径",但代码里没有 2^d、也没有"同层"检查。
  现在这些都已改成可执行判据,并在本文里留档。

⚠️ 为什么"叶子均匀"值得单列一条判据(不只是审美)
    E3/E5 要测"块秩 vs 块大小 / vs ka"。块秩是**块大小**的函数:叶子若有 4 和 8 两种,
    量出来的秩分布就混进了"叶子怎么切的"这个假信号。所以均匀性是**测量口径**问题。
    详见 src/cluster.py 顶部与 DECISIONS D31。

运行:
    python tests/check_cluster.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cluster import Cluster, ClusterTree, build_cluster_tree, order_by_angle  # noqa: E402
from mom import circle_geometry  # noqa: E402

failures: list[str] = []
COUNTS = {"disc": 0, "base": 0}


def ok(label: str, cond: bool, detail: str = "", *, kind: str = "disc") -> None:
    COUNTS[kind] += 1
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}{('   ' + detail) if detail else ''}")
    if not cond:
        failures.append(label)


def report(name: str, value: float, tol: float, note: str = "", *, kind: str = "disc") -> None:
    good = np.isfinite(value) and value <= tol
    COUNTS[kind] += 1
    print(f"  [{'PASS' if good else 'FAIL'}] {name:<46s} = {value:.6e}  (tol {tol:.1e})"
          f"{('   ' + note) if note else ''}")
    if not good:
        failures.append(name)


def must_fail(label: str, violated: list[str], wanted: str, *, other: str = "") -> None:
    """孪生反例:坏树必须让检查器在 `wanted` 这一项上变红。"""
    COUNTS["disc"] += 1
    hit = wanted in violated
    print(f"  [{'PASS' if hit else 'FAIL'}] {label}")
    if hit:
        print(f"          → 抓到的违反项:{violated}")
        if other:
            print(f"          {other}")
    else:
        print(f"          → 只抓到 {violated},唯独没抓到 {wanted} —— 判据无鉴别力!")
        failures.append(label)


# ============================================================================
# 检查器:一棵树 + 原始中点坐标 -> 违反的不变量名列表
# ============================================================================
def _ceil_log2(k: int) -> int:
    """最小的 d 使 2^d ≥ k。**独立实现**(翻倍循环),不用 `bit_length`,也不用 `math.log2`。

    D8:参考值不能抄实现。`cluster._depth_for` 用 `(k-1).bit_length()`;
    这里用翻倍循环 —— 两条独立路径对上才叫交叉验证。
    """
    d, cap = 0, 1
    while cap < k:
        cap *= 2
        d += 1
    return d


def _walk(root: Cluster):
    """先序遍历;遇到"只有一个子节点"的畸形节点就停下来并报告。

    ⚠️ 检查器的职责是吃坏树。直接调 `tree.leaves()` 会在 `None.is_leaf` 上抛
    AttributeError,整套自检 traceback 退出(复核实测)—— 那不是"判红",是崩。
    """
    nodes: list[Cluster] = []
    stack = [root]
    bad = None
    while stack:
        nd = stack.pop()
        nodes.append(nd)
        has_l, has_r = nd.left is not None, nd.right is not None
        if has_l != has_r:
            bad = f"节点 [{nd.start},{nd.stop}) 只有一个子节点"
            break
        if has_l:
            stack.append(nd.right)
            stack.append(nd.left)
    return nodes, bad


def violations(tree, pts: np.ndarray, leaf_size: int, *,
               check_angles: bool = False) -> list[str]:
    """把 src/cluster.py 声称的不变量逐条查一遍,返回**违反项名字**列表(空 = 全过)。

    这个函数被第 ③ 节的坏树复用,所以它必须自己算、不看模块的中间量:
    · `indices` 是**被测导出量**,几何检查必须经过它来取成员(桩替换会当场露馅);
    · 参考深度用 `_ceil_log2`(独立实现),并与"递归实测的最大叶子层"互证;
    · `radius` 的上界性质在第 ② 节单独查(需要圆几何的角度信息)。
    """
    n = int(pts.shape[0])
    nodes, bad = _walk(tree.root)
    if bad is not None:
        return ["malformed"]
    leaves = [nd for nd in nodes if nd.is_leaf]
    out: list[str] = []

    # ---- 1. 导出字段自洽(复核实测:这两项不钉住时,改字段值零违反) ----
    if (tree.n != n or tree.leaf_size != leaf_size
            or tree.n_leaves != len(leaves) or tree.root.level != 0):
        out.append("field")

    # ---- 2. 分割性:叶子首尾相接覆盖 [0,n);且 leaf_indices 拼起来 == order ----
    if not leaves:
        out.append("partition")
    else:
        starts = [c.start for c in leaves]
        stops = [c.stop for c in leaves]
        tile = (starts[0] == 0 and stops[-1] == n
                and all(starts[i + 1] == stops[i] for i in range(len(leaves) - 1)))
        concat = np.concatenate([tree.indices(c) for c in leaves])
        mapping_ok = (concat.shape == tree.order.shape
                      and np.array_equal(concat, tree.order)
                      and np.array_equal(np.sort(concat), np.arange(n)))
        if not (tile and mapping_ok):
            out.append("partition")

    # ---- 3. 嵌套:父区间 = 两子首尾相接,且子层 = 父层 + 1 ----
    for nd in nodes:
        if nd.is_leaf:
            continue
        if not (nd.left.start == nd.start and nd.left.stop == nd.right.start
                and nd.right.stop == nd.stop
                and nd.left.level == nd.level + 1
                and nd.right.level == nd.level + 1):
            out.append("nesting")
            break

    # ---- 4. 叶子尺寸:上界 leaf_size、相差 ≤ 1、且恰好是 [q+1]*r + [q]*(k-r) ----
    if leaves:
        k = -(-n // leaf_size)
        q, r = divmod(n, k)
        expected = [q + 1] * r + [q] * (k - r)
        sizes = [c.size for c in leaves]
        if max(sizes) > leaf_size or max(sizes) - min(sizes) > 1 or sizes != expected:
            out.append("size")

    # ---- 5. 深度:字段 == ⌈log2 k⌉ == 递归实测的最大叶子层;叶子层 ∈ {d, d−1} ----
    measured = max((c.level for c in leaves), default=-1)
    d_ref = _ceil_log2(-(-n // leaf_size))
    if not (tree.depth == d_ref and measured == tree.depth
            and all(c.level in (tree.depth, tree.depth - 1) for c in leaves)):
        out.append("depth")

    # ---- 6. 几何量:center / radius 必须与"该节点自称的成员集合"一致 ----
    scale = max(1.0, float(np.max(np.abs(pts))))
    worst = 0.0
    for nd in nodes:
        idx = tree.indices(nd)
        if idx.size == 0:
            worst = float("inf")
            break
        sub = pts[idx]
        c = sub.mean(axis=0)
        rr = float(np.max(np.linalg.norm(sub - c, axis=1)))
        worst = max(worst, float(np.max(np.abs(c - nd.center))), abs(rr - nd.radius))
    if not (worst <= 1e-12 * scale):
        out.append("geometry")

    # ---- 7. (仅圆几何)相邻段的角间隙绝对值恒为 2π/N(顺时针也接受) ----
    if check_angles and n >= 3:
        if not np.array_equal(np.sort(tree.order), np.arange(n)):
            out.append("angles")          # order 不是置换 ⇒ 谈不上"沿边界连续"
        else:
            c0 = pts.mean(axis=0)
            ang = np.arctan2(pts[:, 1] - c0[1], pts[:, 0] - c0[0])
            a = np.unwrap(ang[tree.order])
            gap = np.diff(a)
            if gap.size and np.max(np.abs(np.abs(gap) - 2.0 * np.pi / n)) > 1e-9:
                out.append("angles")

    return out


# ============================================================================
# 判据 A —— 结构不变量(N × leaf 网格)
# ============================================================================
print("=" * 86)
print("判据 A —— 结构:分割 / 嵌套 / 尺寸 / 深度 / 字段 / 几何量")
print("=" * 86)
print("  ⚠️ 深度有**两条独立路径**:① 公式 ⌈log2 k⌉(本文件用翻倍循环独立实现)")
print("     ② 递归实测的最大叶子层 + 根层为 0。只断言①等于让测试抄一遍实现。")
print("  ⚠️ 抽到的 N=17/leaf=8 是对半切规则的经典反例:N=17,leaf=8 ——")
print("     教科书规则给 {8,4,5}(最大/最小=2),本模块给 {6,6,5}(相差=1)。")

GRID = [(3, 8), (17, 8), (64, 1), (64, 8), (100, 7),
        (640, 16), (1000, 8), (1024, 32), (4096, 64)]

for n, leaf in GRID:
    geom = circle_geometry(1.0, n)
    pts = geom["mid"]
    tree = build_cluster_tree(geom, leaf)
    bad = violations(tree, pts, leaf, check_angles=True)
    sizes = [c.size for c in tree.leaves()]
    ok(f"A  N={n:<5} leaf={leaf:<3} depth={tree.depth:<2} "
       f"叶子数={len(tree.leaves()):<5} 尺寸={sizes[:6]}{'…' if len(sizes) > 6 else ''}",
       bad == [], f"违反:{bad}" if bad else "")

# ============================================================================
# 判据 A2 —— 小 N 穷举
# ============================================================================
print()
print("=" * 86)
print("判据 A2 —— 穷举 N=1..257 × leaf∈{1,2,3,5,7,8,16,17,32,64}")
print("=" * 86)
print("  ⚠️ 上面的网格是**抽样**(9 组)。『叶子均匀』是构造性保证,但构造性保证也要被")
print("     穷举打一遍 —— 只看抽样,无法排除某组 (N, leaf) 下 max−min 变成 2。")

_N_EXHAUST = 257
_LEAVES = (1, 2, 3, 5, 7, 8, 16, 17, 32, 64)
_exh_bad: list[str] = []
_exh_count = 0
for _n in range(1, _N_EXHAUST + 1):
    _phi = np.linspace(0.0, 2.0 * np.pi, _n, endpoint=False)
    _p = np.column_stack([np.cos(_phi), np.sin(_phi)])   # 直接造点,绕开 circle_geometry 的 N>=3
    for _leaf in _LEAVES:
        _t = build_cluster_tree(_p, _leaf)
        _bad = violations(_t, _p, _leaf, check_angles=(_n >= 3))
        _exh_count += 1
        if _bad:
            _exh_bad.append(f"N={_n},leaf={_leaf}:{_bad}")
ok(f"A2 {_exh_count} 组 (N≤{_N_EXHAUST}, leaf∈{len(_LEAVES)} 个值) 零违反",
   not _exh_bad, f"最早 3 个失败:{_exh_bad[:3]}" if _exh_bad else "")

# ============================================================================
# 判据 A3 —— 非圆几何 + 调用方自带的 order
# ============================================================================
print()
print("=" * 86)
print("判据 A3 —— 非圆几何:默认(极角)顺序 vs 调用方自带顺序")
print("=" * 86)

_rng = np.random.default_rng(20261019)
_pts_rand = _rng.standard_normal((300, 2)) * 3.0          # 一团无序点,不是星形边界
_t_rand = build_cluster_tree(_pts_rand, 16)
ok("A3a 无序点云:结构不变量成立(order 由极角给出)",
   violations(_t_rand, _pts_rand, 16) == [],
   kind="base")   # 对任意 order 都真 ⇒ 不是 order 语义的判据,只当随机几何的冒烟基线

#: 调用方自带顺序:按 x 排序。对非星形/非圆的几何,调用方知道什么顺序才是"沿边界"。
_order_x = np.argsort(_pts_rand[:, 0], kind="stable")
_t_custom = build_cluster_tree(_pts_rand, 16, order=_order_x)
ok("A3b 调用方给的 order 被**原样采用**(不是被默认排序悄悄替换)",
   np.array_equal(_t_custom.order, _order_x))
ok("A3c 自带 order 下结构不变量仍成立", violations(_t_custom, _pts_rand, 16) == [],
   "⚠️ 但『连续』的物理含义由调用方负责,本模块只保证沿给定顺序连续", kind="base")

# ============================================================================
# 判据 B —— 几何量(按成员点重算)+ 几何连续性 + radius 的上界性质
# ============================================================================
print()
print("=" * 86)
print("判据 B —— 几何量的一致性 + 角间隙 + radius 是否为真上界")
print("=" * 86)
print("  ⚠️ 口径交代(D8):center/radius 的公式没有外部真值,这里的『重算』用的是同一个")
print("     公式(同样的输入、同样的求和顺序),所以它**不是独立证据**,而是断言")
print("     『节点存储的几何量 ↔ 它声称的成员集合』这两者没有错位 —— 错位才是真 bug")
print("     (K5 篡改半径、`indices` 桩替换都会红)。偏差恰为 0.0 是预期结果。")

for n, leaf in ((640, 8), (4096, 64), (1000, 8)):
    geom = circle_geometry(2.5, n)
    pts = geom["mid"]
    tree = build_cluster_tree(geom, leaf)
    worst = 0.0
    for node in tree.nodes():
        sub = pts[tree.indices(node)]
        c = sub.mean(axis=0)
        worst = max(worst, float(np.max(np.abs(c - node.center))),
                    abs(float(np.max(np.linalg.norm(sub - c, axis=1))) - node.radius))
    report(f"B  N={n} leaf={leaf} center/radius 按成员点重算最大偏差", worst, 1e-12,
           note=f"叶子直径/周长 = "
                f"{max(2 * c.radius for c in tree.leaves()) / (2 * np.pi * 2.5):.3f}",
           kind="base")

for n, leaf in ((640, 8), (4096, 64)):
    geom = circle_geometry(1.0, n)
    tree = build_cluster_tree(geom, leaf)
    c0 = geom["mid"].mean(axis=0)
    ang = np.arctan2(geom["mid"][:, 1] - c0[1], geom["mid"][:, 0] - c0[0])
    a = np.unwrap(ang[tree.order])
    report(f"B  N={n} 相邻段角间隙与 2π/N 的最大偏差",
           float(np.max(np.abs(np.abs(np.diff(a)) - 2*np.pi/n))), 1e-12,
           note="max||Δφ| − 2π/N|")

#: radius 的语义是"**上界**"(均值心半径,不是最小包围圆)。独立锚点:
#: 圆上一段连续弧的最小包围圆半径有闭式 —— 跨度 θ ≤ π 时为 R·sin(θ/2),否则为 R。
#: ⚠️ 但成员是**弦中点**,它们落在半径 `R·cos(π/N)` 的圆上(不是 R),所以闭式里的 R
#: 要换成 R_eff;否则根节点会差 1.2e-5 的假失配(复核实测)。
#: 断言 `radius >= MEC`(上界成立);高估倍数只作 [info] 报出,不当判据。
_R_B = 1.0
_geomB = circle_geometry(_R_B, 640)
_R_EFF = _R_B * np.cos(np.pi / 640)          # 弦中点所在圆的半径
_treeB = build_cluster_tree(_geomB, 8)
_phi = _geomB["phi_mid"]
_ratio = 0.0
_bound_ok = True
for _node in _treeB.nodes():
    _members = np.mod(_phi[_treeB.indices(_node)], 2.0 * np.pi)
    if _members.size < 2:
        continue
    _s = np.sort(_members)
    _span = 2.0 * np.pi - float(np.max(np.diff(np.append(_s, _s[0] + 2.0 * np.pi))))
    _mec = _R_EFF * np.sin(_span / 2.0) if _span <= np.pi else _R_EFF
    if _mec > 0:
        _ratio = max(_ratio, _node.radius / _mec)
    if _node.radius < _mec - 1e-12:
        _bound_ok = False
ok("B4 radius 确实是包围半径的**上界**(对 640/leaf=8 的每个节点,与弧的闭式 MEC 比)",
   _bound_ok, f"最大高估倍数 = {_ratio:.3f}(均值心半径不是 MEC,高估是保守的)")

# ============================================================================
# 判据 C —— 退化输入与坏参数
# ============================================================================
print()
print("=" * 86)
print("判据 C —— 退化输入 / 坏参数 / 非法 order")
print("=" * 86)

geom3 = circle_geometry(1.0, 3)
t3 = build_cluster_tree(geom3, 8)
ok("C1  N=3 < leaf=8 ⇒ 单叶、depth=0(退化输入,不参与『深度≈log2(N/leaf)』判据)",
   len(t3.leaves()) == 1 and t3.depth == 0)

t1 = build_cluster_tree(np.array([[0.0, 0.0]]), 8)
ok("C2  N=1 ⇒ 单叶、depth=0、radius=0",
   t1.depth == 0 and len(t1.leaves()) == 1 and abs(t1.root.radius) < 1e-15)

t64 = build_cluster_tree(circle_geometry(1.0, 64), 1)
ok("C3  leaf=1 ⇒ 每个叶子恰 1 段、depth=6",
   all(c.size == 1 for c in t64.leaves()) and t64.depth == 6)

bad_args = 0
for leaf in (0, -3, 2.5, True):
    try:
        build_cluster_tree(geom3, leaf)
    except ValueError:
        bad_args += 1
ok("C4  leaf_size ∈ {0, -3, 2.5, True} 全部抛 ValueError", bad_args == 4, f"({bad_args}/4)")

def _raises_valueerror(fn, label):
    try:
        fn()
    except ValueError:
        ok(label, True)
    except Exception as exc:  # noqa: BLE001 —— 这里就是要抓"抛错了类型"
        ok(label, False, f"抛的是 {type(exc).__name__} —— 调用方 except ValueError 会漏掉")
    else:
        ok(label, False, "根本没有抛")


_raises_valueerror(
    lambda: build_cluster_tree(np.array([[0.0, np.nan], [1.0, 0.0], [0.0, 1.0]]), 2),
    "C5  中点含 NaN ⇒ ValueError")
_raises_valueerror(
    lambda: build_cluster_tree(geom3, 2, order=np.array([0, 0, 1])),
    "C6  order 不是置换 ⇒ ValueError")
_raises_valueerror(
    lambda: build_cluster_tree(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=complex),
                               2),
    "C7  复数中点 ⇒ ValueError(不是 TypeError;numpy 会静默丢虚部)")
_raises_valueerror(
    lambda: build_cluster_tree(geom3, 2, order=np.arange(2)),
    "C8  order 长度错 ⇒ ValueError")
_raises_valueerror(
    lambda: build_cluster_tree(geom3, 2, order=np.arange(3.0)),
    "C9  order 是 float dtype ⇒ ValueError")

# ============================================================================
# 判据 D —— 几何顺序 order 的性质
# ============================================================================
print()
print("=" * 86)
print("判据 D —— 几何顺序(order)")
print("=" * 86)

g = circle_geometry(1.0, 640)
o = order_by_angle(g)
ok("D1  order 是 0..N-1 的置换", np.array_equal(np.sort(o), np.arange(640)))

_c0 = g["mid"].mean(axis=0)
_ang = np.arctan2(g["mid"][:, 1] - _c0[1], g["mid"][:, 0] - _c0[0])
ok("D2  order[0] 就是极角最小的那段(切口钉在 ±π,不是任意旋转)",
   int(o[0]) == int(np.argmin(_ang)), f"order[0]={int(o[0])}, argmin(ang)={int(np.argmin(_ang))}")
ok("D3  order 与 circle_geometry 的 φ 顺序只差一个循环移位",
   np.array_equal(np.sort((o - o[0]) % 640), np.arange(640)))

#: **独立锚点**(复核员要求的):打乱输入数组顺序后,默认排序必须把边界顺序找回来。
#: 这是 `order_by_angle` 的"桩替换"判据 —— 返回任意旋转的假排序会在这里红。
_pn = 200
_phi_b = np.linspace(0.0, 2.0 * np.pi, _pn, endpoint=False)
_pts_b = np.column_stack([np.cos(_phi_b), np.sin(_phi_b)])   # 行号 = 边界顺序
_perm = _rng.permutation(_pn)
_pts_shuf = _pts_b[_perm]                                     # 打乱:行号 = 打乱后的编号
_t_shuf = build_cluster_tree(_pts_shuf, 8)
_bpos = _perm[_t_shuf.order]                                  # 每个几何位置对应的边界位置
ok("D4  打乱输入后仍恢复边界顺序(循环移位)—— order_by_angle 的独立锚点",
   np.array_equal((_bpos - _bpos[0]) % _pn, np.arange(_pn)),
   f"首元素边界位置 = {int(_bpos[0])}")

# ============================================================================
# 判据 E —— 跨极角切口的 leaf_indices 契约(复核员指出的下游陷阱)
# ============================================================================
print()
print("=" * 86)
print("判据 E —— 跨切口叶子的 indices 契约(原始段号会回绕,不是区间)")
print("=" * 86)

for n, leaf in ((640, 7), (1000, 13)):
    geomE = circle_geometry(1.0, n)
    treeE = build_cluster_tree(geomE, leaf)
    flat = np.concatenate(treeE.leaf_indices())
    ok(f"E  N={n} leaf={leaf}:leaf_indices 按叶序拼起来 == order",
       np.array_equal(flat, treeE.order))
    ok(f"E  N={n} leaf={leaf}:全部原始段号恰好各出现一次",
       np.array_equal(np.sort(flat), np.arange(n)))
    # 断言"回绕真的发生":至少有一个叶子的原始段号不是 +1 等差
    wraps = sum(1 for idx in treeE.leaf_indices() if idx.size > 1 and np.any(np.diff(idx) != 1))
    ok(f"E  N={n} leaf={leaf}:至少 1 个叶子的原始段号发生回绕(文档契约是真的)",
       wraps >= 1, f"回绕叶子数 = {wraps}")

# ============================================================================
# 判据 K —— 鉴别力
# ============================================================================
print()
print("=" * 86)
print("判据 K —— 鉴别力(坏树必须被同一个检查器抓到;D14)")
print("=" * 86)


def _mk(pts: np.ndarray, start: int, stop: int, level: int, left=None, right=None):
    """按模块约定手搓一个 Cluster(几何量照实算,以便只违反**想违反**的那一条)。"""
    sub = pts[start:stop]
    c = sub.mean(axis=0)
    r = float(np.max(np.linalg.norm(sub - c, axis=1)))
    return Cluster(start, stop, level, c, r, left, right)


def _tree(root, n, leaf, depth, n_leaves, order=None):
    return ClusterTree(root=root,
                       order=np.arange(n) if order is None else order,
                       n=n, leaf_size=leaf, depth=depth, n_leaves=n_leaves)


# K0:好树必须**零违反** —— 否则检查器"永远红",同样没有信息量(D14 的另一半)。
g17 = circle_geometry(1.0, 17)
p17 = g17["mid"]
good17 = build_cluster_tree(g17, 8)
v0 = violations(good17, p17, 8, check_angles=True)
ok("K0  N=17/leaf=8 的好树零违反(基线)", v0 == [], f"违反:{v0}" if v0 else "",
   kind="base")

# K1:左右子区间重叠 1 个下标 ⇒ 分割性必须红
r_k1 = _mk(p17, 0, 17, 0,
           _mk(p17, 0, 9, 1, _mk(p17, 0, 4, 2), _mk(p17, 4, 9, 2)),
           _mk(p17, 8, 17, 1, _mk(p17, 8, 12, 2), _mk(p17, 12, 17, 2)))
must_fail("K1  左右子区间重叠 1 个下标 ⇒ 必须抓到『partition』",
          violations(_tree(r_k1, 17, 8, 2, 3), p17, 8), "partition")

# K2:13/4 的偏斜切法 ⇒ 尺寸判据必须红(leaf=16 让其它条都不触发,隔离出这一条)
r_k2 = _mk(p17, 0, 17, 0, _mk(p17, 0, 13, 1), _mk(p17, 13, 17, 1))
must_fail("K2  N=17 切成 13/4 且 leaf=16 ⇒ 必须抓到『size』",
          violations(_tree(r_k2, 17, 16, 1, 2), p17, 16), "size",
          other="(leaf=16 时 13<=16、depth=1 都成立,故只有尺寸该红)")

# K3:order 被打乱(几何顺序不再连续)⇒ 角间隙必须红
p640 = circle_geometry(1.0, 640)["mid"]
perm = np.random.default_rng(20261019).permutation(640)
shuffled = build_cluster_tree(circle_geometry(1.0, 640), 8, order=perm)
must_fail("K3  把 order 换成随机置换 ⇒ 必须抓到『angles』(几何连续性有牙)",
          violations(shuffled, p640, 8, check_angles=True), "angles")

# K4:深度的**字段**被改大一层 ⇒ 深度判据必须红(结构没变,只有字段错)
g64 = circle_geometry(1.0, 64)
p64 = g64["mid"]
good64 = build_cluster_tree(g64, 8)
good64.depth += 1
must_fail("K4  只把 tree.depth 字段改大 1(结构不动)⇒ 必须抓到『depth』",
          violations(good64, p64, 8), "depth")

# K5:半径被篡改 10% ⇒ 几何量必须红
p64b = circle_geometry(1.0, 64)["mid"]
good64b = build_cluster_tree(circle_geometry(1.0, 64), 8)
good64b.root.left.radius *= 1.10
must_fail("K5  某个节点的 radius 被改大 10% ⇒ 必须抓到『geometry』",
          violations(good64b, p64b, 8), "geometry")

# K6:根区间比叶子铺到的范围多 1 ⇒ 只应抓『nesting』
r_k6 = _mk(p17, 0, 18, 0, _mk(p17, 0, 9, 1), _mk(p17, 9, 17, 1))
must_fail("K6  父区间 stop=18 但叶子只铺到 17 ⇒ 必须抓到『nesting』",
          violations(_tree(r_k6, 17, 9, 1, 2), p17, 9), "nesting",
          other="(叶子铺满 [0,17)、尺寸 [9,8] 合规 ⇒ partition/size 都不该红)")

# K7:导出字段腐烂 ⇒ 只应抓『field』
g_ok7 = build_cluster_tree(circle_geometry(1.0, 64), 8)
g_ok7.leaf_size = 3
must_fail("K7  tree.leaf_size 被改成 3 ⇒ 必须抓到『field』",
          violations(g_ok7, p64, 8), "field")

# K8:畸形树(只有左子)⇒ 必须干净地判『malformed』,不许崩
g_mal = _mk(p17, 0, 17, 0, _mk(p17, 0, 9, 1), None)
must_fail("K8  节点只有左子、没有右子 ⇒ 必须抓到『malformed』(不许抛 AttributeError)",
          violations(_tree(g_mal, 17, 8, 1, 2), p17, 8), "malformed")

# K9:尺寸模式被改成"后 r 块各多 1" ⇒ 只应抓『size』(钉住 docstring 的切法约定)
r_k9 = _mk(p17, 0, 17, 0,
           _mk(p17, 0, 11, 1, _mk(p17, 0, 5, 2), _mk(p17, 5, 11, 2)),
           _mk(p17, 11, 17, 1))
must_fail("K9  把叶子切法改成 [5,6,6](docstring 约定是 [6,6,5])⇒ 必须抓到『size』",
          violations(_tree(r_k9, 17, 8, 2, 3), p17, 8), "size",
          other="(尺寸仍然 ≤8、相差 1 ⇒ 只有『模式』这一条该红)")

# ============================================================================
print()
print("=" * 84)
if failures:
    print(f"结果: {len(failures)} 项未通过 -> {failures}")
    raise SystemExit(1)
print(f"结果: 全部通过({COUNTS['disc']} 条有鉴别力 + {COUNTS['base']} 条基线,共 "
      f"{COUNTS['disc'] + COUNTS['base']} 项;D24 口径)")
