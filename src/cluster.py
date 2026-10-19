"""二叉树聚类 —— H 矩阵分块的段索引结构(路线图 3.1)。

============================================================================
它解决什么问题
============================================================================
ACA / H 矩阵把阻抗矩阵 Z 划分成两类块:
    · **近场块**(两个簇几何上相邻)  —— 稠密存,照常用求解器;
    · **可容许块**(两个簇远离且各自紧凑)—— 用 ACA 低秩近似,存储 O(r) 而不是 O(m·n)。
划分质量决定两件事,而且**都直接进论文的测量口径**:

1. **低秩性**:可容许块的两个簇必须**几何上彼此远离、各自紧凑**。若一个"簇"其实是
   绕圆半圈的一串段,它的"直径"没有意义,可容许判据(见 3.3 的
   `dist(簇A, 簇B) >= η·max(diam A, diam B)`)会被打穿,块秩也降不下来。
   ⇒ 所以本模块把"簇 = **几何顺序上连续的一段**"做成硬约束。

2. **块秩的统计**:E3/E5 要测"块秩 vs 块大小 / vs ka"。块秩是**块大小**的函数 ——
   若叶子大小从 4 到 8 不等,量出来的秩分布里就混进了"叶子怎么切的"这个假信号。
   ⇒ 所以本模块把"叶子大小相差 ≤ 1"做成**构造性保证**,而不是靠测试碰运气。

============================================================================
本模块的组织方式(与教科书做法不同,理由在下面)
============================================================================
记 `k = ⌈N / leaf_size⌉`(叶子数),`q, r = divmod(N, k)`。

    叶子  = 把 [0, N) 切成 k 个连续块,前 r 块各 q+1 段、其余 q 段;
    树    = 对这 k 个**块**做近完全二叉树(每次按叶子数对半,左边取 ⌈·/2⌉)。

由此**构造性**得到(不是靠测试碰运气):

1. **分割性**:叶子两两不交、并集覆盖全部 N 个段各一次;
2. **连续性**:每个节点是几何顺序上的一段连续区间,父 = 两子首尾相接;
3. **均匀性**:叶子大小 ∈ {q, q+1} ≤ `leaf_size`(相差 ≤ 1),且**前 r 块各多 1 段**
   (`q, r = divmod(N, k)`;这条切法被测试逐位钉住);
4. **深度**:树深 `= ⌈log2 k⌉`(`k = ⌈N/leaf_size⌉`)⇒ `≈ log2(N/leaf_size)`;
   叶子所在层 ∈ {深, 深−1}(k 不是 2 的幂时两层都有)。
   ⚠️ 字面判据 `≈ log2(N/leaf)` 只在 **1 ≤ leaf ≤ N** 时有意义;`leaf > N` 是退化输入
   (只有 1 个叶子,深度 0),此时 log2(N/leaf) < 0,不要拿它当判据。

⚠️ **术语**:这棵树是**高度平衡(叶子层 d 或 d−1)**,**不是**"完全二叉树"
(k=6 时叶子层从左到右是 [3,3,2,3,3,2],末层并不是从左填满)。论文里写"完全二叉树"
会被审稿人抓。

⚠️ **为什么不用教科书常见的"对半切到 ≤ leaf_size 为止"**
那个规则(`n <= leaf_size` 停,否则切 `⌊n/2⌋`)在 N=17、leaf=8 时给出叶子 {8, 4, 5}:
最大/最小 = **2**。它的深度确实还是 ⌈log2(N/leaf)⌉,但"叶子均匀"不成立 ——
而这条在 E3/E5 里会变成"块秩 vs 块大小"的系统性混淆。
本模块的做法在同样的 N=17、leaf=8 下给 k=3 个叶子 {6, 6, 5}:最大/最小 = 1.2,
严格满足"相差 ≤ 1"。

⚠️ **为什么不是"固定 2^d 个叶子"**(那是本模块的第一版设计,已被证伪)
若强行令叶子数 = 2^d(d = ⌈log2(N/leaf)⌉),在 `leaf=1`、N=257 时 2^d = 512 > N,
于是出现 **255 个大小为 0 的空叶子**(第一版在 `radius = max(...)` 上直接崩:
`zero-size array to reduction`)。"先定叶子数 k、再建树"没有这个问题 ——
k ≤ N 恒成立,不存在空叶子。

============================================================================
几何顺序(`order`)
============================================================================
输入的各段中点不保证是沿边界排好的(段号 = 调用方的枚举顺序)。本模块内部
**按绕点集重心的极角**排序得到 `order`,并把它一起返回:
    `tree.order[i]` = 几何顺序第 i 位对应的**原始段号**。
下游装配 `Z[np.ix_(idx_a, idx_b)]` 必须用同一份 `order`,否则簇与矩阵下标对不上。
圆几何下 `mom.circle_geometry` 本来就按 φ 递增枚举,故 `order` 是恒等置换的**一次旋转**
(切口在 φ≈±π)—— **不要因此把它省掉**:换几何(或换枚举顺序)时省掉它就是错的,
而且不会报错。

⚠️ **默认极角排序只对"重心落在多边形核(kernel)内"的边界成立**。凸边界与圆边界
自动满足;细长的 L 形是星形但重心可能落在域外,那时相邻的 order 位置在物理上并不
相邻,必须由调用方通过 `order=` 传入自己的边界顺序。"非星形"不是充分条件。

============================================================================
API
============================================================================
    tree = build_cluster_tree(geom, leaf_size=16)      # 或直接传 (N,2) 中点坐标
    tree.root, tree.order, tree.n, tree.leaf_size, tree.depth, tree.n_leaves
    tree.leaves()            # 左→右(几何顺序)的叶子列表
    tree.nodes()             # 先序遍历的全部节点
    tree.indices(cluster)    # 该簇的**原始段号**数组(按几何顺序)
    tree.leaf_indices()      # 全部叶子的原始段号
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "Cluster",
    "ClusterTree",
    "build_cluster_tree",
    "order_by_angle",
]


@dataclass
class Cluster:
    """一个簇:几何顺序上连续的半开区间 ``[start, stop)``。

    ⚠️ `start/stop` 是 **`ClusterTree.order` 里的位置**,不是原始段号;
    原始段号用 ``ClusterTree.indices(cluster)`` 取。这样区分是为了让
    "树的结构"与"矩阵的行列编号"解耦(换几何时只换 order)。

    `center` / `radius` 是本簇成员中点的**算术平均**与**以该均值为心的包围半径**
    (`max |mid - center|`)。⚠️ 它**不是最小包围圆(MEC)的半径** —— 实测:半圆簇
    (N=640 的 root.left,320 段)均值心半径 1.1828 vs 真 MEC 0.99998(**高估 18.3%**);
    共线点 {0,1,10} 高估 **26.7%**。
    对 3.3 的影响是**保守**的:可容许判据 `dist(τ,σ) ≥ η·max(diam τ, diam σ)` 里
    `diam = 2·radius` 会**高估**块的尺寸 ⇒ 可容许块只会变少、不会变多 ⇒ 结果是安全的,
    但压缩率会被系统性压低。3.3 若要用紧的 MEC,必须自己实现,不能沿用这里的语义。
    """
    start: int
    stop: int
    level: int
    center: np.ndarray
    radius: float
    left: "Cluster | None" = None
    right: "Cluster | None" = None

    @property
    def size(self) -> int:
        """本簇包含的段数。"""
        return self.stop - self.start

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


@dataclass
class ClusterTree:
    """高度平衡二叉树(叶子层 = depth 或 depth−1)。

    叶子大小 ∈ {⌊N/k⌋, ⌈N/k⌉},树深 = ⌈log2 k⌉,k = ⌈N/leaf_size⌉。
    ⚠️ 它不是"完全二叉树":k=6 时叶子层从左到右是 [3,3,2,3,3,2],末层不是从左填满。
    """
    root: Cluster
    order: np.ndarray      # (N,) 几何顺序 -> 原始段号
    n: int
    leaf_size: int
    depth: int
    n_leaves: int

    def leaves(self) -> list[Cluster]:
        """全部叶子,按几何顺序从左到右。"""
        out: list[Cluster] = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node.is_leaf:
                out.append(node)
            else:
                stack.append(node.right)
                stack.append(node.left)
        return out

    def nodes(self) -> list[Cluster]:
        """全部节点,先序(父在子前)。"""
        out: list[Cluster] = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            out.append(node)
            if not node.is_leaf:
                stack.append(node.right)
                stack.append(node.left)
        return out

    def indices(self, cluster: Cluster) -> np.ndarray:
        """该簇成员对应的**原始段号**(按几何顺序排列,**不是**升序,也**不是**区间)。

        ⚠️ 跨极角切口(φ≈±π)的叶子会在这里**回绕**,例如返回
        ``[995, 996, 997, 998, 999, 0, 1, 2, ...]``。所以:
        · 不能把它当 `arange(start, stop)` 或等差数列;
        · 不能拿它当"近场的段号区间" —— 3.3 判近场必须按**几何距离**,不能按段号差。
        测试里有一条 (640, 7) 的判据专门断言这种回绕真的会出现。
        """
        return self.order[cluster.start:cluster.stop]

    def leaf_indices(self) -> list[np.ndarray]:
        return [self.indices(c) for c in self.leaves()]


# ============================================================================
# 校验
# ============================================================================

def _as_points(points) -> np.ndarray:
    """把 geom dict 或 (N,2) 数组统一成 float64 (N,2),并校验。

    ⚠️ 复数/不可转的数必须抛 **ValueError** 而不是让 numpy 抛 TypeError:
    与 `mom.py` / `rcs.py` 的输入校验口径保持一致(判据只看"坏输入是否被拒",
    但 TypeError 会让调用方的 `except ValueError` 漏掉它)。
    """
    if isinstance(points, dict):
        if "mid" not in points:
            raise ValueError("geom 字典缺少 'mid' 键(段中点坐标 (N,2))")
        points = points["mid"]
    # ⚠️ 复数必须显式拒绝:本机 numpy 2.5 的 `np.asarray(complex, dtype=float)`
    # 只发一个 ComplexWarning 然后**静默丢掉虚部**(实测,与 rcs._as_real_array 同一教训),
    # 是最坏的一类输入错误 —— 坐标被悄悄改成别的点,建出来的簇看着一切正常。
    if np.iscomplexobj(points):
        raise ValueError(f"中点坐标不能是复数(会被静默丢虚部),收到 dtype={np.asarray(points).dtype}")
    try:
        pts = np.asarray(points, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"中点坐标必须能转成 float64: {exc}") from exc
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"中点坐标必须是 (N,2),收到形状 {pts.shape}")
    if pts.shape[0] < 1:
        raise ValueError("至少需要 1 个段")
    if not np.all(np.isfinite(pts)):
        raise ValueError("中点坐标含 NaN/inf —— 静默传下去会让半径变成 nan,提前拒绝")
    return pts


def _as_leaf_size(leaf_size) -> int:
    if isinstance(leaf_size, bool) or not isinstance(leaf_size, (int, np.integer)):
        raise ValueError(f"leaf_size 必须是整数,收到 {leaf_size!r}")
    if leaf_size < 1:
        raise ValueError(f"leaf_size 必须 >= 1,收到 {leaf_size!r}")
    return int(leaf_size)


# ============================================================================
# 几何顺序
# ============================================================================

def order_by_angle(points) -> np.ndarray:
    """绕点集重心的极角顺序(逆时针,切口在 ±π)。

    对圆边界这就是沿边界的一圈;切口对环形边界无害(任何切口等价)。
    非星形几何不要用这个函数,直接给 `build_cluster_tree(..., order=...)`。
    """
    pts = _as_points(points)
    center = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    return np.argsort(ang, kind="stable")


# ============================================================================
# 建树
# ============================================================================

def _depth_for(n_leaves: int) -> int:
    """最小的 d 使 `2^d >= n_leaves`(即 ⌈log2(n_leaves)⌉;n_leaves=1 时为 0)。

    用整数位长而不是 `math.ceil(math.log2(...))`:后者在比值恰为 2 的幂时
    依赖浮点表示,而这里错一层会让整棵树的叶子大小翻倍。
    """
    return 0 if n_leaves <= 1 else (n_leaves - 1).bit_length()


def build_cluster_tree(points, leaf_size: int, *, order=None) -> ClusterTree:
    """把 N 个段组织成高度平衡二叉树(叶子层 depth 或 depth−1)。

    Parameters
    ----------
    points : dict 或 (N,2) 数组
        `mom.circle_geometry` 返回的 geom 字典(取 `"mid"`),或直接给段中点坐标。
    leaf_size : int
        叶子大小上限(>= 1)。实际叶子大小 = ⌊N/k⌋ 或 ⌈N/k⌉,其中 k = ⌈N/leaf_size⌉。
    order : (N,) 整数数组,可选
        **几何顺序**:`order[i]` 是几何顺序第 i 位的原始段号,必须是 0..N-1 的置换。
        默认按绕重心的极角排序(`order_by_angle`)。**重心不在边界核内的几何**
        (细长 L 形等)必须自己给。

    Returns
    -------
    ClusterTree

    复杂度:排序 O(N log N) 主导;建树本身是 O(N·depth) = O(N log k) ——
    每个节点都要对自己的成员点求 **mean 与 max|·|**,逐层累计恰好扫过全部点
    (Σ size = N·(depth+1)),**不是 O(k)**。实测 N=2×10⁵、leaf=16:排序 0.004 s,
    建树 0.216 s(排序只占 2%);N=10⁶/leaf=16 → 1.19 s / +70 MB,
    N=10⁶/leaf=1 → 15.0 s / +800 MB(叶子越多,节点数越多,常数越大)。
    """
    pts = _as_points(points)
    n = pts.shape[0]
    leaf_size = _as_leaf_size(leaf_size)

    if order is None:
        order = order_by_angle(pts)
    else:
        order = np.asarray(order)
        if order.shape != (n,):
            raise ValueError(f"order 长度必须是 {n},收到 {order.shape}")
        if not np.issubdtype(order.dtype, np.integer):
            raise ValueError(f"order 必须是整数数组,收到 dtype={order.dtype}")
        if not np.array_equal(np.sort(order), np.arange(n)):
            raise ValueError("order 必须是 0..N-1 的一个置换(每个段恰好出现一次)")

    sorted_pts = pts[order]

    # ---- 先定 k 个**大小只差 ≤1** 的连续叶子块(前 r 块各多 1 段) ----
    n_leaves = -(-n // leaf_size)                 # ceil(n / leaf_size)
    q, r = divmod(n, n_leaves)
    bounds = np.empty(n_leaves + 1, dtype=int)
    bounds[0] = 0
    for i in range(n_leaves):
        bounds[i + 1] = bounds[i] + q + (1 if i < r else 0)

    def _node(i0: int, i1: int, level: int) -> Cluster:
        start, stop = int(bounds[i0]), int(bounds[i1])
        sub = sorted_pts[start:stop]
        center = sub.mean(axis=0)
        radius = float(np.max(np.linalg.norm(sub - center, axis=1)))
        if i1 - i0 == 1:
            return Cluster(start, stop, level, center, radius)
        # 叶子数对半:左边取 ⌈(i1-i0)/2⌉,保证两侧都非空且深度 = ⌈log2 k⌉
        imid = i0 + (i1 - i0 + 1) // 2
        return Cluster(start, stop, level, center, radius,
                       _node(i0, imid, level + 1),
                       _node(imid, i1, level + 1))

    root = _node(0, n_leaves, 0)
    return ClusterTree(root=root, order=order, n=n, leaf_size=leaf_size,
                       depth=_depth_for(n_leaves), n_leaves=n_leaves)
