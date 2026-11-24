"""自适应交叉近似(ACA)—— 单个矩阵块的低秩分解(路线图 3.2)。

============================================================================
它解决什么问题
============================================================================
H 矩阵把阻抗矩阵 Z 分块:近场块稠密存,可容许块(两簇远离且各自紧凑)低秩近似。
"低秩"这三个字必须有可执行的算法来兑现 —— 本模块就是那个算法:给定一个块
``A``(m×n),求秩 r ≪ min(m,n) 的 ``U V`` 使 ``||A − U V||_F ≤ tol·||A||_F``。

与 SVD 的分工(**D3**):SVD 给"数值秩"的**唯一定义**(NN 的训练标签用它);
ACA 给"装配时真正跑得动"的分解(它只需要矩阵元素,不需要整个矩阵的谱)。
ACA 收敛秩依赖主元选取 ⇒ **不能当标签**,但必须自己能被独立验证。

============================================================================
算法:部分主元 ACA(ACA with partial pivoting)
============================================================================
记 ``R_k = A − Σ_{l≤k} u_l v_l``(第 k 步之后的残差)。第 k 步:

1. **选行**:第 1 步取 ``i_0 = start_row``(默认 0);以后取
   ``i_k = argmax_i |u_k(i)|`` —— 上一个残差列里最大的那个位置。
   这是"部分主元"的核心:它让下一步在**残差还很大**的行上取主元,而不是被
   固定行序牵着走(全主元 ACA 还要扫描整块,代价 O(mn·r),不值得)。
2. **选列**:``j_k = argmax_j |R_k(i_k, j)|``;主元 ``δ_k = R_k(i_k, j_k)``。
3. **一层外积**:``u_k = R_k(:, j_k)/δ_k``,``v_k = R_k(i_k, :)``,
   ``R_{k+1} = R_k − u_k v_k``。
4. 停止:``||R_{k+1}||_F ≤ tol·||A||_F`` 或达到 ``max_rank``。

构造性不变量(**测试逐条钉住**):

    u_k(i_k) = 1                       (列除以主元;回退路径同样成立)
    v_k(j_k) = δ_k                     (行就是残差行)
    k 步之后 rank(U V) ≤ k             ⇒ 数值秩-5 的矩阵至少要 5 步才能精确
    residual[0] = 1                    (非零矩阵的约定;零矩阵为 0)

⚠️ **``||R_k||_F`` 不保证单调不增**——这一点曾是本模块写错的断言。反例(2×2,
见 ``tests/check_aca.py`` 的 counterexample 段):

    A = [[0.01, 0.01], [1, -1]]  ⇒  residual = [1.0, 1.414143, 0.0]

第一步残差从 1 涨到 1.414,但算法仍然精确。⇒ 单调性只是**随机低秩测试矩阵上的观察**,
不是不变量;``check_aca.py`` 里保留的那条判据已标成观察项,不再当定理。

⚠️ **台账 3.2 原验收「随机秩-5 矩阵上 3 步内残差 < 1e-10」对本项目的固定生成器不可满足**:
k 步 ACA 的近似秩 ≤ k,而数值秩-5、σ₅/σ₁ ≥ 1e-3 的矩阵,任何秩-3 近似都有
Eckart–Young 下界 ``||A − A_3||_F ≥ sqrt(σ₄² + σ₅²)``。对本项目的高斯因子矩阵
(60×60 complex,seeds 0–199)实测 3 步残差 **0.566–0.990**(中位 0.713),EY3 下界
0.418–0.549,故 3 步到 1e-10 不可能。**注意限定词**:若只按"代数秩 5"而允许
σ₄=σ₅=1e-12,则数值秩其实是 3,3 步可以到 1e-12 —— 所以判据必须同时钉住数值秩与 σ₅/σ₁。
已按 D29 修订为「固定生成器 + 5 步 ≤ 1e-12 + 4 步 ≥ 0.5·EY4 + 3 步 ≥ EY3」,
见 DECISIONS D32 与台账 ``revisedNote``。

⚠️ **盲点/退化**:残差行可能整行为数值零而残差矩阵不为零(构造性反例见测试 K3:
第一行全零的矩阵)。此时按 ``|u|`` 降序**回退**到下一个候选行,并计入
``n_fallbacks``(按"回退事件"计数,不是跳过的行数);若所有候选行都为零,
判定"数值秩耗尽"并停止(不硬跑,``converged=False``)。回退顺序由 ``pivot_rows``
的精确值钉住(三个 fixture),换顺序会红。

============================================================================
与 3.3 / E3 的关系(边界)
============================================================================
本模块只接受**已经在内存里的稠密块**(``np.ndarray``):残差 ``||R||_F`` 是精确算的,
不做估计。理由:当前管线本来就先装配稠密 Z(见 2.1/2.2 的成本),E3/E4 是**从稠密 Z
切片**测块秩;真正省内存的"按需取元素"装配属于 3.3 的存储统计,等它需要时再加
oracle 适配层。现在不写没有测试覆盖的估计分支。

============================================================================
API
============================================================================
    fac = aca_lowrank(A, tol=1e-10)          # -> ACAFactor
    fac.rank, fac.u, fac.v                   # A ≈ fac.u @ fac.v
    fac.pivot_rows, fac.pivot_cols           # 主元位置(可复现)
    fac.residual                             # 每步之后的相对 Frobenius 残差(k+1 个)
    fac.converged                            # 是否在 max_rank 之前达到 tol
    fac.reconstruct()                        # u @ v
    fac.rel_error(A)                         # ||A − u v||_F / ||A||_F
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["ACAFactor", "aca_lowrank"]


@dataclass
class ACAFactor:
    """ACA 的结果。``u @ v`` 是低秩近似;``residual[k]`` 是 k 步后的相对残差。"""

    u: np.ndarray
    v: np.ndarray
    pivot_rows: np.ndarray
    pivot_cols: np.ndarray
    residual: np.ndarray
    converged: bool
    tol: float
    norm_a: float
    n_fallbacks: int = 0

    @property
    def rank(self) -> int:
        return int(self.u.shape[1])

    def reconstruct(self) -> np.ndarray:
        return self.u @ self.v

    def rel_error(self, a: np.ndarray) -> float:
        a = np.asarray(a)
        den = float(np.linalg.norm(a))
        if den == 0.0:
            # 零矩阵:近似也是零 ⇒ 相对误差按约定取 0(不返回 0/0 的 nan)
            return 0.0
        return float(np.linalg.norm(a - self.reconstruct()) / den)


def _validate(a, tol: float, max_rank) -> tuple[np.ndarray, int]:
    a = np.asarray(a)
    if a.ndim != 2:
        raise ValueError(f"A 必须是二维矩阵,收到 ndim={a.ndim}")
    if a.size and not np.issubdtype(a.dtype, np.number):
        raise ValueError(f"A 必须是数值矩阵,收到 dtype={a.dtype}")
    if a.size and not np.all(np.isfinite(a)):
        raise ValueError("A 含 NaN/inf —— 残差与主元都会失去意义,拒绝计算")
    if not np.isfinite(tol) or tol < 0:
        raise ValueError(f"tol 必须是有限非负数,收到 {tol!r}")
    if max_rank is not None and int(max_rank) < 0:
        raise ValueError(f"max_rank 必须非负,收到 {max_rank!r}")
    return a, (min(a.shape) if max_rank is None else min(int(max_rank), *a.shape))


def aca_lowrank(a, tol: float = 1e-10, max_rank: int | None = None, *,
                start_row: int = 0) -> ACAFactor:
    """对稠密块 ``a`` 做部分主元 ACA,返回 ``ACAFactor``。

    参数
    ----
    a : (m, n) array_like
        稠密块。整数/实数/复数都接受;输出 dtype 至少 float64(复数则 complex128)。
    tol : float
        相对 Frobenius 残差的停止阈值。``tol=0`` 只在数值秩耗尽时停。
    max_rank : int | None
        秩上限(默认 ``min(m, n)``)。达到上限仍未收敛时 ``converged=False``。
    start_row : int
        第一步的候选行,必须满足 ``0 ≤ start_row < m``(越界抛 ValueError,不静默取模)。
        行的选择只影响路径,不影响"能否收敛"。
    """
    a, rmax = _validate(a, tol, max_rank)
    m, n = a.shape
    if m and not (0 <= int(start_row) < m):
        raise ValueError(f"start_row 必须在 [0, {m}) 内,收到 {start_row!r}")
    dtype = np.result_type(a.dtype, np.float64)
    a = np.ascontiguousarray(a, dtype=dtype)
    norm_a = float(np.linalg.norm(a))

    if norm_a == 0.0 or rmax == 0 or m == 0 or n == 0:
        # 零矩阵:相对残差按约定记 0(近似与 A 完全一样)。空矩阵同理。
        resid = np.array([0.0 if norm_a == 0.0 else 1.0])
        return ACAFactor(
            u=np.zeros((m, 0), dtype=dtype), v=np.zeros((0, n), dtype=dtype),
            pivot_rows=np.zeros(0, dtype=int), pivot_cols=np.zeros(0, dtype=int),
            residual=resid, converged=(norm_a == 0.0),
            tol=float(tol), norm_a=norm_a)

    # 数值零的判据:相对于 ||A||_F 的机器精度量级。低于它的主元视为"行已消完"。
    tiny = np.finfo(np.float64).eps * norm_a * 100.0

    u = np.zeros((m, 0), dtype=dtype)
    v = np.zeros((0, n), dtype=dtype)
    residual = [1.0]
    pivot_rows: list[int] = []
    pivot_cols: list[int] = []
    n_fallbacks = 0
    converged = False
    i = int(start_row) if m else 0

    for _k in range(rmax):
        # ---- 选主元行:优先沿用调用方/上一步给的行;整行数值为零则回退 ----
        row, j, delta, i_used, fb = _pick_pivot(a, u, v, i, tiny)
        if delta is None:      # 所有候选行的残差都数值为零 ⇒ 数值秩耗尽
            break
        n_fallbacks += fb
        col = a[:, j] - u @ v[:, j]

        u = np.concatenate([u, (col / delta).reshape(m, 1)], axis=1)
        v = np.concatenate([v, row.reshape(1, n)], axis=0)
        residual.append(float(np.linalg.norm(a - u @ v)) / norm_a)
        pivot_rows.append(i_used)
        pivot_cols.append(j)

        if residual[-1] <= tol:
            converged = True
            break
        # 下一步的候选行 = 本步残差列里绝对值最大的那一行(部分主元)
        i = int(np.argmax(np.abs(u[:, -1])))

    return ACAFactor(
        u=u, v=v,
        pivot_rows=np.asarray(pivot_rows, dtype=int),
        pivot_cols=np.asarray(pivot_cols, dtype=int),
        residual=np.asarray(residual, dtype=float),
        converged=converged, tol=float(tol), norm_a=norm_a,
        n_fallbacks=n_fallbacks)


def _pick_pivot(a: np.ndarray, u: np.ndarray, v: np.ndarray, i0: int,
                tiny: float) -> tuple[np.ndarray, int, complex | float | None, int, int]:
    """在残差矩阵上取主元:先用行 ``i0``;整行数值为零则按 **|u| 降序**回退。

    回退顺序不是随手选的:``|u|`` 是"上一步残差列",它大的行说明残差还大,
    优先在那儿取主元是部分主元的自然延续。这条规则**必须被精确钉住**
    ——2026-11-24 复核实测:把顺序换成行号序,旧版 59 项自检仍全绿。
    现在 ``check_aca.py`` 对 60×60 / 对角 / 首行全零三个 fixture 断言
    ``pivot_rows`` 的**精确值**,换顺序就会红。

    返回 ``(残差行, 列号, 主元值, 实际用的行号, 回退事件数)``。
    找不到非零主元时 ``主元值 = None``。
    """
    m, n = a.shape
    row = a[i0, :] - (u[i0, :] @ v if u.shape[1] else 0.0)
    j = int(np.argmax(np.abs(row)))
    delta = row[j]
    if abs(delta) > tiny:
        return np.asarray(row), j, delta, i0, 0

    # 回退:按 |u| 降序（没有 u 时按行号序）。i0 那一行已确认数值为零,
    # 跳过它是省一次重算;即使不跳,重算结果也是零、不会中标。
    order = np.argsort(-np.abs(u[:, -1]), kind="stable") if u.shape[1] else np.arange(m)
    for i in order:
        i = int(i)
        if i == i0:
            continue
        row = a[i, :] - (u[i, :] @ v if u.shape[1] else 0.0)
        j = int(np.argmax(np.abs(row)))
        delta = row[j]
        if abs(delta) > tiny:
            return np.asarray(row), j, delta, i, 1
    return np.zeros(n, dtype=a.dtype), 0, None, i0, 0
