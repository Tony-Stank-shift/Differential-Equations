"""E2 —— 稠密矩量法的成本 vs 未知量数 N(路线图 2.1 时间 / 2.2 内存)。

被测管线(每一步都是项目里已在用的函数,不另写数值代码)::

    geom  = circle_geometry(radius, N)
    Z     = assemble_impedance(k, geom)          # 预期 O(N^2 * Q)
    V     = assemble_excitation(k, geom)
    J, _  = solve_dense(Z, V)                    # LAPACK,预期 O(N^3)

⚠️ 这台机器上时间测量有 ~2.5× 的随机散布 —— 先读这一节再用数字
----------------------------------------------------------------------
最初的版本把 100…6400 放在同一条进程里顺序测,发现数字比单点跑慢 2.0~2.6×,
于是得出"**顺序测量被进程内历史污染(堆碎片/未归还内存)**"的结论,并写进了
docstring。**这个结论是错的** —— 按 D16(采纳机制断言前先复现它的方法)做了
交替对照实验,同一台机器、同一个 N=800、同一份代码:

    轮次   A:新进程单点     B:顺序扫描的第 4 个点
     1        7.641 s            3.270 s
     2        6.404 s            6.001 s
     3        3.946 s            3.056 s

两种方式**都**从 3.06 s 散布到 7.64 s —— 差异不来自进程内顺序,而来自
**机器负载**(本机 24 逻辑核、物理内存占用 57~58%、还有别的进程在跑),
同一配置的重复测量本身就有 ~2.5× 的散布。第 7 轮复核的锚点 290 ns/(N²Q)
给定 N=800/Q=16 应为 2.97 s —— 上表最好的那次 3.056 s 与它吻合 3%。

⇒ 正确做法不是"一点一进程"(那解决不了负载),而是**每点重复多次,
报 min / median / max**,并用 **min** 估计真实的计算成本(负载只会让测量偏慢,
不会偏快,所以 min 是成本的上界估计中最干净的一个)。本模块据此实现,
并把每次重复都落盘,便于第三方核对散布。

⚠️ 第二条陷阱:**不要把"求解"塞在高 N 的装配后面一起拟合**。装配是 O(N²),
求解是 O(N³);在 N≤3200 时装配主导(实测 233 s vs 0.84 s),求解那几点的规律
被固定开销压住 —— 把 (1600, 3200, 6400) 的 t_solve 直接拟合只得到 N^1.62。
所以本模块把求解**单独剥离**成一份实验(``SOLVE_REPEATS``:不装配,直接对随机稠密
复矩阵计时),让 N³ 项自己站出来。

两个部分
--------
- ``--part time``   (2.1)装配 + 求解的时间 vs N,以及**时间墙**。
- ``--part solve``  (2.1 对照)只跑纯求解扫描。**存在的理由是分离两种解释**:
  理想 flop 数严格 ∝ N³,但多线程下实测有效指数只有 1.8–2.5。差别来自"每 flop 的
  实际吞吐随 N 上升"(实测 50 → 518 GFLOP/s),而这几乎全是**并行效率**(小矩阵
  摊不开 24 线程)。把 ``OPENBLAS_NUM_THREADS=1`` 钉死并行效率后再看指数,
  是判定"偏差是并行开销还是算法阶"的对照实验。线程数由环境变量控制,如实记进
  ``meta.env.thread_env``;产物是 ``exp02b_solve_scaling_{single,multi}.json``。
- ``--part memory`` (2.2)峰值内存 vs N,以及**内存墙**。
- ``--part all``   以上全部(默认,供 ``run_all.py``)。

零散运行
--------
- ``--point N``  在**当前**进程里只做 N 这一个点(供编排器调用)
- ``--solve N``  在**当前**进程里只做"随机复矩阵求解"这一个点
- ``--fill-time N [reps]``  给**已有**的 ``exp02a`` 产物补一个装配点,把新点并进去后
  用同一份 ``_finish_time`` 重算拟合/时间墙/图;**不重跑**任何已有点
  (时间测量受负载影响,重跑会把整张表换成另一组噪声;见 2026-11-24 的 D33)。

输出
----
- ``results/exp02a_dense_time.json``   / ``figures/exp02a_dense_time.png``
- ``results/exp02b_solve_scaling_*.json`` / ``figures/exp02b_solve_scaling_*.png``
- ``results/exp02c_dense_memory.json`` / ``figures/exp02c_dense_memory.png``
- ``results/exp02_points.jsonl``       每次重复的原始记录(逐次可核对散布)
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

# 既支持 `python experiments/exp02_dense_scaling.py` 直接跑,也支持被 run_all.py 导入
# (后者已经加过 src,重复 insert 无害)。2026-10-13:漏这一行时直接跑会
# ModuleNotFoundError: No module named 'mom'。
sys.path.insert(0, str(ROOT / "src"))

#: 主算例:ka = 5(k=1, radius=5),与 1.3 / 1.4 自检的主算例一致
K_WAVE = 1.0
RADIUS = 5.0

#: 2.1 装配+求解的扫描点 -> 重复次数(每点一条进程;机器负载导致 ~2.5× 散布,
#: 所以大 N 少测几次、小 N 多测几次。预估合计 ~600 s。)
TIME_REPEATS = {400: 3, 800: 3, 1600: 3, 3200: 2, 6400: 2, 8000: 2}
#: ⚠️ 8000 是 2026-11-24 补进来的(台账范围到 8000,主运行当时只到 6400):
#: 实测单次 523–560 s(装配 + 求解),2 次 ≈ 18 min。
#: 补点时**不要重跑**已有的 400…6400 —— 用 ``--fill-time 8000``,它复用 _finish_time。
#: 2.1 纯求解的扫描点 -> 重复次数(不需装配;预估合计 ~120 s)
#: ⚠️ 必须走到 N≈12000 才看得到 N³:2000…10000 段 BLAS 效率还在爬(50→244 GFLOP/s),
#: 直接拟合只得到 N^2.05(2026-10-13 实测)。12000 处 32 B/元 × N² ≈ 4.6 GiB,
#: 所以下面的可用内存守卫放到 0.9 —— 这是本机在 15 GiB 总量 / 当时 ~6 GiB 可用下
#: 能触到的最大规模,再往上就 OOM 了。
SOLVE_REPEATS = {2000: 3, 4000: 3, 8000: 2, 10000: 2, 12000: 2}
#: 2.2 峰值内存的扫描点 -> 重复次数(峰值内存不受负载影响,1 次即可)
#: 基础四点覆盖 64× 的 N² 跨度,足够把 `peak = e + f·N²` 的自由度(2 个)定住 5 倍。
MEM_REPEATS = {800: 1, 1600: 1, 3200: 1, 6400: 1}
#: 2.2 的**顶部锚点候选阶梯**(按所需内存从大到小)。为什么要阶梯而不是硬编码一个点:
#: 本机可用内存随负载在 5–7 GiB 之间漂,而守卫(`need+1.0 ≤ 0.85·free`)只在临界处起作用 ——
#: 硬编码 11400 时,一旦运行那一刻可用内存掉到 5.2 GiB,整次运行就会**没有大 N 锚点**,
#: 只剩 6400 外推 1.8× 到 4 GB,说服力不足。
#: 预注册规则:按顺序取**第一个**通过守卫的点;一个都过不了就如实记 skipped,**不降阈值**。
#: ⚠️ 2026-10-19 补:阶梯顶必须 **≥ 11425**(4 GiB 的拟合过零点)。
#: 第一版顶是 10400(峰值 3.33 GiB,离 4 GB 差 11%),只能外推;
#: 第二版顶 11400 也只到 3.981 GiB —— **离过零点还差 25**(差 0.2%),仍然只能外推。
#: 复盘(第 3 轮子代理复核证伪了"等机器空闲"的说法):当时可用 6.20 GiB,
#: 真实守卫按 `need+1 ≤ 0.85·free` 放到 N≈11,690,**本来就够跑**;
#: 没拿到实测点纯粹是**阶梯顶设短了**,不是内存不够。
#: 现在顶取 11600(预测 4.12 GiB,留 3% 余量),其次 11500(预测 4.05 GiB,余量 1.3%)。
MEM_TOP_LADDER = (11600, 11500, 11400, 11200, 11050, 10800, 10400, 10000, 9600, 8800, 8000)


def _med(vals: list[float]) -> float:
    return float(np.median(vals))


def _aggregate(runs: list[dict], key: str) -> dict:
    """把同一 N 的多次重复汇总成 min / median / max。"""
    v = [r[key] for r in runs]
    return {
        "n": len(v),
        "min": float(min(v)),
        "median": _med(v),
        "max": float(max(v)),
        "spread_ratio": float(max(v) / min(v)) if min(v) > 0 else float("nan"),
        "all": [float(x) for x in v],
    }


# ----------------------------------------------------------------------------
# 进程内存
# ----------------------------------------------------------------------------
class _PMC(ctypes.Structure):
    _fields_ = [
        ("cb", wt.DWORD),
        ("PageFaultCount", wt.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


_k32 = ctypes.WinDLL("kernel32", use_last_error=True)
_k32.GetCurrentProcess.restype = wt.HANDLE
try:  # Win7+: 优先 kernel32 的 K32 版本
    _GetProcessMemoryInfo = _k32.K32GetProcessMemoryInfo
except AttributeError:  # pragma: no cover
    _GetProcessMemoryInfo = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
_GetProcessMemoryInfo.argtypes = [wt.HANDLE, ctypes.POINTER(_PMC), wt.DWORD]
_GetProcessMemoryInfo.restype = wt.BOOL


def _mem() -> tuple[int, int]:
    """返回 (当前工作集字节, 峰值工作集字节)。

    ⚠️ 必须显式设 argtypes/restype:``GetCurrentProcess()`` 返回伪句柄 ``-1``,
    若按 ctypes 默认的 c_int 传递会被截成 32 位 ``0xFFFFFFFF`` ⇒ 句柄无效,
    ``GetProcessMemoryInfo`` 直接失败(2026-10-13 实测踩过)。
    """
    c = _PMC()
    c.cb = ctypes.sizeof(c)
    ok = _GetProcessMemoryInfo(_k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
    if not ok:
        raise OSError(f"GetProcessMemoryInfo 失败 (winerror={ctypes.get_last_error()})")
    return int(c.WorkingSetSize), int(c.PeakWorkingSetSize)


class _MEMSTATUS(ctypes.Structure):
    _fields_ = [
        ("dwLength", wt.DWORD),
        ("dwMemoryLoad", wt.DWORD),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def system_memory() -> dict:
    """物理内存总量 / 此刻可用量(字节)。注意 avail 是**时点快照**,随其它进程波动。"""
    fn = _k32.GlobalMemoryStatusEx
    fn.argtypes = [ctypes.POINTER(_MEMSTATUS)]
    fn.restype = wt.BOOL
    m = _MEMSTATUS()
    m.dwLength = ctypes.sizeof(m)
    if not fn(ctypes.byref(m)):
        raise OSError(f"GlobalMemoryStatusEx 失败 (winerror={ctypes.get_last_error()})")
    return {
        "total_phys_b": int(m.ullTotalPhys),
        "avail_phys_b": int(m.ullAvailPhys),
        "total_pagefile_b": int(m.ullTotalPageFile),
        "avail_pagefile_b": int(m.ullAvailPageFile),
        "memory_load_pct": int(m.dwMemoryLoad),
    }


# ----------------------------------------------------------------------------
# 单点测量(在当前进程里做一件事)
# ----------------------------------------------------------------------------
#: 内存守卫的**绝对**安全余量。⚠️ 2026-10-13 第 13 轮复核证伪了旧的 `need <= 0.95*free`:
#: 它放行了 N=14000(实测峰值 5.931 GiB / 当时可用 6.975 GiB = **85.0%**),只剩 1.04 GiB 余量。
#: 一旦触发交换,计时就彻底作废 —— 交换比算法慢几个数量级,而且测出来的是硬盘不是 CPU。
SAFETY_MARGIN_B = int(1.0 * 2**30)


def memory_admit(n: int) -> dict:
    """判断能不能跑 N=n 的稠密复矩阵求解,并把判断依据**原样返回**(供落盘)。

    估计式 `32 B/元 + 0.2 GiB` 是实测标定的:N=10000/12000/14000 的实测峰值
    3.057 / 4.374 / 5.931 GiB,与估计 3.18 / 4.49 / 6.04 GiB 的偏差 ≤4%(且**偏保守**)。
    注意守卫本身在小 N 处是"装饰"(N=2000 估计 0.32 GiB、实测 0.166 GiB,高估 93%),
    它真正起作用的是 N≥10000 这一段。
    """
    ms = system_memory()
    need = 32 * n**2 + int(0.2 * 2**30)
    free, total = ms["avail_phys_b"], ms["total_phys_b"]
    ok = (need + SAFETY_MARGIN_B <= 0.85 * free) and (need <= 0.5 * total)
    return {"n": int(n), "need_b": int(need), "free_b": int(free), "total_b": int(total),
            "margin_b": SAFETY_MARGIN_B, "memory_load_pct": ms["memory_load_pct"],
            "admit": bool(ok),
            "rule": "need + 1.0 GiB <= 0.85*avail 且 need <= 0.5*total"}


def guard_capacity(avail_b: int, total_b: int) -> float:
    """**真实内存守卫**下,当前可用内存能跑的最大 N(与 `memory_admit` 同一套常数)。

    守卫价格:`need(n) = 32n² + 0.2 GiB`;条件:
        `need + 1.0 GiB ≤ 0.85·avail` 且 `need ≤ 0.5·total`
    ⇒ `N_max = sqrt((min(0.85·avail − 1 GiB, 0.5·total) − 0.2 GiB) / 32)`。

    ⚠️ 不要用「拟合峰值反函数」`e + f·N² = 0.85·avail` 去算这个数:那会把守卫的
    **0.2 GiB 基座**与 **32 B/元** 的价格换成拟合的 32.32 B/元,两者差 ~1%
    (2026-10-19 复核实测:6.2031 GiB 下拟合口径 11,815 vs 真实守卫 11,690)。
    """
    cap = min(0.85 * avail_b - SAFETY_MARGIN_B, 0.5 * total_b)
    v = (cap - int(0.2 * 2**30)) / 32.0
    return float(math.sqrt(v)) if v > 0 else float("nan")


def _report_skip(skip: dict, reasons: list) -> None:
    reasons.append(skip)
    print(f"  {skip['n']:>6} | 跳过:需约 {skip['need_b'] / 2**30:.2f} GiB + 余量 "
          f"{skip['margin_b'] / 2**30:.2f} GiB > 0.85×可用 {skip['free_b'] / 2**30:.2f} GiB"
          f"(占用 {skip['memory_load_pct']}%)")


def _require_resid(rec: dict, tol: float = 1e-8) -> None:
    """唯一能发现"LAPACK 悄悄换了路径/精度"的数值健康指标,必须**有断言**。

    2026-10-13 第 13 轮复核指出:此前 `resid_rel` 只被打印、从不判定,所以任何静默的
    数值退化都不会让任何东西变红。实测范围 2.47e-12 … 8.52e-11(随 N 上升,是并行归约
    累积,不是病态 —— 复核已验 cond₂ 稳定在 1e4–1e5),1e-8 留了 100× 余量。
    """
    r = float(rec["resid_rel"])
    if not (r == r) or r > tol:
        raise RuntimeError(
            f"数值健康检查失败:N={rec['n_seg']} 的相对残差 {r:.3e} > {tol:g} "
            f"—— 说明 LAPACK 路径/精度变了,这份时序不可用")


def measure_one(n_seg: int, quad_order: int | None = None) -> dict:
    """装配 + 求解一个点。

    ``quad_order=None`` 走 ``assemble_impedance`` 的签名默认值(即
    ``quad.DEFAULT_QUAD_ORDER``),不硬编码 16 —— 免得默认值一改这里悄悄失效。
    """
    from mom import (  # noqa: PLC0415
        assemble_excitation,
        assemble_impedance,
        circle_geometry,
        solve_dense,
    )

    ws0, peak0 = _mem()
    t0 = time.perf_counter()
    geom = circle_geometry(RADIUS, n_seg)
    t_geom = time.perf_counter() - t0

    t0 = time.perf_counter()
    if quad_order is None:
        Z = assemble_impedance(K_WAVE, geom)
        q_used = None
    else:
        Z = assemble_impedance(K_WAVE, geom, quad_order=quad_order)
        q_used = quad_order
    t_assemble = time.perf_counter() - t0
    ws1, _peak1 = _mem()

    V = assemble_excitation(K_WAVE, geom)

    t0 = time.perf_counter()
    J, _resid = solve_dense(Z, V)
    t_solve = time.perf_counter() - t0
    ws2, peak2 = _mem()

    return {
        "kind": "assemble+solve",
        "n_seg": int(n_seg),
        "quad_order": q_used,
        "t_geom_s": t_geom,
        "t_assemble_s": t_assemble,
        "t_solve_s": t_solve,
        "t_total_s": t_geom + t_assemble + t_solve,
        "ws_before_b": ws0,
        "ws_after_assemble_b": ws1,
        "ws_after_solve_b": ws2,
        "peak_b": peak2,
        "z_footprint_b": ws1 - ws0,
        "z_bytes_per_entry": (ws1 - ws0) / float(n_seg) ** 2,
        "z00_abs": float(abs(Z[0, 0])),
        "j_max_abs": float(np.max(np.abs(J))),
    }


def solve_only(n: int, seed: int = 20261013) -> dict:
    """只对随机稠密复矩阵做 ``np.linalg.solve``,把 N³ 从装配里剥离出来。

    随机矩阵的**条件数**可能与阻抗矩阵不同,但这只影响 LAPACK 是否走
    迭代精化/分块路径,**不影响渐近 N³ 的阶**;本实验要的是"阶",所以可以接受。
    """
    rng = np.random.default_rng(seed)
    ws0, _ = _mem()
    t0 = time.perf_counter()
    # ⚠️ **就地填充复数数组**。不要写
    #     A = (rng.standard_normal((n,n)) + 1j*rng.standard_normal((n,n))).astype(complex)
    #   —— 那样同时活着的是 **2 份 (n,n) float64 临时量 + 1 份 `1j*` 的复数 + 1 份加法结果
    #   + 1 份 astype 的复制**(第 13 轮复核纠正:原注释只写"两份 float64",少算了复数那几份),
    #   实测 N=12000 峰值被抬到 **5.40 GiB / 48.96 B/元**(新写法 32.6 B/元),
    #   而当时可用只有 5.74 GiB ⇒ 差点 OOM。
    #   改后就地填 view:实测 44.5/35.9/33.2/32.8/32.6 B/元(N=2000…12000),
    #   固定基座 ~0.13 GiB 之外就是理论的 32 B/元。
    #   ⚠️ 两种写法的**分布相同**(复核实测 mean 0/std 0.9997/corr −0.0014/E|z|²=1.9993),
    #   只有 PRNG 消费顺序不同 ⇒ 改前/改后的 t_solve 严格可比(t_solve 只包 np.linalg.solve)。
    #   为了以后不再出现"分不清哪条记录是哪种实现",每条记录都带 `solver_impl`。
    A = np.empty((n, n), dtype=np.complex128)
    rng.standard_normal(A.view(np.float64).shape, dtype=np.float64, out=A.view(np.float64))
    b = np.empty(n, dtype=np.complex128)
    rng.standard_normal(b.view(np.float64).shape, dtype=np.float64, out=b.view(np.float64))
    t_alloc = time.perf_counter() - t0
    ws1, _ = _mem()

    t0 = time.perf_counter()
    x = np.linalg.solve(A, b)
    t_solve = time.perf_counter() - t0
    _, peak = _mem()

    flops = (8.0 / 3.0) * n**3
    return {
        "kind": "solve-only",
        "n_seg": int(n),
        # 记录实现版本:2026-10-13 之前是"先 draw 两份 float64 再 astype"(峰值 40–49 B/元),
        # 之后是就地填 view(32–33 B/元)。jsonl 是跨会话追加的,没有这个字段就无法
        # 区分旧记录 —— 第 13 轮复核正是在这里把一条**旧会话**的 jsonl 记录当成了
        # exp02a 的点(我复核后证伪:exp02a 的 N=10000 两次 repeat 都是 32.83 B/元)。
        "solver_impl": "inplace-view-fill",
        "t_alloc_s": t_alloc,
        "t_solve_s": t_solve,
        "peak_b": peak,
        "ws_after_alloc_b": ws1,
        "alloc_footprint_b": ws1 - ws0,
        "ideal_flops": flops,
        "gflops": flops / t_solve / 1e9,
        "resid_rel": float(np.linalg.norm(A @ x - b) / np.linalg.norm(b)),
    }


def _loglog_fit(xs, ys) -> dict:
    xs = list(xs)
    ys = list(ys)
    if len(xs) < 2:
        return {"intercept": float("nan"), "slope": float("nan"),
                "max_abs_resid_log": float("nan"), "n_points": len(xs),
                "points": [int(x) for x in xs],
                "note": "少于 2 个点,不拟合"}
    lx = np.log(np.asarray(xs, dtype=float))
    ly = np.log(np.asarray(ys, dtype=float))
    A = np.column_stack([np.ones_like(lx), lx])
    coef, *_ = np.linalg.lstsq(A, ly, rcond=None)
    resid = A @ coef - ly
    return {
        "intercept": float(coef[0]),
        "slope": float(coef[1]),
        "max_abs_resid_log": float(np.max(np.abs(resid))),
        "n_points": int(len(xs)),
        "points": [int(x) for x in xs],
    }


def env_facts() -> dict:
    """把会影响时序的环境事实记进产物,否则"某次跑了 12 分钟"无法归因。

    2026-10-13 加:用户机器**断电**中断过一次 2.1,重启后 BLAS 线程数与负载都可能
    不同;没有这些字段就无法判断中断前后两次测量是否可比。
    """
    import platform

    facts = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "cpu_count_logical": os.cpu_count(),
        "thread_env": {k: os.environ.get(k) for k in
                       ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                        "NUMEXPR_NUM_THREADS")},
    }
    try:
        cfg = np.show_config(mode="dicts")
        facts["numpy_config_keys"] = sorted(cfg)
        facts["numpy_blas_name"] = ((cfg.get("Build Dependencies") or {})
                                    .get("blas", {}) or {}).get("name")
    except Exception:  # pragma: no cover
        facts["numpy_blas_name"] = None
    return facts


def _spawn(args: list[str], timeout_s: float) -> dict:
    """在**新进程**里跑 ``args``,读回它打印的最后一行 JSON。"""
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    with tempfile.TemporaryFile("w+", encoding="utf-8", errors="replace") as fh:
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), *args],
            stdout=fh,
            stderr=subprocess.STDOUT,
            env=env,
            timeout=timeout_s,
        )
        fh.seek(0)
        out = fh.read()
    if proc.returncode != 0:
        raise RuntimeError(f"{args} 退出码 {proc.returncode}:\n{out[-2000:]}")
    lines = [ln for ln in out.splitlines() if ln.startswith("{")]
    if not lines:
        raise RuntimeError(f"{args} 没打印 JSON:\n{out[-2000:]}")
    return json.loads(lines[-1])


#: 本次进程的**会话标记**。jsonl 是跨会话追加的,没有它就无法自证"哪几条记录属于同一次运行",
#: 也就无法核对我声称的"单/多线程配对"。2026-10-13 第 13 轮复核实测踩过:复核员只能靠 N 值
#: 回退推断出 26 个会话,配对关系是**启发式**的而不是自证的。
SESSION_TAG = time.strftime("%Y%m%dT%H%M%S", time.gmtime())


def _dump_points(record: dict) -> None:
    RESULTS.mkdir(exist_ok=True)
    record = dict(record, ts_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  session_tag=SESSION_TAG)
    with (RESULTS / "exp02_points.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def effective_thread_label(env: dict) -> tuple[str, str]:
    """判定"这组结果到底跑在几个线程上",并说明依据。

    ⚠️ 不能写成 ``next(v for v in thread_env.values() if v)`` —— 那是**顺序敏感**的:
    只设 `MKL_NUM_THREADS=1` 而 numpy 实际链接 OpenBLAS 时,会把标签撒谎成 single。
    正确做法是先看 numpy 用的是哪个 BLAS,只认它认的那个变量。返回 (标签, 依据)。
    """
    blas = (env.get("numpy_blas_name") or "").lower()
    te = env.get("thread_env") or {}
    if "openblas" in blas:
        key, why = "OPENBLAS_NUM_THREADS", f"numpy 链接 {env.get('numpy_blas_name')}"
    elif "mkl" in blas:
        key, why = "MKL_NUM_THREADS", f"numpy 链接 {env.get('numpy_blas_name')}"
    else:
        # 认不出后端:只要**任何**一个线程变量被设成 1,就按"被钉死"报,并在依据里写明不确定
        setones = [k for k, v in te.items() if str(v) == "1"]
        if setones:
            return "1", f"认不出 BLAS 后端({env.get('numpy_blas_name')!r});但 {', '.join(setones)} = 1"
        return "default", f"认不出 BLAS 后端({env.get('numpy_blas_name')!r}),且无线程变量被设"
    val = te.get(key)
    if val in (None, ""):
        return "default", f"{why};{key} 未设置 ⇒ 后端自选(实测会跑满核)"
    return str(val), f"{why};{key}={val}"


# ----------------------------------------------------------------------------
# 2.1 时间
# ----------------------------------------------------------------------------
def run_time() -> dict:
    print("=" * 78)
    print("E2-A 稠密求解的**时间** vs N(2.1)")
    print("=" * 78)
    ms = system_memory()
    print(f"  物理内存   : {ms['total_phys_b'] / 2**30:.2f} GiB"
          f"(此刻可用 {ms['avail_phys_b'] / 2**30:.2f} GiB,占用 {ms['memory_load_pct']}%)")
    print(f"  ka = {K_WAVE * RADIUS:g}(k={K_WAVE:g}, radius={RADIUS:g})")
    print("  ⚠️ 本机同配置重复测量的散布可达 ~2.5×(负载,非算法),故每点重复多次;")
    print("     拟合用 **min**(负载只会让测量偏慢,min 是成本最干净的估计),median 一并报出。")

    agg: dict[int, dict] = {}
    print()
    print(f"  {'N':>6} | {'t_assemble min/med/max [s]':>28} | {'散布':>5} |"
          f" {'ns/(N²Q) 用 min':>15} | {'t_solve med [s]':>15}")
    print("  " + "-" * 84)
    for n, reps in TIME_REPEATS.items():
        runs = []
        for i in range(reps):
            r = _spawn(["--point", str(n)], timeout_s=3600)
            r["repeat"] = i
            runs.append(r)
            _dump_points(r)
        entry = {
            "repeats": reps,
            "t_assemble": _aggregate(runs, "t_assemble_s"),
            "t_solve": _aggregate(runs, "t_solve_s"),
            "t_total": _aggregate(runs, "t_total_s"),
            "ws_before_b": runs[0]["ws_before_b"],
            "z_footprint_b": runs[0]["z_footprint_b"],
            "z_bytes_per_entry": runs[0]["z_bytes_per_entry"],
            "peak_b": runs[0]["peak_b"],
            "runs": runs,
        }
        agg[n] = entry
        a = entry["t_assemble"]
        print(f"  {n:>6} | {a['min']:>9.3f} /{a['median']:>9.3f} /{a['max']:>9.3f}      |"
              f" {a['spread_ratio']:>5.2f} |"
              f" {a['min'] / (n**2 * 16) * 1e9:>15.1f} |"
              f" {entry['t_solve']['median']:>15.3f}")

    solve_agg: dict[int, dict] = {}
    #: 被内存守卫挡掉的点。**必须落盘**:此前 skip 只打印在 stdout,JSON 里什么都不留,
    #: 第三方无法区分"这个点没跑"与"跑了但没成功"(第 13 轮复核的 Q7②)。
    solve_skips: list[dict] = []
    print()
    print("  —— 纯求解(随机稠密复矩阵,**不含装配**;同点也重复)——")
    print(f"  {'N':>6} | {'t_solve min/med/max [s]':>28} | {'散布':>5} |"
          f" {'GFLOP/s 用 min':>15} | {'相对残差':>10}")
    print("  " + "-" * 84)
    for n, reps in SOLVE_REPEATS.items():
        # A(16 B/元) + LAPACK 侧的第二份 + 解释器基线;/就地填充后 N=10000 实测峰值
        # 3.06 GiB(vs 理论 2.98),故取 34 B/元。40 B/元 是修 bug 前的旧值,会把 N=12000 误挡。
        adm = memory_admit(n)
        if not adm["admit"]:
            _report_skip(adm, solve_skips)
            continue
        runs = []
        for i in range(reps):
            r = _spawn(["--solve", str(n)], timeout_s=3600)
            r["repeat"] = i
            runs.append(r)
            _dump_points(r)
        _require_resid(runs[0])
        agg_t = _aggregate(runs, "t_solve_s")
        solve_agg[n] = {"repeats": reps, "t_solve": agg_t, "runs": runs,
                        "resid_rel": runs[0]["resid_rel"]}
        flops = (8.0 / 3.0) * n**3
        print(f"  {n:>6} | {agg_t['min']:>9.3f} /{agg_t['median']:>9.3f} /{agg_t['max']:>9.3f}      |"
              f" {agg_t['spread_ratio']:>5.2f} | {flops / agg_t['min'] / 1e9:>15.1f} |"
              f" {runs[0]['resid_rel']:>10.2e}")

    return _finish_time(agg, solve_agg, solve_skips, ms)


def _finish_time(agg, solve_agg, solve_skips, ms, *, write=True, payload_ms=None,
                 meta_extra=None):
    """时间实验的**分析 + 落盘**（从 ``run_time`` 抽出,供 ``--fill-time`` 复用）。

    为什么要抽出来:补一个点时**绝不能重跑**已有的点(时间测量受负载影响,重跑会把
    整张表换成另一组噪声),但拟合 / 时间墙 / 图必须按新的点集重算。如果补点脚本
    自己复制一份公式,两处迟早会分叉(D26)⇒ 分析只保留这一份实现。

    ``write=False``:只返回 payload,不落盘、不画图(给回归对比用:
    ``_scratch/replay_exp02a.py`` 用它复算旧 JSON,逐位比对)。
    ``payload_ms``:补点场景下 ``meta.system_memory`` 仍保留**主运行**的机器状态,
    补点那次的机器状态另记在 ``meta_extra`` 里(不能拿新状态冒充主运行)。
    """
    ns = np.array(sorted(agg), dtype=float)
    ta_min = np.array([agg[int(n)]["t_assemble"]["min"] for n in ns])
    ts_min = np.array([agg[int(n)]["t_solve"]["min"] for n in ns])
    aa_min = np.array([agg[int(n)]["t_assemble"]["min"] for n in ns])
    ts_med = np.array([agg[int(n)]["t_solve"]["median"] for n in ns])

    fit_a = _loglog_fit(ns, ta_min)
    fit_a_med = _loglog_fit(ns, [agg[int(n)]["t_assemble"]["median"] for n in ns])
    sn = np.array(sorted(solve_agg), dtype=float)
    fit_s = _loglog_fit(sn, [solve_agg[int(n)]["t_solve"]["min"] for n in sn]) if len(sn) >= 2 else None
    # 小 N 段 BLAS 效率还在爬(实测 50→244 GFLOP/s),会把手里的指数压低,
    # 所以**另报一个只取大 N 的指数**,两者并列才公平。
    sn_big = sn[sn >= 8000]
    fit_s_big = (_loglog_fit(sn_big, [solve_agg[int(n)]["t_solve"]["min"] for n in sn_big])
                 if len(sn_big) >= 2 else None)
    ns_per = float(np.median([agg[int(n)]["t_assemble"]["min"] / (n**2 * 16)
                              for n in ns]) * 1e9)

    print()
    print("  —— 幂律拟合(双对数最小二乘,用 min)——")
    print(f"  装配      T ~ N^{fit_a['slope']:.3f}  (预期 2;最大残差 {fit_a['max_abs_resid_log']:.4f} log)"
          f"   [median 口径: N^{fit_a_med['slope']:.3f}]")
    if fit_s:
        print(f"  纯求解    T ~ N^{fit_s['slope']:.3f}  (预期 3;最大残差 {fit_s['max_abs_resid_log']:.4f} log)"
              f"   [全段 2000…12000]")
    if fit_s_big:
        print(f"  纯求解    T ~ N^{fit_s_big['slope']:.3f}  (预期 3;最大残差 {fit_s_big['max_abs_resid_log']:.4f} log)"
              f"   [只取 N≥8000]")
        # ⚠️ 这一行原来写的是"BLAS 效率已到平台",**被实测证伪**:吞吐在这段仍是
        #    378 → 491 → 487 → 518 GFLOP/s(N=8000/10000/12000/14000),还在爬。
        #    ⇒ 多线程下**没有任何可达的 N 段**是常效率的,所以这里的指数不是渐近指数。
        #    真正的对照是 `--part solve` + OPENBLAS_NUM_THREADS=1(见 exp02b_solve_scaling_single.json)。
        print("             ⚠️ 但这段**不是**常效率段:吞吐仍从 378 爬到 518 GFLOP/s,")
        print("                所以这个指数是**有效指数**、不是渐近指数。要测渐近指数必须钉死线程:")
        print("                `OPENBLAS_NUM_THREADS=1 python experiments/exp02_dense_scaling.py --part solve`")
    print(f"  装配常数 = {ns_per:.0f} ns/(N²Q)  (第 7 轮复核锚点 290;本机大 N 段实测区间"
          f" {min(agg[int(n)]['t_assemble']['min'] / (n**2 * 16) * 1e9 for n in ns):.0f}"
          f"–{max(agg[int(n)]['t_assemble']['min'] / (n**2 * 16) * 1e9 for n in ns):.0f},"
          f" 见下表按 N 的逐点值)")

    # ---- 时间墙:必须把两项分开,不能只拟合 N³ ----
    # 装配是 O(N²Q),求解是 O(N³)。在可达的 N 范围内**装配一直主导**
    # (实测 N=6400:装配数百秒 vs 求解数秒),所以"时间墙"实际上由装配决定;
    # 早先只拟合 t = a + b·N³ 得到的 N≈9000 是把 O(N²) 的项硬塞进 N³ 的结果。
    A2 = np.column_stack([np.ones_like(ns), ns**2])
    coef_a2, *_ = np.linalg.lstsq(A2, ta_min, rcond=None)
    resid_a2 = A2 @ coef_a2 - ta_min
    if len(sn) >= 2:
        A3 = np.column_stack([np.ones_like(sn), sn**3])
        coef_s3, *_ = np.linalg.lstsq(A3, [solve_agg[int(n)]["t_solve"]["min"] for n in sn], rcond=None)
    else:  # pragma: no cover
        coef_s3 = (0.0, 0.0)

    def t_model(n_):
        return coef_a2[0] + coef_a2[1] * n_**2 + coef_s3[0] + coef_s3[1] * n_**3

    def n_wall(sec):
        lo, hi = 10.0, 1e6
        if t_model(hi) < sec:
            return float("nan")
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if t_model(mid) < sec:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    walls = {f"{sec:g}s": n_wall(sec) for sec in (60.0, 300.0, 600.0, 1800.0)}
    print()
    print("  —— 时间墙(两项模型 t = a + b·N² + c + d·N³,都用 min 拟合)——")
    print(f"  装配项 a + b·N² : a = {coef_a2[0]:.3f} s,b = {coef_a2[1]:.4e} s"
          f"(最大残差 {np.max(np.abs(resid_a2)):.3f} s)")
    print(f"  求解项 c + d·N³ : c = {coef_s3[0]:.3f} s,d = {coef_s3[1]:.4e} s")
    for k, v in walls.items():
        share = (coef_a2[0] + coef_a2[1] * v**2) / t_model(v) if v == v else float("nan")
        print(f"  总预算 {k:>6}: N ≈ {v:,.0f}   (装配占 {share * 100:.1f}%)")
    print("  ⇒ 时间墙由**装配**决定,不是求解 —— O(N³) 的求解要在 N 远大于上表时才追上。")

    RESULTS.mkdir(exist_ok=True)
    payload = {
        "meta": {
            "ka": K_WAVE * RADIUS, "k": K_WAVE, "radius": RADIUS,
            "time_repeats": {str(k): v for k, v in TIME_REPEATS.items()},
            "solve_repeats": {str(k): v for k, v in SOLVE_REPEATS.items()},
            "method": "repeats per point, fit on min; machine-load spread reported",
            "system_memory": payload_ms if payload_ms is not None else ms,
            "env": env_facts(),
        },
        "assemble_solve": {str(k): v for k, v in agg.items()},
        "solve_only": {str(k): v for k, v in solve_agg.items()},
        # 实际跑到的点 vs 常量表:两者不一致时,差异只会出现在这里(第 13 轮复核 Q7②)
        "solve_points_run": sorted(int(k) for k in solve_agg),
        "solve_points_skipped": solve_skips,
        "session_tag": SESSION_TAG,
        "fit": {"assemble_min": fit_a, "assemble_median": fit_a_med,
                "solve_only_min": fit_s, "solve_only_min_big_n": fit_s_big},
        "assemble_ns_per_n2q": ns_per,
        "time_wall": {
            "assemble_a_b": [float(coef_a2[0]), float(coef_a2[1])],
            "assemble_max_resid_s": float(np.max(np.abs(resid_a2))),
            "solve_c_d": [float(coef_s3[0]), float(coef_s3[1])],
            "walls": walls,
        },
    }
    payload["meta"].update(meta_extra or {})
    if not write:
        return payload
    out = RESULTS / "exp02a_dense_time.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  已写入 {out.relative_to(ROOT)}")
    fig = make_time_figure(payload)
    if fig:
        print(f"  已写入 {fig.relative_to(ROOT)}")
    return payload


def fill_time(n_target: int, reps: int = 2) -> dict:
    """给**已有**的时间产物补一个装配点,不重跑任何已有点(2.1 的 N=8000 填空)。

    触发:2.1 的台账扫描范围到 N=8000,而主运行(2026-10-12)只到 6400 —— 大 N 点太贵,
    当时押后。补点与主运行**不在同一个窗口**,因此:

    - 每次启动前重新过一遍 ``memory_admit``(不降阈值),逐次记录当刻可用内存与占用;
    - 这些机器状态写进 ``meta.fill_run``,``meta.system_memory`` 仍保留**主运行**的那份;
    - 目标 N 已存在就**拒绝覆盖** —— 时间点重测会把整张表换成另一组噪声,要重测必须
      显式删掉那一条再跑;
    - 分析走 ``_finish_time``(与 ``run_time`` 同一份实现),不复制公式(D26)。
    """
    src = RESULTS / "exp02a_dense_time.json"
    if not src.exists():
        raise SystemExit(f"找不到 {src};先跑 --part time")
    old = json.loads(src.read_text(encoding="utf-8"))
    agg = {int(k): v for k, v in old["assemble_solve"].items()}
    solve_agg = {int(k): v for k, v in old["solve_only"].items()}
    skips = list(old.get("solve_points_skipped") or [])
    if n_target in agg:
        raise SystemExit(f"N={n_target} 已在 {src.name} 里;拒绝覆盖(要重测请先删掉那一条)")
    if n_target not in TIME_REPEATS:
        print(f"  [warn] N={n_target} 不在 TIME_REPEATS 常量里,meta.time_repeats 不会列出它")

    ms = system_memory()
    print("=" * 78)
    print(f"2.1 补点:N={n_target} × {reps} 次(不重跑任何已有点)")
    print(f"  可用物理内存 {ms['avail_phys_b'] / 2**30:.2f} GiB,占用 {ms['memory_load_pct']}%")
    print("  负载纪律:拟合用 min;每次启动前重过内存守卫,逐次记录机器状态。")

    runs: list[dict] = []
    admit_log: list[dict] = []
    for i in range(reps):
        adm = memory_admit(n_target)
        if not adm["admit"]:
            _report_skip(adm, [])
            raise SystemExit("内存守卫不通过,补点中止(**不降阈值**)")
        r = _spawn(["--point", str(n_target)], timeout_s=7200)
        r["repeat"] = i
        r["avail_phys_b_at_admit"] = adm["free_b"]
        r["memory_load_pct_at_admit"] = adm["memory_load_pct"]
        runs.append(r)
        admit_log.append({k: adm[k] for k in ("need_b", "free_b", "memory_load_pct", "rule")})
        _dump_points(r)
        print(f"  #{i}: t_assemble {r['t_assemble_s']:.3f} s / t_solve {r['t_solve_s']:.3f} s"
              f" / 峰值 {r['peak_b'] / 2**20:.1f} MiB(可用 {adm['free_b'] / 2**30:.2f} GiB)")

    entry = {
        "repeats": reps,
        "t_assemble": _aggregate(runs, "t_assemble_s"),
        "t_solve": _aggregate(runs, "t_solve_s"),
        "t_total": _aggregate(runs, "t_total_s"),
        "ws_before_b": runs[0]["ws_before_b"],
        "z_footprint_b": runs[0]["z_footprint_b"],
        "z_bytes_per_entry": runs[0]["z_bytes_per_entry"],
        "peak_b": runs[0]["peak_b"],
        "runs": runs,
    }
    agg[n_target] = entry
    print(f"  ⇒ 装配 min/med/max = {entry['t_assemble']['min']:.3f} / "
          f"{entry['t_assemble']['median']:.3f} / {entry['t_assemble']['max']:.3f} s"
          f"(散布 {entry['t_assemble']['spread_ratio']:.2f}×)")

    meta_extra = {"fill_run": {
        "n": int(n_target), "repeats": int(reps),
        "at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "why": "2.1 台账扫描范围到 N=8000;主运行(2026-10-12)只到 6400",
        "system_memory": ms,
        "admit": admit_log,
        "session_tag": SESSION_TAG,
    }}
    if "env" in old["meta"]:
        # 其它点都是主运行测的:环境快照必须保留那一份,不能拿今天的 Python/numpy 冒充
        meta_extra["env"] = old["meta"]["env"]
    return _finish_time(agg, solve_agg, skips, ms,
                        payload_ms=old["meta"].get("system_memory"),
                        meta_extra=meta_extra)


def make_time_figure(payload: dict):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"  [warn] 跳过作图:{exc}")
        return None

    a_s = payload["assemble_solve"]
    ns = np.array(sorted(int(k) for k in a_s), dtype=float)
    a_min = np.array([a_s[str(int(n))]["t_assemble"]["min"] for n in ns])
    a_max = np.array([a_s[str(int(n))]["t_assemble"]["max"] for n in ns])
    s_min = np.array([a_s[str(int(n))]["t_solve"]["min"] for n in ns])
    s_max = np.array([a_s[str(int(n))]["t_solve"]["max"] for n in ns])

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    _fs = payload["fit"].get("solve_only_min_big_n") or payload["fit"].get("solve_only_min") or {"slope": float("nan")}
    ax = axes[0]
    ax.errorbar(ns, a_min, yerr=[np.zeros_like(a_min), a_max - a_min],
                fmt="o-", capsize=3, label=r"assemble $Z$ (min, bar = max)")
    ax.errorbar(ns, s_min, yerr=[np.zeros_like(s_min), s_max - s_min],
                fmt="s-", capsize=3, label=r"dense solve (min, bar = max)")
    ax.loglog(ns, a_min[-1] * (ns / ns[-1]) ** 2, "k--", lw=1, label=r"$\propto N^2$")
    ax.loglog(ns, s_min[-1] * (ns / ns[-1]) ** 3, "k:", lw=1, label=r"$\propto N^3$")
    ax.set_xlabel("unknowns $N$")
    ax.set_ylabel("wall time [s]")
    ax.set_title(f"dense time vs $N$: assemble $\\propto N^{{{payload['fit']['assemble_min']['slope']:.3f}}}$,"
                 f" solve $\\propto N^{{{_fs['slope']:.3f}}}$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    so = payload["solve_only"]
    if so:
        sn = np.array(sorted(int(k) for k in so), dtype=float)
        st = np.array([so[str(int(n))]["t_solve"]["min"] for n in sn])
        st_max = np.array([so[str(int(n))]["t_solve"]["max"] for n in sn])
        ax.errorbar(sn, st, yerr=[np.zeros_like(st), st_max - st],
                    fmt="^-", color="tab:red", capsize=3, label="solve only (random dense)")
        ax.loglog(sn, st[-1] * (sn / sn[-1]) ** 3, "k:", lw=1, label=r"$\propto N^3$")
        for n, t in zip(sn, st):
            ax.annotate(f"{(8 / 3) * n ** 3 / t / 1e9:.0f}", (n, t),
                        textcoords="offset points", xytext=(4, -11), fontsize=7)
        if payload["fit"].get("solve_only_min_big_n"):
            ax.set_title(f"solve isolated  ($\\propto N^{{{payload['fit']['solve_only_min_big_n']['slope']:.3f}}}$,"
                         " $N\\geq 8000$)  · number = GFLOP/s")
    ax.set_xlabel("$N$")
    ax.set_ylabel("wall time [s]")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    path = FIGURES / "exp02a_dense_time.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------
# 2.2 内存
# ----------------------------------------------------------------------------
def run_memory() -> dict:
    """2.2:峰值内存 vs N,给出内存墙与「峰值首次超过 4 GB」的 N。

    ⚠️ 三个口径必须分开写(否则论文里那张图会被审稿人问穿):
    - **`Z 足迹`** = 装配前后的工作集差,预期 **16 B/元**(complex128 稠密矩阵本体);
    - **`峰值`** = 整个进程的 `PeakWorkingSetSize`,预期 **`e + f·N²`** 且 **`f ≈ 32 B/元`**
      (Z 本体 16 + `np.linalg.solve` 内部复制的一份 16);
    - **`内存墙`** 按**当时可用量**报:同一个 `f` 下"能跑多大"取决于机器此刻有多少空闲,
      只按总量(15.38 GiB)报是错的。
    每点一条**新进程**:峰值工作集是进程内单调量,顺序测量会让小 N 的峰值被大 N 污染
    (⚠️ 这一条只对**峰值**成立;时间测量的"顺序污染"已被第 13 轮复核证伪,见模块 docstring)。
    """
    print("=" * 78)
    print("E2-B 稠密求解的**内存** vs N(2.2)")
    print("=" * 78)
    ms = system_memory()
    avail = ms["avail_phys_b"]
    print(f"  物理内存   : {ms['total_phys_b'] / 2**30:.2f} GiB"
          f"(此刻可用 {avail / 2**30:.2f} GiB,占用 {ms['memory_load_pct']}%)")
    print("  ⚠️ 每点一条**新进程**:峰值工作集在进程内单调,顺序测量会让小 N 的峰值"
          "被大 N 污染。内存量与机器负载无关,故每点 1 次即可。")
    print("  ⚠️ 本实验顺带产出的 t_assemble/t_solve **不得用于 2.1 的时间拟合**"
          "(负载未受控;见 payload.meta.timing_note)。")

    points: list[dict] = []
    skips: list[dict] = []

    # ---- 顶部锚点:按预注册阶梯取第一个能过守卫的 ----
    top = None
    for cand in MEM_TOP_LADDER:
        adm = memory_admit(cand)
        if adm["admit"]:
            top = cand
            print(f"  顶部锚点取 N={cand}:估计需 {adm['need_b'] / 2**30:.2f} GiB,"
                  f"守卫 `need+1.0 ≤ 0.85×可用`({adm['free_b'] / 2**30:.2f} GiB)通过")
            break
        _report_skip(adm, skips)
    plan = list(MEM_REPEATS) + ([top] if (top is not None and top not in MEM_REPEATS) else [])

    print()
    print(f"  {'N':>6} | {'基线':>9} | {'Z 足迹':>10} | {'B/元':>6} | {'峰值':>10} |"
          f" {'峰值-基线':>10} | {'B/元(N²)':>9}")
    print("  " + "-" * 88)
    for n in plan:
        reps = MEM_REPEATS.get(n, 1)
        adm = memory_admit(n)          # 大 N 点要跑十几分钟,运行前再核一次可用内存
        if not adm["admit"]:
            _report_skip(adm, skips)
            continue
        runs = []
        for i in range(reps):
            r = _spawn(["--point", str(n)], timeout_s=3600)
            r["repeat"] = i
            # 记下"这一条是在多少可用内存下跑的":判据要求内存墙按**当时**可用量报。
            # ⚠️ 这是**守卫检查那一刻**的可用量,不是子进程真正分配时的(两者可能差几十秒;
            # 复核 2026-10-19 指出原名 `_at_spawn` 名不副实,已改 `_at_admit`)。
            r["avail_phys_b_at_admit"] = adm["free_b"]
            r["memory_load_pct_at_admit"] = adm["memory_load_pct"]
            runs.append(r)
            _dump_points(r)
        r = runs[0]
        points.append(r)
        base = r["ws_before_b"]
        delta = r["peak_b"] - base
        print(f"  {n:>6} | {base / 2**20:>8.1f}M | {r['z_footprint_b'] / 2**20:>9.1f}M |"
              f" {r['z_bytes_per_entry']:>6.1f} | {r['peak_b'] / 2**20:>9.1f}M |"
              f" {delta / 2**20:>9.1f}M | {delta / n**2:>9.2f}")

    if len(points) < 2:
        raise RuntimeError(
            f"2.2 至少需要 2 个点才能拟合 `peak = e + f·N²`,实际只跑成 {len(points)} 个"
            f"(跳过 {len(skips)} 个);这是机器可用内存不足,不是数据缺陷。")

    ns = np.array([r["n_seg"] for r in points], dtype=float)
    zf = np.array([r["z_footprint_b"] for r in points], dtype=float)
    pk = np.array([r["peak_b"] for r in points], dtype=float)
    per_entry = np.array([r["z_bytes_per_entry"] for r in points], dtype=float)
    fit_z = _loglog_fit(ns, zf)

    A = np.column_stack([np.ones_like(ns), ns**2])
    coef_m, *_ = np.linalg.lstsq(A, pk, rcond=None)
    resid_m = A @ coef_m - pk
    e_b, f_b = float(coef_m[0]), float(coef_m[1])

    def n_at_peak(cap_b: float) -> float:
        v = (cap_b - e_b) / f_b
        return float(math.sqrt(v)) if v > 0 else float("nan")

    # 小 N 点的足迹只有 ~10 MB,而减基线(ws_before ≈ 60 MB)自带加性固定开销
    # (~0.35 MB @ N=800,即 3.4%;几何/分配器保留,不随 N 变)。它把单点系数抬到
    # 16.55、把**全段** log-log 截距抬到 17.89 B/元。那两个数不是物理系数,别引用:
    # 另报「大 N 口径」(N ≥ 3200)的斜率与逐点中位。
    print()
    print("  —— 拟合 ——")
    big = ns >= 3200
    fit_z_big = _loglog_fit(ns[big], zf[big]) if int(big.sum()) >= 2 else None
    z_coef_big = float(np.median(per_entry[big])) if big.any() else float("nan")
    print(f"  Z 足迹  ~ N^{fit_z['slope']:.3f}  (全段,预期 2;最大残差 {fit_z['max_abs_resid_log']:.4f} log)")
    if fit_z_big:
        print(f"  Z 足迹  ~ N^{fit_z_big['slope']:.3f}  (只取 N≥3200;最大残差"
              f" {fit_z_big['max_abs_resid_log']:.4f} log)")
    print(f"  Z 系数  = {z_coef_big:.2f} B/元  (N≥3200 逐点中位;理论 16 = complex128)")
    print(f"            ⚠️ 全段 log-log 截距 {math.exp(fit_z['intercept']):.2f} B/元 **不可引用**:"
          f"小 N 点的固定基线噪声把小值抬高")
    print(f"  峰值    = {e_b / 2**20:.1f} MiB + {f_b:.2f} B·N²  "
          f"(最大残差 {float(np.max(np.abs(resid_m))) / 2**20:.1f} MiB;系数/32 = {f_b / 32.0:.3f})")

    # ---- 台账 2.2 的原验收:「峰值内存首次 > 4 GB 的 N 值」 ----
    # 4 GB 与 4 GiB 差 7.4%,两个口径都报(本项目对"GB/GiB 混写"已有前科:σ_2D 的 2π)。
    four_gb, four_gib = 4.0e9, 4.0 * 2**30
    crossing = {"4.0 GB (4e9 B)": n_at_peak(four_gb),
                "4 GiB (4·2^30 B)": n_at_peak(four_gib)}
    observed_dec = [int(n) for n, p in zip(ns, pk) if p > four_gb]
    observed_gib = [int(n) for n, p in zip(ns, pk) if p > four_gib]
    print()
    print("  —— 峰值首次越过 4 GB 的 N(台账 2.2 原验收) ——")
    print("  ⚠️ GB(4e9)与 GiB(4·2^30)差 7.4%,两个口径都报;『实测』= 有观测点越过了它,"
          "『外推』= 只有拟合。")
    for k, n_cross, obs in (
            ("4.0 GB (4e9 B)", crossing["4.0 GB (4e9 B)"], observed_dec),
            ("4 GiB (4·2^30 B)", crossing["4 GiB (4·2^30 B)"], observed_gib)):
        how = f"✅ 实测越过:N = {obs}" if obs else "⚠️ 外推(没有观测点越过)"
        print(f"  {k:>20}: N ≈ {n_cross:,.0f}   {how}")
    if not observed_gib:
        # 要用**实测**越过,守卫要求 need+1.0 ≤ 0.85·free,按外推 N 反算需要多少空闲。
        need_at_cross = 32 * crossing["4 GiB (4·2^30 B)"] ** 2 + int(0.2 * 2**30)
        free_for_obs = (need_at_cross + SAFETY_MARGIN_B) / 0.85
        if avail >= free_for_obs:
            # 2026-10-19 复核证伪了第一版这里写的"等机器空闲再补":门槛当时**已经满足**,
            # 真因是阶梯顶(11400)低于过零点(11425)。判据的锚点设短了,不能赖机器。
            print(f"     · 要实测越过 4 GiB 需可用 ≥ {free_for_obs / 2**30:.2f} GiB,"
                  f"此刻 {avail / 2**30:.2f} GiB **已够** ⇒ 没越过是**阶梯顶设低了**"
                  f"(本轮顶 N={int(ns[-1])}),不是内存不够;下一轮把阶梯顶抬到 ≥ "
                  f"{math.ceil(crossing['4 GiB (4·2^30 B)']):d}。")
        else:
            print(f"     · 要实测越过 4 GiB 暂不可行:需可用 ≥ {free_for_obs / 2**30:.2f} GiB,"
                  f"此刻 {avail / 2**30:.2f} GiB ⇒ 等机器空闲再补,不降阈值。")
    print(f"     · 实测最大峰值:N={int(ns[-1])} → {float(pk[-1]) / 2**30:.3f} GiB"
          f" = {float(pk[-1]) / 1e9:.3f} GB")
    walls = {
        "avail_now_x0.80": n_at_peak(avail * 0.80),
        # ⚠️ 这一项是**拟合口径**(用 e+f·N² 反解 0.85·avail),与真实守卫差 ~1%;
        # 论文/台账引用墙时必须用下一项 `guard_capacity_now`(与 memory_admit 同一套常数)。
        "fit_capacity_x0.85": n_at_peak(0.85 * avail - SAFETY_MARGIN_B),
        "guard_capacity_now": guard_capacity(avail, ms["total_phys_b"]),
        "idle_window_8.5GiB_x0.80": n_at_peak(8.5 * 2**30 * 0.80),
        "total_15.38GiB_x0.80": n_at_peak(ms["total_phys_b"] * 0.80),
        "budget_11GiB": n_at_peak(11 * 2**30),
        "budget_8GiB": n_at_peak(8 * 2**30),
    }
    print()
    print("  —— 内存墙(同一个拟合,换「可用量口径」就换 N) ——")
    for k, v in walls.items():
        print(f"  {k:>26}: N ≈ {v:,.0f}")
    print(f"  ⇒ 论文引用墙请用 `guard_capacity_now`(真实守卫);"
          f"`fit_capacity_x0.85` 是拟合口径,比守卫松约 1%。")

    RESULTS.mkdir(exist_ok=True)
    payload = {
        "meta": {
            "ka": K_WAVE * RADIUS, "n_list": [int(r["n_seg"]) for r in points],
            "mem_repeats": {str(k): v for k, v in MEM_REPEATS.items()},
            "mem_top_ladder": [int(x) for x in MEM_TOP_LADDER],
            # 预注册阶梯选中了谁(pick) vs 实际跑成了谁(run):若 top 在逐点复核时被 skip,
            # 两者会不同 —— 分开记,JSON 才不会自相矛盾(2026-10-19 复核指出)。
            "mem_top_ladder_pick": None if top is None else int(top),
            "mem_top_run": int(ns[-1]),
            "method": "one fresh process per point; peak = process PeakWorkingSetSize",
            "timing_note": "t_assemble/t_solve 是装配的副产品,负载未受控,"
                           "不得用于 2.1 的时间拟合",
            "system_memory": ms,
            "env": env_facts(),
        },
        "points": points,
        "points_skipped": skips,
        "fit": {"z_footprint": fit_z,
                "z_footprint_big_n": fit_z_big,
                "z_bytes_per_entry_median": float(np.median(per_entry)),
                "z_bytes_per_entry_big_median": z_coef_big,
                "z_footprint_intercept_b_coef": float(math.exp(fit_z["intercept"])),
                "peak_a_b": [e_b, f_b],
                "peak_max_resid_b": float(np.max(np.abs(resid_m))),
                "peak_coef_over_32": f_b / 32.0},
        "mem_walls": walls,
        "four_gb_crossing": {
            "n": crossing,
            "observed_over_4gb_decimal": observed_dec,
            "observed_over_4gib": observed_gib,
            "basis": ("实测" if (observed_dec and observed_gib) else
                      "4.0 GB 实测 / 4 GiB 拟合外推" if observed_dec else
                      "peak = e + f·N² 拟合外推(两个口径都未实测)"),
        },
    }
    out = RESULTS / "exp02c_dense_memory.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  已写入 {out.relative_to(ROOT)}")
    fig = make_memory_figure(payload)
    if fig:
        print(f"  已写入 {fig.relative_to(ROOT)}")
    return payload



def make_memory_figure(payload: dict):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"  [warn] 跳过作图:{exc}")
        return None

    pts = payload["points"]
    ns = np.array([r["n_seg"] for r in pts], dtype=float)
    zf = np.array([r["z_footprint_b"] for r in pts], dtype=float)
    pk = np.array([r["peak_b"] for r in pts], dtype=float)
    e_b, f_b = payload["fit"]["peak_a_b"]

    fig, ax = plt.subplots(figsize=(7.0, 5.4))
    ax.loglog(ns, pk / 2**20, "^-", label="process peak (measured)")
    # 拟合曲线必须画:小 N 处峰值被固定基座抬高,只画 32N² 会让读者误判拟合质量。
    ax.loglog(ns, (e_b + f_b * ns**2) / 2**20, "-", color="#1f77b4", lw=1, alpha=0.55,
              label=r"fit $e+fN^2$")
    ax.loglog(ns, zf / 2**20, "o-", label=r"$Z$ footprint")
    ax.loglog(ns, 16 * ns**2 / 2**20, "k--", lw=1, label=r"$16N^2$ B")
    ax.loglog(ns, 32 * ns**2 / 2**20, "k:", lw=1, label=r"$32N^2$ B")
    # 台账 2.2 的验收线:4 GB 与 4 GiB 两条都画(差 7.4%,0.47% 的差距只有图上能看出来)。
    ax.axhline(4.0e9 / 2**20, color="0.35", ls="-.", lw=1, label="4 GB = 4e9 B")
    ax.axhline(4.0 * 2**30 / 2**20, color="0.60", ls="-.", lw=1,
               label=r"4 GiB = $4\cdot2^{30}$ B")
    ax.annotate(f"{pk[-1] / 2**30:.3f} GiB\n{pk[-1] / 1e9:.3f} GB",
                (ns[-1], pk[-1] / 2**20), textcoords="offset points", xytext=(-64, -34),
                ha="right", fontsize=8)
    ax.set_xlabel("unknowns $N$")
    ax.set_ylabel("memory [MiB]")
    ax.set_title("E2-B dense memory vs $N$ (one fresh process per point)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7.5, loc="lower right")
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    path = FIGURES / "exp02c_dense_memory.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------
# 入口
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# E2-B:只跑"纯求解"扫描。存在的理由是**把两种解释分开**:
#   多线程下实测 T/N³ 从 5.31e-11(N=2000)降到 5.15e-12(N=14000),有效指数只有
#   1.8–2.5,而**理想 flop 数严格是 N³**。差别只能来自"每 flop 的实际吞吐随 N 上升"
#   (实测 50 → 518 GFLOP/s),这几乎全是**并行效率**(小矩阵摊不开 24 线程)。
#   ⇒ 对照实验:把 OPENBLAS_NUM_THREADS=1,并行效率被钉死,指数应当回到 3。
#   单线程/多线程由**环境变量**控制(不在代码里写死),并如实记进 meta.env。
# ----------------------------------------------------------------------------
SOLVE_ONLY_REPEATS = {2000: 2, 4000: 2, 8000: 2, 12000: 1, 14000: 1}


def run_solve() -> dict:
    print("=" * 78)
    print("E2-B 纯求解(随机稠密复矩阵)的 N³ 律 —— 并行效率对照组")
    print("=" * 78)
    env = env_facts()
    print(f"  Python {env['python']} / numpy {env['numpy']} / {env['cpu_count_logical']} 逻辑核")
    print(f"  线程环境: {env['thread_env']}")
    # ⚠️ 不要写 `next(v for v in thread_env.values() if v)`:那是顺序敏感的,只设 MKL 而
    # numpy 实际链接 OpenBLAS 时会把标签**撒谎**成 single(第 13 轮复核 Q7④)。
    tag, why = effective_thread_label(env)
    print(f"  ⇒ 本组为 **{'单线程' if tag == '1' else (tag + ' 线程' if tag != 'default' else '默认(多线程)')}** 对照"
          f"(依据:{why})")
    ms = system_memory()
    print(f"  可用物理内存 {ms['avail_phys_b'] / 2**30:.2f} GiB,占用 {ms['memory_load_pct']}%")
    print()
    print(f"  {'N':>6} | {'t_solve min/med/max [s]':>28} | {'散布':>5} |"
          f" {'GFLOP/s 用 min':>15} | {'T/N³ [s]':>11} | {'p_local':>8} | {'残差':>9}")
    print("  " + "-" * 100)

    agg: dict[int, dict] = {}
    skips: list[dict] = []
    prev = None
    for n, reps in SOLVE_ONLY_REPEATS.items():
        adm = memory_admit(n)
        if not adm["admit"]:
            _report_skip(adm, skips)
            continue
        runs = []
        for i in range(reps):
            r = _spawn(["--solve", str(n)], timeout_s=7200)
            r["repeat"] = i
            runs.append(r)
            _dump_points(r)
        _require_resid(runs[0])
        t = _aggregate(runs, "t_solve_s")
        gfl = (8.0 / 3.0) * n**3 / t["min"] / 1e9
        agg[n] = {"repeats": reps, "t_solve": t, "runs": runs,
                  "resid_rel": runs[0]["resid_rel"], "gflops_min": gfl,
                  "peak_b": runs[0]["peak_b"], "t_over_n3": t["min"] / n**3}
        p_loc = (math.log(t["min"] / prev[1]) / math.log(n / prev[0])) if prev else float("nan")
        print(f"  {n:>6} | {t['min']:>9.3f} /{t['median']:>9.3f} /{t['max']:>9.3f}      |"
              f" {t['spread_ratio']:>5.2f} | {gfl:>15.1f} | {t['min'] / n**3:>11.3e} |"
              f" {p_loc:>8.3f} | {runs[0]['resid_rel']:>9.2e}")
        prev = (n, t["min"])

    sn = np.array(sorted(agg), dtype=float)
    ts = np.array([agg[int(n)]["t_solve"]["min"] for n in sn])
    fit = _loglog_fit(sn, ts) if len(sn) >= 2 else None
    A = np.column_stack([sn**2, sn**3])
    coef2, *_ = np.linalg.lstsq(A, ts, rcond=None)
    pred2 = A @ coef2
    mre2 = float(np.max(np.abs(pred2 - ts) / ts))

    print()
    if fit:
        print(f"  单幂律: T ~ N^{fit['slope']:.3f}(预期 3;最大残差 {fit['max_abs_resid_log']:.4f} log)")
    print(f"  两项模型: T = {coef2[0]:.4e}·N² + {coef2[1]:.4e}·N³  最大相对残差 {mre2 * 100:.2f}%")
    print(f"  T/N³ : " + " → ".join(f"{agg[int(n)]['t_over_n3']:.3e}" for n in sn))
    print(f"  吞吐 : " + " → ".join(f"{agg[int(n)]['gflops_min']:.0f}" for n in sn) + " GFLOP/s")

    RESULTS.mkdir(exist_ok=True)
    payload = {
        "meta": {
            "ka_unused": None,
            "method": "random dense complex LU only; repeats per point, fit on min",
            "solve_only_repeats": {str(k): v for k, v in SOLVE_ONLY_REPEATS.items()},
            "threads": tag,
            "threads_evidence": why,
            "solve_points_run": sorted(int(k) for k in agg),
            "solve_points_skipped": skips,
            "session_tag": SESSION_TAG,
            "system_memory": ms,
            "env": env,
        },
        "solve_only": {str(k): v for k, v in agg.items()},
        "fit": {"single_power_min": fit,
                "two_term_N2_N3_coef": [float(coef2[0]), float(coef2[1])],
                "two_term_max_rel_resid": mre2},
        "per_point": [{"n": int(n), "t_min_s": float(agg[int(n)]["t_solve"]["min"]),
                       "gflops": float(agg[int(n)]["gflops_min"]),
                       "t_over_n3": float(agg[int(n)]["t_over_n3"])} for n in sn],
    }
    tag_name = "single" if tag == "1" else "multi"
    out = RESULTS / f"exp02b_solve_scaling_{tag_name}.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  已写入 {out.relative_to(ROOT)}")
    fig = make_solve_figure(payload, tag_name)
    if fig:
        print(f"  已写入 {fig.relative_to(ROOT)}")
    return payload


def make_solve_figure(payload: dict, tag: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover
        print("  (matplotlib 不可用,跳过图)")
        return None
    pts = payload["per_point"]
    n = np.array([p["n"] for p in pts], float)
    t = np.array([p["t_min_s"] for p in pts], float)
    g = np.array([p["gflops"] for p in pts], float)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax1.loglog(n, t, "o-", color="#1f77b4", label="measured $t_{solve}$")
    ref = t[0] * (n / n[0]) ** 3
    ax1.loglog(n, ref, "--", color="#d62728", label="$N^3$ reference")
    ax1.set_xlabel("$N$ (segments)"); ax1.set_ylabel("$t$ [s]")
    ax1.set_title(f"dense complex LU — {tag}-thread")
    ax1.grid(True, which="both", alpha=.3); ax1.legend(fontsize=9)
    ax2.semilogx(n, g, "s-", color="#2ca02c")
    ax2.set_xlabel("$N$ (segments)"); ax2.set_ylabel("GFLOP/s (min)")
    ax2.set_title("achieved throughput")
    ax2.grid(True, which="both", alpha=.3)
    fig.tight_layout()
    path = FIGURES / f"exp02b_solve_scaling_{tag}.png"
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def run(part: str = "all") -> dict:
    out: dict = {}
    if part in ("time", "all"):
        out["time"] = run_time()
    if part in ("solve", "all"):
        out["solve"] = run_solve()
    if part in ("memory", "all"):
        out["memory"] = run_memory()
    return out


def _point_main(n: int) -> int:
    r = measure_one(n)
    print(f"N={n} t_assemble={r['t_assemble_s']:.4f}s t_solve={r['t_solve_s']:.4f}s "
          f"peak={r['peak_b'] / 2**20:.1f}MiB")
    print(json.dumps(r, ensure_ascii=False))
    return 0


def _solve_main(n: int) -> int:
    r = solve_only(n)
    print(f"N={n} t_solve={r['t_solve_s']:.4f}s gflops={r['gflops']:.1f}")
    print(json.dumps(r, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    argv = sys.argv[1:]
    if len(argv) >= 2 and argv[0] == "--point":
        raise SystemExit(_point_main(int(argv[1])))
    if len(argv) >= 2 and argv[0] == "--solve":
        raise SystemExit(_solve_main(int(argv[1])))
    if len(argv) >= 2 and argv[0] == "--fill-time":
        reps = int(argv[2]) if len(argv) >= 3 else 2
        fill_time(int(argv[1]), reps)
        raise SystemExit(0)
    if len(argv) >= 2 and argv[0] == "--part":
        run(argv[1])
    else:
        run("all")
