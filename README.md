# 二维电磁积分散射问题的快速矩阵分解与神经网络加速

> 《微分方程》课程论文 · 上海大学 · 2026 秋冬

## 1. 题目

**基于快速矩阵分解算法的电磁积分散射问题的求解**

研究内容:利用快速矩阵分解算法,设计高效的神经网络模型,来求解二维的电磁积分散射问题。

## 2. 技术路线(R2)

```
二维 TM 电磁散射(理想导体 PEC 圆柱)
        │
        ├─ 电场积分方程 EFIE
        │
        ├─ 矩量法 MoM 离散  ──→  稠密阻抗矩阵 Z
        │                              │
        │                              ├─ 基线:直接 LU 求解
        │                              │        O(N^3) 时间 / O(N^2) 内存
        │                              │
        │                              └─ 加速:ACA + H 矩阵
        │                                       近 O(N log N) 时间 / 近 O(N) 内存
        │                                            │
        │                                            └─ 神经网络预测远场块秩
        │                                               免去自适应判秩的迭代开销
        │
        └─ Mie 级数解析解  ──→  正确性验证
```

## 3. 实验清单

| 编号 | 内容 | 目的 |
|---|---|---|
| E1 | MoM 解 vs Mie 级数解析解 | 正确性验证 |
| E2 | 稠密求解时间 / 内存 vs N | 展示 O(N^3) 增长与内存墙 |
| E3 | ACA 块秩特性 | 低秩性的经验证据 |
| E4 | H 矩阵组装 / 求解时间 vs N | 展示加速效果 |
| E5 | 块秩 vs 电尺寸 ka | 找出失效边界 |
| E6 | 固定秩 / ACA 自适应 / NN 预测 三方对比 | 端到端收益 |

## 4. 环境

使用本机用户级 Python 3.12(标准安装,已有 numpy / matplotlib / torch):

```
python
```

实测版本:

| 包 | 版本 |
|---|---|
| Python | 3.12.10 |
| numpy | 2.5.1 |
| scipy | 1.18.1 |
| matplotlib | 3.11.1 |
| torch | 2.13.0+cpu |

无 GPU,全部计算走 CPU。

### 补齐依赖

本机已装齐。若在其他机器重建:

```
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

### 一键复现

```
python run_all.py
```

### 验证脚本

```
python tests/check_green.py      # 1.1 格林函数
python tests/check_quad.py       # 1.2 奇异积分
python tests/check_mom.py        # 1.3 矩量法(含黄金值 J 与鉴别力 K 判据)
```

### 自耦合积分走哪条路(容易记反,务必看这条)

统一入口 `quad.self_coupling_integral(k, half_len)`,默认 `method="auto"`,按 `k*a` 自动选:

| 条件 | 实际选中 | 说明 |
|---|---|---|
| `k*a <= 0.5` | `series_self_integral`(级数逐项解析) | **真实 MoM 网格恒落在这一支** |
| `k*a > 0.5` | `closed_self_integral`(Struve 闭式) | D10 的"主路径",本项目里未被 `auto` 选中过 |

`mom.py` 的段长用**弦长** `2R sin(π/N)`,故 `k*a = π·ka/N ≈ 1e-2`(实测 ka=5/N=400 时
`3.93e-2`)。两条路数值都正确,但**论文里不要写"默认走 Struve 闭式"**。详见
`DECISIONS.md` 的 D10 补充更正与 `src/quad.py` 模块 docstring 顶端。

## 5. 目录

```
proj/
├─ README.md          本文件
├─ requirements.txt   依赖
├─ run_all.py         一键复现
├─ STATE.md           当前进度
├─ DECISIONS.md       关键技术决策与理由
├─ src/
│    green.py         格林函数(含 R->0 渐近分支)
│    quad.py          奇异积分处理
│    mom.py           矩量法:填充 Z、V
│    rcs.py           散射宽度与 Mie 级数解析解
│    cluster.py       二叉树聚类
│    aca.py           部分主元 ACA
│    hmatrix.py       H 矩阵组装与存储
│    nnrank.py        神经网络秩预测
│    solve.py         线性求解
├─ experiments/
│    registry.md      实验登记簿(配置 -> 脚本 -> 输出 -> 图)
├─ results/           数值结果
└─ figures/           图(dpi >= 300)
```

## 6. 相关文档

- 进度:`STATE.md`
- 决策:`DECISIONS.md`
- 实验记录:`experiments/registry.md`
