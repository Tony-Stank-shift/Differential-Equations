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

- Python 3.12
- 依赖见 `requirements.txt`
- **全部安装在项目内虚拟环境 `.venv`,不污染系统 Python 与 DSH 运行时**

### 搭建

```
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
```

### 一键复现

```
.venv/Scripts/python run_all.py
```

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
