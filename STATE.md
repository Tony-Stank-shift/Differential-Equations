# STATE — 当前进度

> 每次小项完成后更新。上下文压缩或中断后,靠本文件无损接上。

## 闸门进度

- [x] 闸门 ① 题意还原
- [x] 闸门 ② 知识盘点
- [x] 闸门 ③ 差距补课(M1–M6 已讲,自测全部通过)
- [x] 闸门 ④ 方案与创新度(路线 R2 / 适当创新 / 23 项 WBS)
- [ ] 闸门 ⑤ 执行

## WBS 状态

| # | 小项 | 状态 |
|---|---|---|
| 0 | 目录骨架 + 环境确认 | doing |
| 1.1 | green.py:格林函数 + R→0 渐近 | todo |
| 1.2 | quad.py:奇异积分 | todo |
| 1.3 | mom.py:稠密求解 | todo |
| 1.4 | ⭐ rcs.py + Mie 对比 | todo |
| 2.1 | exp02:稠密时间/内存 vs N | todo |
| 2.2 | 定位内存墙 | todo |
| 3.1 | cluster.py:二叉树聚类 | todo |
| 3.2 | aca.py:部分主元 ACA | todo |
| 3.3 | hmatrix.py:组装 + 存储统计 | todo |
| 3.4 | exp04:秩 vs ka | todo |
| 4.1 | 数据集生成器 | todo |
| 4.2 | 特征工程 | todo |
| 4.3 | 网络 + 非对称损失 | todo |
| 4.4 | exp06:端到端三方对比 | todo |
| 5.1 | 论文正文 + 公式排版 | todo |
| 5.2 | 全部图表 | todo |
| 5.3 | 附录 B:N=4 手算 | todo |
| 5.4 | 附录 A:代码清单 | todo |
| 5.5 | 全文通读 + 参考文献 | todo |
| 6.1 | 第三方独立复核 | todo |
| 6.2 | 组内分工交接 | todo |
| S1 | [可选] 方向性分块秩测量 | todo |

## 当前阻塞

- **依赖缺口**:scipy / matplotlib / torch 未安装。安装方案已提交用户审批。

## 已知环境事实

- Python 3.12.14(DSH 运行时)
- 机器:Intel Core Ultra 9 290HX Plus,24 逻辑核,16 GB RAM(实测 15.38 GB)
- 无 GPU,全部计算走 CPU
- 实测复数 LU 有效算力约 39–63 GFLOPs
- 内存是唯一硬瓶颈:安全算例上限 N ≤ 5000;勉强 N = 8000–10000(须关闭浏览器等)

## 下一步

1. 获批准后建 `.venv` 并安装 scipy + matplotlib(本阶段);torch 留到第 4 阶段
2. 开工 1.1:`green.py`
