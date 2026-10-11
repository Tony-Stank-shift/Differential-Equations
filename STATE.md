# STATE — 当前进度

> 每次小项完成后更新。上下文压缩或中断后,靠本文件无损接上。
> 最后更新:2026-10(1.1 完成、第三方复核完毕并反馈修入)

## 闸门进度

- [x] 闸门 ① 题意还原
- [x] 闸门 ② 知识盘点
- [x] 闸门 ③ 差距补课(M1–M6 已讲,自测全部通过)
- [x] 闸门 ④ 方案与创新度(路线 R2 / 适当创新 / 23 项 WBS)
- [~] 闸门 ⑤ 执行 —— 进行中

## WBS 状态

| # | 小项 | 状态 |
|---|---|---|
| 0 | 目录骨架 + 环境确认 | **done** |
| 1.1 | green.py:格林函数 + R→0 渐近 | **done**(自检 + 复核均通过,复核 9 项反馈已修入) |
| 1.2 | quad.py:奇异积分 | doing |
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
| 6.1 | 第三方独立复核 | 进行中(1.1 已复核) |
| 6.2 | 组内分工交接 | todo |
| S1 | [可选] 方向性分块秩测量 | todo |

## ⚠️ 开工 1.2 前必须记住的三条(全部来自 1.1 的复核)

1. **符号体系已锁定**(DECISIONS D6):`e^{+jωt}` / `(∇²+k²)G = +δ` / `G = (j/4)H₀⁽²⁾` / `u_inc = exp(+jk·r)`。**这四条绑在一起,不许单独改**。仓库外的知识库笔记曾写错源项符号,已就地更正。
2. **`log_singular_part` 的余项只有在 kR ≲ 0.05 时才能当常数 1 用**。实测偏离:kR=0.05 → 1.75e-3;0.3 → 3.99e-2;**0.5 → 9.36e-2(近 10%)**。`quad.py` 的自耦合项若把它当常数,对角元会错约 10%。
3. **两个守卫已就位**:`hankel0_2_small` 在 |x|>5 时告警;`log_singular_part` 在 kR=0 时告警。它们防的是静默错。

## 已落盘

- `proj/src/green.py` —— `EULER_GAMMA` / `SMALL_THRESHOLD=0.5` / `hankel0_2_small` / `hankel0_2` / `green_2d` / `log_singular_part`;顶部写死全局约定
- `proj/tests/check_green.py` —— 判据 A(对 mpmath 60dps 真值)+ 判据 B(对 scipy)+ 拼接缩放检验 + 余项门槛表 + 误用守卫测试
- `proj/{README,requirements,.gitignore,run_all.py,DECISIONS,STATE}.md`、`proj/experiments/registry.md`
- Git:`b5773a1` 骨架 → `3ef34de` 1.1 初版 → (待提交)复核反馈修入

## 环境事实(实测)

- Python:`python`(3.12.10);numpy 2.5.1 / scipy 1.18.1 / matplotlib 3.11.1 / torch 2.13.0+cpu / **mpmath 1.3.0**
- 机器:Intel Core Ultra 9 290HX Plus,24 逻辑核,**16 GB RAM(实测 15.38 GB)**,无 GPU
- 实测复数 LU 有效算力约 39–63 GFLOPs
- **内存是唯一硬瓶颈**:安全算例上限 N ≤ 5000;勉强 N = 8000–10000(须关闭浏览器等)
- 沙箱在 workspace-write 下**禁止 pwsh 子进程出站联网**(curl 全部返回 `000`);机器本身有网,Google/HF/Scholar 需梯子
- ⚠️ 中文路径在子进程 stderr 中会被 GBK 破坏(Python 自身运行不受影响)
- 本机 PowerShell 为 5.1(`Get-Content` 无 `-AsByteStream`;`Get-CimInstance` 被沙箱拒绝)

## 下一步

开工 **1.2 `quad.py`:奇异积分(自耦合项)** —— 本项目最易翻车处。要点:
- 核拆成 `log_singular_part`(解析可积)+ 正则余项,但**必须按 kR ≲ 0.05 的门槛判断余项能否取 1**
- 验收判据同样要用独立真值(mpmath 或高精度数值积分),不能拿自己的实现当基准(纪律 D8)
