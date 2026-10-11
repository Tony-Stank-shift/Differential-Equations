# STATE — 当前进度

> 每次小项完成后更新。上下文压缩或中断后,靠本文件无损接上。
> 最后更新:2026-10(1.2 完成:自检全通过 + 第三方复核 4 项已全部修入)

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
| 1.1 | green.py:格林函数 + R→0 渐近 | **done**(自检 + 复核均通过,复核 9 项已修入) |
| 1.2 | quad.py:奇异积分 | **done**(自检全通过 + 第三方复核 4 项已修入;主路径已换 Struve 闭式) |
| 1.3 | mom.py:稠密求解 | doing |
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
| 6.1 | 第三方独立复核 | 进行中(1.1、1.2 均已复核通过) |
| 6.2 | 组内分工交接 | todo |
| S1 | [可选] 方向性分块秩测量 | todo |

## ⚠️ 开工 1.3 前必须记住的五条

1. **符号体系已锁定**(DECISIONS D6):`e^{+jωt}` / `(∇²+k²)G = +δ` / `G = (j/4)H₀⁽²⁾` / `u_inc = exp(+jk·r)`。**这四条绑在一起,不许单独改**。
2. **自耦合积分走 `self_coupling_integral`(统一入口,默认 `method="auto"`)**,不要自己去分段。主路径是 Struve 闭式(D10),O(1) 全域有效。
3. **1.3 的近奇异积分**(场点靠近但不在段上)用**备用路径**:`analytic_part_integral` + `regular_part` 的 Gauss 求积。注意它**只在 `k*a <= 0.5` 保证 1e-12**(实测 ka=0.5 → 8.58e-13,ka=1 → 1.83e-11,ka=5 → 8.62e-8);ka 大时要提高 `order`(ka=5 需 ≥512)。误差全在 `∫R2` 上,不是主部。
4. **不要对自耦合用任何"整段高斯求积"**:`direct_gauss` 实测误差停在 `3.8e-3 ~ 9.6e-3`,提高阶数无效(被积函数本身发散)。`naive_self_coupling`(把余项当常数 1)最差 `6.06e+00`。
5. **凡给数值函数定参数,先扫参数空间找失效边界并写成断言** —— 这是 D10 的教训。`check_quad.py` 判据 C2 就是这么来的(它断言分段路径在大 ka **必须**失效)。

## 已落盘

- `proj/src/green.py` —— `EULER_GAMMA` / `SMALL_THRESHOLD=0.5` / `hankel0_2_small` / `hankel0_2` / `green_2d` / `log_singular_part`
- `proj/src/quad.py` —— **主路径** `closed_self_integral` + `hankel0_antiderivative`(Struve 闭式);**对照** `series_self_integral`;**备用** `analytic_part_integral` / `regular_part` / `split_self_integral`;入口 `self_coupling_integral(method=auto|closed|series|split)`;**错误做法** `direct_gauss` / `naive_self_coupling`;常量 `C0` / `C1` / `DEFAULT_QUAD_ORDER`
- `proj/tests/check_green.py` —— 判据 A(mpmath 60dps)+ B(scipy)+ 拼接缩放 + 余项门槛表 + 守卫测试
- `proj/tests/check_quad.py` —— 判据 A / A2(大 ka 守护)/ A3(双实现)/ B / B3 / C / **C2(防回归)** / D / E / F(备用路径断言)/ G(浮点边界)
- `proj/tests/repro_review.py` —— 复核最小反例存档(用 mpmath 独立真值,非自比)
- `proj/{README,requirements,.gitignore,run_all.py,DECISIONS,STATE}.md`、`proj/experiments/registry.md`

## 1.2 最终实测数字

**自检(`check_quad.py`,EXIT=0 全通过)**

| 判据 | 内容 | 结果 |
|---|---|---|
| A | 主路径 8 组 (k,a),ka ∈ [0.05, 5] vs mpmath | **`3.138e-16`** |
| A2 | 大 ka:10 / 52 / 256 / 1000 / 1e4 vs mpmath Struve | **`2.442e-14`** |
| A2 | 闭式公式本身 vs `mp.quad`(小 ka) | **`0.000e+00`** |
| A3 | 闭式 vs 级数(两条独立实现),ka ∈ [0.01, 0.5] | **`3.110e-16`** |
| B | 对数主部 / 二阶主部 / 两者之和 | `1.787e-16` / `3.427e-16` / `1.853e-16` |
| B3 | `|R2-1|/x²` 单调趋零;比值 vs `x⁴|ln(x/2)|` ∈ [0.0124, 0.0183] | PASS |
| C | 分段路径 order 32/64/128 → `3.178e-15` / `5.468e-15` / `2.437e-14` | PASS |
| **C2** | **分段路径 ka=1e4 必须失效(`1.058e+01`),主路径同期 `2.442e-14`** | PASS |
| D1 | `naive_self_coupling` 最差 | **`6.058e+00`** |
| D2 | `direct_gauss` | `3.836e-3 ~ 9.606e-3` |
| E | `dI/da = 2H₀⁽²⁾(ka)`;三条路径同点 | `9.607e-11`;三者 `0.000e+00` |
| F | 备用路径 ka ≤ 0.5(有断言) | **`8.576e-13`** |
| G | `a = 0.5/k` × 3 缩放 × 3 method = 45 次调用 | 无无谓 ValueError |

**第三方复核(`quad.py`,独立 subagent,默认找茬)**

- 判定:**通过**(声称 ①②③④⑤ 在声称域内全成立),无 blocker;发现 4 项,已全部修入
- 复核员给了**三套独立真值**:自推 Struve 闭式、ODE 积分(`solve_ivp` DOP853)、`mp.quad`;两者互证 z=1000 时 `1.47e-13`
- 复核员的数字我已逐条复现(`tests/repro_review.py`):分段 ka=52 → `1.40e-12`、ka=256 → `4.06`、ka=1000 → `5.53`;备用路径 ka=5 order64 → `8.62e-08`;`2a·H₁⁽²⁾` 误差 `5.26e-01` / `3.25e+00` —— **全部一致**
- 复核员**推翻了我的两条断言**:①"这个积分没有闭式"错(有 Struve 闭式);②"主路径"其实在 ka ≳ 52 就失效
- 复核确认**正确**的部分:级数系数递推逐行审无异、`C0`/`C1` 逐位相同、对称因子 2 的修正正确、朴素近似确实会错 ~10%

## 环境事实(实测)

- Python:`python`(3.12.10);numpy 2.5.1 / scipy 1.18.1 / matplotlib 3.11.1 / torch 2.13.0+cpu / **mpmath 1.3.0**
- 机器:Intel Core Ultra 9 290HX Plus,24 逻辑核,**16 GB RAM(实测 15.38 GB)**,无 GPU;实测复数 LU 有效算力约 39–63 GFLOPs
- **内存是唯一硬瓶颈**:安全算例上限 N ≤ 5000;勉强 N = 8000–10000(须关闭浏览器等)
- 沙箱在 workspace-write 下**禁止 pwsh 子进程出站联网**(curl 全部返回 `000`);机器本身有网,Google/HF/Scholar 需梯子
- ⚠️ 中文路径在子进程 stderr 中会被 GBK 破坏(Python 自身运行不受影响)
- 本机 PowerShell 为 5.1(`Get-Content` 无 `-AsByteStream`;`Get-CimInstance` 被沙箱拒绝)

## 下一步

开工 **1.3 `mom.py`:矩量法稠密求解**。要点:
- 几何:二维 PEC 圆柱,边界离散成 N 段(长度尽量均匀),配点取段中点
- 非对角元:`Z_mn = -(j/4) ∫_{段 n} G(r_m, r') dl'`,段上 `R` 有正的下确界;N 较大时会出现近奇异,走备用路径(`analytic_part_integral` + `regular_part`)
- 自耦合元:`Z_mm = -(j/4) * self_coupling_integral(k, Δ/2)`
- 激励:`V_m = u_inc(r_m) = exp(+jk(x_m cosθ_i + y_m sinθ_i))`
- 验收:解出表面电流后算 RCS,与 Mie 级数解析解对比(那是 1.4,但 1.3 要把接口留好)
- 判据必须用独立真值(纪律 D8):Mie 级数是解析解,天然满足
