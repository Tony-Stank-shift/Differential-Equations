# STATE — 当前进度

> 每次小项完成后更新。上下文压缩或中断后,靠本文件无损接上。
> 最后更新:2026-10(1.2 quad.py 完成,自检全通过;第三方复核已派出)

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
| 1.2 | quad.py:奇异积分 | **done**(自检全通过:8 组 (k,a) vs mpmath 最大 rel_err `5.5e-15`;自捕获 2 个真 bug;待第三方复核回收) |
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
| 6.1 | 第三方独立复核 | 进行中(1.1 已复核通过;1.2 复核已派出) |
| 6.2 | 组内分工交接 | todo |
| S1 | [可选] 方向性分块秩测量 | todo |

## ⚠️ 开工 1.3 前必须记住的五条

1. **符号体系已锁定**(DECISIONS D6):`e^{+jωt}` / `(∇²+k²)G = +δ` / `G = (j/4)H₀⁽²⁾` / `u_inc = exp(+jk·r)`。**这四条绑在一起,不许单独改**。
2. **自耦合积分的主路径是级数逐项解析积分**(DECISIONS D9),不是数值求积。1.3 里 `Z_mm = -(j/4) · I(k, a)`,直接调 `self_coupling_integral`。
3. **级数路径只在 `k·a ≤ 0.5` 收敛**,超出抛 `ValueError`(不静默给坏值)。`ka > 0.5` 时 `self_coupling_integral` 会按 `δ = 0.5/k` 自动分段(内段级数 + 外段 Gauss),**外段有两条(±),对称因子不能漏**。
4. **1.3 的近奇异积分要用"拆主部 + 数值求积"那条备用路径**:先把对数主部与**二阶主部**一起拆掉(`C0`/`C1`,见 `quad.py`),余项才是 `O(x⁴ln x)`。只拆对数主部时余项是 `O(x²ln x)`,40 点以下的 Gauss 明显不够;而把余项当常数 1 会让对角元错约 10%。
5. **守卫清单**:`hankel0_2_small` 在 |x|>5 告警;`log_singular_part` 在 kR=0 告警;`series_self_integral` 在 ka>0.5 抛异常。它们防的都是静默错。

## 已落盘

- `proj/src/green.py` —— `EULER_GAMMA` / `SMALL_THRESHOLD=0.5` / `hankel0_2_small` / `hankel0_2` / `green_2d` / `log_singular_part`;顶部写死全局约定
- `proj/src/quad.py` —— `gauss_legendre` / `log_term_integral` / `poly2_term_integral` / `analytic_part_integral` / `regular_part` / **`series_self_integral`(主路径)** / **`self_coupling_integral`(主路径,自动分段)** / `direct_gauss` + `naive_self_coupling`(错误做法,仅供自检量化危害)
- `proj/tests/check_green.py` —— 判据 A(对 mpmath 60dps 真值)+ 判据 B(对 scipy)+ 拼接缩放检验 + 余项门槛表 + 误用守卫测试
- `proj/tests/check_quad.py` —— 判据 A/A2(主路径对 mpmath 高精度积分)+ 判据 B(两个闭式主部)+ 判据 B3(二阶系数正确性)+ 判据 C(分段稳定性)+ 判据 D(错误做法危害量化)+ 判据 E(形状/边界/微积分一致性)
- `proj/{README,requirements,.gitignore,run_all.py,DECISIONS,STATE}.md`、`proj/experiments/registry.md`

## 1.2 实测数字(留给论文与复核)

- 主路径 vs mpmath:8 组 (k,a),ka ∈ [0.05, 5],最大相对误差 **`5.5e-15`**;ka ≤ 0.5 的 5 组恰为 `0`
- 级数单独(ka = 0.01…0.5,6 点)最大相对误差 **`1.9e-16`**
- 闭式对数主部 **`1.8e-16`**;闭式二阶主部 **`3.4e-16`**
- 余项门槛验证:`|R2-1|/x²` 随 x 单调趋零(x=0.4 → 4.7e-3,x=0.02 → 2.3e-5),证明二阶项确实被拆干净
- 错误做法对照:只拆对数主部最差 **`6.06e+0`**(ka=5);直接对奇异核求积 **`3.8e-3 ~ 9.6e-3`**
- 微积分一致性 `dI/da = 2H₀⁽²⁾(ka)`:**`3.7e-11`**
- 交叉验证(k=2, a=0.3):级数路径 `1.3e-16` / 主部+求积路径 `1.9e-12`

## 环境事实(实测)

- Python:`python`(3.12.10);numpy 2.5.1 / scipy 1.18.1 / matplotlib 3.11.1 / torch 2.13.0+cpu / **mpmath 1.3.0**
- 机器:Intel Core Ultra 9 290HX Plus,24 逻辑核,**16 GB RAM(实测 15.38 GB)**,无 GPU
- 实测复数 LU 有效算力约 39–63 GFLOPs
- **内存是唯一硬瓶颈**:安全算例上限 N ≤ 5000;勉强 N = 8000–10000(须关闭浏览器等)
- 沙箱在 workspace-write 下**禁止 pwsh 子进程出站联网**(curl 全部返回 `000`);机器本身有网,Google/HF/Scholar 需梯子
- ⚠️ 中文路径在子进程 stderr 中会被 GBK 破坏(Python 自身运行不受影响)
- 本机 PowerShell 为 5.1(`Get-Content` 无 `-AsByteStream`;`Get-CimInstance` 被沙箱拒绝)

## 下一步

开工 **1.3 `mom.py`:矩量法稠密求解**。要点:
- 几何:二维 PEC 圆柱,边界离散成长度接近的直线段(N 段 → N 个未知量,N 段的中点做配点)
- `Z_mn = -(j/4) ∫_{段 n} G(r_m, r') dl'`,自耦合 `m=n` 直接调 `self_coupling_integral`
- 激励向量 `V_m = u_inc(r_m) = exp(+jk(x_m cosθ_i + y_m sinθ_i))`
- 验收:解出表面电流后算 RCS,与 Mie 级数解析解对比(这一步是 1.4,但 1.3 就要把接口留好)
- 判据必须用独立真值(纪律 D8):Mie 级数是解析解,天然满足
 
