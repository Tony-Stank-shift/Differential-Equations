# STATE — 当前进度

> 每次小项完成后更新。上下文压缩或中断后,靠本文件无损接上。
> 最后更新:2026-10(1.3 完成:自检全通过;待第三方复核)

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
| 1.3 | mom.py:稠密求解 | **自检 done,待第三方复核** |
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
| 6.1 | 第三方独立复核 | 进行中(1.1、1.2 已通过;1.3 待派) |
| 6.2 | 组内分工交接 | todo |
| S1 | [可选] 方向性分块秩测量 | todo |

## ⚠️ 开工 1.4 前必须记住的五条

1. **符号体系已锁定**(DECISIONS D6 + D11):`e^{+jωt}` / `(∇²+k²)G = +δ` / `G = (j/4)H₀⁽²⁾` / `u_inc = exp(+jk·r)`;**装配式 `Z_mn = +(k/4)∫H₀⁽²⁾`;电流重建场 `E_z^sca = −(k/4)ΣJ_n∫H₀⁽²⁾`**。这一串符号是一条闭合链,改任一处其余必须同改。判据 F(远场与 Mie 同号)就是它的守卫。
2. **自耦合积分走 `self_coupling_integral`(统一入口,默认 `method="auto"`)**,不要自己去分段。主路径是 Struve 闭式(D10),O(1) 全域有效。
3. **近奇异积分(场点靠近但不在段上)用备用路径**:`analytic_part_integral` + `regular_part` 的 Gauss 求积。它**只在 `k*a <= 0.5` 保证 1e-12**(实测 ka=0.5 → 8.58e-13,ka=1 → 1.83e-11,ka=5 → 8.62e-8);ka 大时要提高 `order`(ka=5 需 ≥512)。
4. **不要对自耦合用任何"整段高斯求积"**:`direct_gauss` 实测误差停在 `3.8e-3 ~ 9.6e-3`,提高阶数无效。`naive_self_coupling` 最差 `6.06e+00`。
5. **凡给数值函数定参数,先扫参数空间找失效边界并写成断言**(D10 的教训);**凡给物理量写符号断言,先确认这个量在此离散化下代表什么**(D11 的教训,判据 A3 就是这么写错的)。

## 已落盘

- `proj/src/green.py` —— `EULER_GAMMA` / `SMALL_THRESHOLD=0.5` / `hankel0_2_small` / `hankel0_2` / `green_2d` / `log_singular_part`
- `proj/src/quad.py` —— **主路径** `closed_self_integral` + `hankel0_antiderivative`(Struve 闭式);**对照** `series_self_integral`;**备用** `analytic_part_integral` / `regular_part` / `split_self_integral`;入口 `self_coupling_integral(method=auto|closed|series|split)`;**错误做法** `direct_gauss` / `naive_self_coupling`
- `proj/src/mom.py` —— `circle_geometry` / `assemble_impedance` / `assemble_excitation` / `solve_dense` / `surface_current` / `scattered_field_on_target` / `total_field_residual` / `mie_surface_current` / `mie_scattered_field`
- `proj/tests/check_green.py` —— 判据 A(mpmath 60dps)+ B(scipy)+ 拼接缩放 + 余项门槛表 + 守卫测试
- `proj/tests/check_quad.py` —— 判据 A / A2(大 ka 守护)/ A3(双实现)/ B / B3 / C / **C2(防回归)** / D / E / F(备用路径断言)/ G(浮点边界)
- `proj/tests/check_mom.py` —— 判据 A(循环/对称/耦合衰减)/ B(残差 + 真值代回)/ C(电流 vs Mie)/ D(收敛阶)/ E(节点边界条件残差)/ **F(远场符号守护)** / G(ka 扫描)/ H(入射角)/ I(输入校验)
- `proj/tests/repro_review.py` —— 复核最小反例存档(用 mpmath 独立真值,非自比)
- `proj/{README,requirements,.gitignore,run_all.py,DECISIONS,STATE}.md`、`proj/experiments/registry.md`

## 1.3 最终实测数字

**判据(`check_mom.py`,EXIT=0 全通过;1.1 / 1.2 回归同时 EXIT=0)**

| 判据 | 内容 | 结果 |
|---|---|---|
| A1 | 循环结构 `max|Z[m,n] − Z[0,(n−m)%N]|` | **`2.116e-15`** |
| A2 | `max|Z − Zᵀ|` | **`2.315e-15`** |
| A4 | `|Z_00| / |Z_0,对径|` | `10.050` |
| B1 | 相对残差 `‖ZJ−V‖/‖V‖` | **`5.490e-16`** |
| B2 | Mie 真值代回 `‖Z·Jmie−V‖/‖V‖` | **`1.707e-04`** |
| C1 | ka=5, N=400 电流相对误差 vs Mie | **`1.9639e-04`**(tol 5e-4) |
| C2 | ka=1, N=200 电流相对误差 | **`8.7189e-05`** |
| C3 | 逐点最大误差 / `max|Jmie|` | `1.8415e-04` |
| D | 收敛:100→200→400→800 | `3.111e-03` → `7.830e-04` → `1.964e-04` → `4.917e-05` |
| D | 误差比 | **3.973 / 3.987 / 3.994 ⇒ O(h²)** |
| E1 | 节点 `|E_tot|max/|E_inc|max` | `1.652e-04` |
| E2 | 节点残差 / 电流误差 | `0.841` |
| F | `Re[E(from J)/E_mie]`,ρ = 10a / 40a / 150a | **`+0.999984` / `+0.999987` / `+0.999988`** |
| G | ka = 0.5 / 1 / 5 / 10 / 20 电流相对误差 | `4.774e-04` / `8.452e-04` / `7.830e-04` / `5.427e-04` / **`1.475e-03`** |
| H | θ_i = 0 / π4 / π2 / π | 全 `1.9639e-04`(与 θ_i 无关,符合圆对称) |
| I | 四项输入校验 | 全部抛 `ValueError` |

**判据 A3 已删除**(判据设计错误,非代码错):原设想 `Im(Z_mm) < 0` 对应辐射损耗,实测 k=1、a=5 时 `Im(Z_mm) = +9.1448e-02`。根因是 `Z_mm` 含自场奇异性贡献的自能项,**不是输入阻抗**。测试文件中留了注释记载,详见 D11。

**1.3 修入的两个真 bug**:① `E_z^sca` 相对 `∫J·G` 的负号丢失(远场与 Mie 反号,判据 F 抓住);② `mie_*` 级数截断 `ceil(ka)+12` 不够,改为 `ceil(ka + 4·ka^{1/3}) + 10`。

## 环境事实(实测)

- Python:`python`(3.12.10);numpy 2.5.1 / scipy 1.18.1 / matplotlib 3.11.1 / torch 2.13.0+cpu / **mpmath 1.3.0**
- 机器:Intel Core Ultra 9 290HX Plus,24 逻辑核,**16 GB RAM(实测 15.38 GB)**,无 GPU;实测复数 LU 有效算力约 39–63 GFLOPs
- **内存是唯一硬瓶颈**:安全算例上限 N ≤ 5000;勉强 N = 8000–10000(须关闭浏览器等)
- 沙箱在 workspace-write 下**禁止 pwsh 子进程出站联网**(curl 全部返回 `000`);机器本身有网,Google/HF/Scholar 需梯子
- ⚠️ 中文路径在子进程 stderr 中会被 GBK 破坏(Python 自身运行不受影响)
- 本机 PowerShell 为 5.1(`Get-Content` 无 `-AsByteStream`;`Get-CimInstance` 被沙箱拒绝)
- Git:仓库在 `proj\`,`user.name` / `user.email` 未配置(commit 需带 `-c user.name=... -c user.email=...`)

## 下一步

开工 **1.4 `rcs.py` + Mie 对比**(⭐ 头号里程碑:与 Mie 级数对比,N=320 相对误差 < 2%)。要点:
- 二维 TM 的远场(ρ → ∞):`E_z^sca(φ) ≈ (j/4)·√(2/(π kρ))·e^{−j(kρ−π/4)}·(由 J 求的散射幅度)`,或直接用 `scattered_field_on_target` 在远场求值
- 更稳的做法:2D 散射宽度 `σ_2D(φ) = lim_{ρ→∞} 2πρ |E_sca|²/|E_inc|²`,与 Mie 级数同量比较
- **1.3 的接口已备好**:`scattered_field_on_target(k, geom, J, pts)` 就是这个对比要用的重建函数
- 判据必须用独立真值(纪律 D8):Mie 级数天然满足;两条独立真值路径 `mie_surface_current` / `mie_scattered_field` 已在 `mom.py` 里
