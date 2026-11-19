# 恢复力设备单精度修正（2026-10-25）

状态：本轮恢复力精度修正、Windows CPU/CUDA 源码构建和运行包文件交付已限定通过，最终 senior PASS。已启动的旧服务需要正常重启才加载新版本。

## 已确认的原因与修改

外部按原 `build.rs` 规则扫描七架构产物，报告 4,453,032 条指令、退出 101，以下七个入口包含未许可 FP64：`legacy_schwarz_export_entry`、`reduced_schwarz_joint_export_entry`、`restoring_stitch_apply_entry`、`restoring_history_capture_entry`、`joint_restoring_ball_apply_entry`、`joint_restoring_fem_apply_entry`、`joint_restoring_angular_apply_entry`。本机没有该次外部扫描的原始日志，以上为交接所报。

这些入口内联了共用恢复力头文件中的双精度标量计算。修正位于 `restoring_curve.hpp`、`restoring_context.hpp`、`restoring_observation.hpp` 和 `restoring_stitch.hpp`，没有修改 PCG 算法或扩大 FP64 许可表。

生产 CPU/CUDA/HIP/Metal 入口统一采用稳定 f32 路径：用局部指数缩放扩展中间计算范围，用有理化与原有 FMA 运算保留接近零时的差值，再按原语义窄化最终结果。能量允许舍入到零；数学非零的力、曲率和结构量下溢仍报告错误。域、分段、固定积分锚点、尾部和历史规则共用，ABI 布局保持原样。

旧双精度实现以显式 `_legacy` 名称保留作对照。它本身会在近零相减时丢失小项，因此不再把旧结果盲目当成真值；独立 Decimal200 参考计算同一个数学能量及其导数。消减错误曾将非零结果算成零的输入会得到纠正后的输出。`legacy Schwarz` 是引擎原有导出路径名称，与这些旧数值对照函数不是一回事。

MSL 拼接时仅排除不支持的双精度辅助与旧对照定义，共享生产 f32 保留。该分支已做实际预处理检查，没有 double/legacy 定义残留；这是静态分支验证，不是 Metal 编译或运行证明。

源码变化后，使用项目原生成器更新构建所必需的实现身份字段，补上 observation 实现输入。不删除身份检查，不手改摘要，不进行额外全树校验。

## 构建审计

首次构建缺少 `nvdisasm`，编译或运行成功不能代替 SASS 检查。另确认 Windows CUDA 分支原先没有像非 Windows 分支一样调用原架构和 FP64 守卫，本轮补接这两项。

`tools/build_source.py` 的正式 CUDA `all` / `rust` 路径在构建前要求 `cuobjdump`、`nvdisasm` 可执行，拒绝 FP64 绕过变量，并启用正式审计模式；Cargo 跟踪该模式以避免沿用旧跳过结果。正式模式不能以缺工具、无法反汇编等跳过路径返回成功。原指令规则与空 `KNOWN` 表保留；普通 CPU、其他后端和非正式开发路径保留其原选择语义。

## 当前实测

- 编译的主机 f32 实现：11 个 coordinate 和 14 个恢复力通道用例与独立高精度参考比较通过；能量、坐标、梯度、Hessian 分别计算误差尺度，另检查 min-subnormal、max-float 两侧及分段边界。
- 首个 sm89 CUDA 探针：原 FP64 守卫实际扫描 28,016 条指令，退出 0。它是小探针，不代表完整库。
- RTX 4070 Ti SUPER：14 个实际 CUDA 数值读回用例通过；启用曲线的 legacy/reduced Schwarz 导出均返回 fault 0、count 6，平动对角由基础值 2 变为曲线和阻尼合计 9，转动部分保持 0。此为真实小内核，不是整场景稳定性测试。
- 14 个受影响 CUDA 编译单元已全部编译成功并完成七架构链接；完整库原守卫结果见下节，配套 CUDA Rust05 自然构建退出0、587.719秒，正式构建日志也实际执行逐七架构原守卫，总计4,858,272条、无回退。
- CPU 本轮构建退出 0，Cargo 报告 9 分 14 秒；新建普通四面体盒沿当前前端 Build→Run 自然退出 0，到第 3 帧、0.05 秒。采样 0–3 共四帧、每帧 8 个有限顶点，平均下降约 0.012404 米；无固定产品 contract、未走 ABS04 路由。这不是 CPU 骨链物理证明。后续改动只接 Windows CUDA 审计，不改变已编译的 CPU 数值分支。
- 随后仅为 MSL 隔离旧对照定义并更新所必需的 source stamp，CPU 配套构建 05 再次自然退出 0，Cargo 报告 9 分 11 秒。CPU/CUDA 生产算术未因该平台条件隔离而改变；未重复同一普通 CPU 场景。

完整编译曾发现重复声明已有 FMA 函数，已改为直接复用原 `seam` 实现并重新通过主机/GPU测试；原失败记录保留。CPU 构建的首次启动脚本错误及接线修正期间的主动中断也不计为通过。

## 完整七架构 SASS 结果

CUDA 13 工具的 whole-image 回退虽退出 0、计 4,119,696 条，但不能读取 sm61，不能当作全七架构证据。换用官方 CUDA 12.8.55 的 `cuobjdump` 和 `nvdisasm` 后，对同一个新 DLL 运行原规则，正式模式禁止回退：**退出 0，总计 4,858,272 条指令，无未许可 FP64**。

| 架构 | 实际扫描指令数 |
| --- | ---: |
| sm61 | 738,576 |
| sm75 | 744,240 |
| sm86 | 722,488 |
| sm89 | 722,488 |
| sm90 | 721,152 |
| sm100 | 603,376 |
| sm120 | 605,952 |

两次扫描的差额 738,576 正好是 sm61，旧的回退记录保留。七架构 DLL 链接脚本退出 0，耗时 1,125.641 秒；上述正式全扫退出 0，耗时 20.75 秒。正式审计策略的四个实际 Cargo 用例也通过，包含“whole-image 可读但指定架构选择失败”必须拒绝；开发模式的原回退保留。

CUDA Rust04 曾因私有启动脚本同时包含 Path/PATH 而误选13版工具，正式门准确拒绝sm61；原失败保留。统一私有环境键后，05在启动阶段确认12.8.55工具，最终自然构建0。没有改生产精度规则来使04通过。

最后的MSL条件隔离只影响Metal文本，CPU/CUDA生产算术保持；实施身份生成器正常更新source stamp。该stamp由Rust compute模块消费，DLL身份头只含未改变的layout/table/mask，因此复用已验证的同一七架构DLL，CPU05与CUDA Rust05正常重编更新配套元数据。

## 本地运行包

`<本机运行包目录>` 已定向更新CPU/CUDA求解器、服务、绑定和后端库，未重拷整套Python/SDK。仅包内目录与Windows系统目录的PATH下，两后端`--backend`均0，各自包内`_ppf_cts_py.dll`真实导入均0；这不是新场景模拟。替换时旧空闲服务占用的映射原文件已保留到备份，未杀Blender或修改连接选择；正常重启本地服务后才切换该服务自身。配套插件为运行包根目录的v2安装ZIP。

本机原始记录：`plugin-external-handoff-20261024/fp64-sass12-final01/{RESULT.json,original-fp64-guard.log}`、`fp64-cuda-seven03/`。这些开发机日志不作为源码编译依赖，接手方使用本仓库测试与正式构建入口复现。

## 接手方复现入口

在 C++ 开发者环境运行：

```text
python -B crates/ppf-cts-solver/tests/restoring_f32/run.py
```

完成正常 CUDA kernelgen 后，在 CUDA/C++ 开发者环境运行，按实际显卡填写 `--arch`：

```text
python -B crates/ppf-cts-solver/tests/restoring_f32/run_device.py --cuda-root /path/to/cuda --arch 89 --out /path/to/new-test-output
```

完整库仍须通过正式 `tools/build_source.py --backend cuda --phase all` / `rust` 的原守卫；探针不能代替这一步。

## 验证边界

只有 Windows 与 sm89 有本轮实际运行证据；其他 CUDA 架构的 SASS 审计不等于对应硬件实跑。Linux、Metal、ROCm 未在本轮实跑。普通 CPU 场景与骨骼 CPU 扩展的验证范围分别报告。旧的曲线关闭场景 180 帧结果不拿来证明本次新数值路径。

原用户 Undo 原生访问异常仍未在私有场景精确复现；插件生命周期缓解与定向回归另见 `PLUGIN_AUTHORING_FOLLOW_FIX_20261025.md`，不能因本次数值或构建检查通过而宣称任意场景、任意操作均无问题。
