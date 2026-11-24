# 骨骼求解与动画模式

Windows CPU 与 CUDA 使用同一个源码根。构建入口和依赖见 [构建说明](../BUILDING_BONES.md)。Windows 启动器按实际后端启用下表的优化。完整环境变量名及取值见 [启动配置](../frontend/_bone_trial_options.py) 中的 `candidate_switches`。

## 严格模式

下表的 `driver/` 相对于 `crates/ppf-cts-solver/src/`。各项优化根据当步的矩阵、约束和接触数据决定是否适用。

| 后端 | 已启用的方法 | 实现入口 |
| --- | --- | --- |
| CPU、CUDA | 对合格的纯刚体 FP32 系统，用每体六自由度的完整算子和紧凑块 Cholesky 求解。候选解和精化使用当轮原生算子检查分组真实残差。预算、分解或残差检查失败时继续原生求解。混合 FEM 保留自由网格行。 | `driver/pcg.rs`、`driver/full_body_native.rs`、`driver/compact_full_body_factor.rs` |
| CPU、CUDA | 捕获当前矩阵时复用顶点体索引和旋转坐标，按原有次序累加至连续 6×6 块。关节 Hessian 仅导出实际宽度内的项。 | `driver/body_block_projection.rs`、`driver/body_block_compact.rs`、`driver/compact_joint_export.rs` |
| CUDA | 纯刚体直解按实际接触准入。没有装配接触时，符合约束条件且每个球铰连通分量均有运动根锚定的自由飞行也可尝试。 | `driver/pcg.rs`、`driver/anchored_flight.rs` |
| CUDA | 对符合条件的设备常驻 PCG 和经典 PCG 乘积使用 CUDA Graph 重放，逐轮执行停止和错误检查。运动根达到原结束条件且接受时钟精确保留根事件端点时直接进入最终修正。 | `driver/pcg.rs`、`driver/step.rs` |
| CPU | 用查询盒和刚体叶盒剔除无贡献查询。纯刚体的边边、点面和点边查询可先并行探测。骨布混合时对自由壳使用保守包围盒，保留双向接触。最终相交先并行探测，有候选时仍走原有序记录路径。 | `driver/rigid_query_cull.rs`、`driver/contact.rs`、`driver/intersection.rs` |
| CPU | 邻接证明所有边都由合法邻面覆盖时省去独立点边 CCD。存在裸边、邻接不全时执行原查询。第二次网格扫掠在完整证明与接受端点一致时复用。CPU 启动值为 `PPF_B31_CERTIFICATE_MODE=off`、`PPF_B32_CERTIFICATE_MODE=reuse`。CUDA 沿用其两处已有的证明复用。 | `driver/covered_point_edge.rs`、`driver/collider.rs`、`driver/step.rs` |
| CUDA | 动态面、边和点碰撞树的兼容同层 refit 合并提交。点面 CCD 在严格内点胜出时复用已算分离向量，正规浮点数走等价的最低位运算，特殊值沿用原算法。该算术仅在三个 CUDA 点面扫掠调用处实例化。 | `driver/contact.rs`、`kernels/contact/aabb_refit_siblings.kernel.cpp`、`kernels/contact/ccd_sweep.kernel.cpp`、同目录的 `accd.hpp` 与 `distance.hpp` |
| CPU、CUDA | 对静止或单段仿射平移的硬 Pin，按 CCD 接受的前缀重采样约束，核对姿态、时长与场景时钟。超出适用域使用原有缩步和回滚。 | `backend.rs`、`driver/step.rs`、`driver/affine_pin_prefix_test.rs` |

## Blender 插件开关

配套的 Blender 1.0.67 插件在求解设置中提供“动画模式”，默认关闭。连接 Windows CPU 或 CUDA 解释器并启动服务后，查询响应中的 `solver_mode_v1: true` 使开关可用。Transfer 在 `Param` 消息顶层传入字符串 `solver_mode`。关闭时为 `strict`，开启时为 `animation`。旧客户端缺省为 `strict`，其他值在构建会话时被拒绝。

动画模式向原生子进程传入 `PPF_ANIMATION_BOUNDED_PROGRESS=1`。在原本允许结束、当前有效进度仍偏低时，最多再执行两轮普通 Newton 迭代，争取本次接受更大的步长。实现位于 `driver/bounded_animation_progress.rs` 和 `driver/step.rs`。CG 容差、CCD 与最终时钟检查照常执行。接受的子步和轨迹可能改变，纯网格或混合场景的收益随接触和步进而变。

每次 Transfer 都把模式保存到本次会话的启动命令。对该项目再次查询，响应的 `solver_mode` 返回已封存值。普通 Windows 启动命令和本地 R14 启动器均按会话设置子进程环境，严格模式会清除继承的动画变量。续算沿用当前会话的模式。要让新的面板选择生效，重新 Transfer 并从新会话运行。两份工程可以在同一解释器上使用不同模式。

升级前的普通 G/D2 会话可从各自的启动命令识别模式。早期 R14 用解释器全局文件启动，部分旧会话没有可核实的模式，此时查询会省略 `solver_mode`。若日志明确记录该旧会话使用动画，而启动命令没有封存值，续算会提示 `LEGACY_ANIMATION_RESUME_REQUIRES_RETRANSFER`。要延续这种旧缓存，可用原解释器完成续算；用新解释器时重新传输并从头运行。

## 实机参考

同一 768 刚体高接触输入，CG 容差 0.01，模拟 10 秒并写满 241 帧。Windows RTX 4070 Ti SUPER 的 11.22、优化严格和动画整程时间分别为 292.845、135.649 和 127.819 秒。i5-13490F 的两次历史 11.22 测量为 784.400 和 1019.994 秒，优化严格和动画分别为 313.694 和 276.176 秒。密集接触后的链尾排列有可见差异。

独立 CUDA 点面 CCD 对照在同配方严格全程中将 157.327 秒降至 143.142 秒，节省 9.02%。这些数字来自配套 Windows CPU 和 SM89 CUDA 成品。不同后端、显卡和网格场景需按各自完整求解耗时评估。
