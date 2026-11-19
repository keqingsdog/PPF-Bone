# 本次生产性能改动的泛化检查

范围：当前 `formal-source-20261029/blender_addon/ui/bone_overlay.py` 的 `_native_frame_publication`、`_consume_native_completion` 和 RP completion observer；`core/rigid_pose.py` 本轮姿态性能 helper。此为针对这些路径的源码检查，不是整个插件的安全审计，也不是对全部场景的运行验收。本次未运行 Blender、性能测试、哈希或全目录比较。

- 在上述两个生产文件中检索 `Skirt144`、`144`、`408`、`native-async-proof`、固定 `range(1, 31)` 和 `async`，无匹配。特别是 link 循环从当前场景每个 viewport 的实际 rows 遍历，不指定骨骼数、连接数、测试帧或路径。
- link 局部缓存使用完整 `(arm_name, session_uid, arm_uuid)`、完整 endpoint tuple 及 anchor 组成的键；不同 anchor 不共享世界点，selected/color 仍由每条 link 原样携带。三个缓存字典在每次 `_native_frame_publication` 调用中新建，调用后不保留 RNA，不跨帧或跨 completion 保存规划结果。
- 消费者在有效场景、generation、epoch、frame/subframe 与 completion serial 的 token 下去重同一次发布；新的 completion/epoch 会重新构造。RP observer 在成功提交姿态后通知消费者，并不使用私有异步 worker。私有 async proof 仅在工作目录测试脚本中，本次两个生产路径没有接入该原型。
- RP 优化按实际 target/binding 数量执行；inherit 模式相同才复用当前 Bone 进行规划，模式不同仍创建和清理临时骨架。连接记录缓存的作用域为一次 `_apply`，退出时弹出。批量刚体校验是逐批输入的通用快速路径，非 list/tuple、异常和阈值附近回退原逐项校验，并不使用固定数量或特定帧。

仍保留的合法约束：PDRD 刚体变换要求有限值、底行容差 1e-8、正交和行列式误差门 5e-4；批量路径距门 1e-10 内回逐项流程。帧需匹配 scene UUID、输入代次、帧号、接受时间和来源，completion仅在整数帧且当前场景帧等于起始帧加缓存帧时发布。显示还要求有效所有权、代理/骨骼身份、session、enabled 和有限可逆矩阵。`PDRD` 分支与 FEM/动态 PC2 是不同显示路径，不应将本项 PDRD 测试外推为所有 FEM 行为。

测试样例的 144 代理、408 连接、1..30 帧只是实测负载范围，不是代码门槛。不同数量、模型、版本和用户实际场景仍需相应测试，不能从本次静态检查宣称全场景兼容。
