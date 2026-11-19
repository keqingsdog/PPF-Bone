# 1.0.26 完整烘焙缩放与恢复保护

本版在 1.0.25 上修改 `blender_addon/core/bone_bake.py`、`single_bake_restore.py`、`full_bake_restore.py`，插件版本为 1.0.26，原生 workspace 版本仍为 0.1.0。本轮没有修改或重建 native 后端。

完整烘焙对具有有效 PDRD 绑定、冻结作者尺度与可证明静态继承来源的骨骼，使用恢复后的作者三分量生成位置/旋转动画，不将坐标分解浮点微差写成缩放动画。未证明来源保持原逐帧路径。原 TRS 可表示性和 3e-5 烘焙姿态验收门保持不变；真实 FEM 动态尺度不进入该静态路径。

严格常值清理支持 legacy 和 Blender 5.1 layered Action，按当前 slot 的实际 channelbag 操作，仅删除本次生成且逐轴精确恒定、等于当前静态值的目标曲线。完整三分量保留，源 Action、其他 slot 与未证明动画不被清理。Full/Single 恢复前检查用户静态 scale 修改，编辑后拒绝覆盖并保留记录；旧记录兼容。

5.1.2 私有测试已覆盖静态对象父级双骨、非 1 比例、多 slot、FULL/NONE 继承、Full/Keep/真实 prepare_upload、恢复、单轴编辑拒绝、保存重开和失败回滚；4.2 覆盖受影响 legacy 正反例。实际 1.0.26 ZIP 的 Extension Manager 安装及三个新进程功能场景均自然退出 0，实际模块 manifest 为 1.0.26。

输入使用合成合法 pose 缓存；后台 Full Operator 由脚本驱动其生产 modal 步骤。没有新增 native/FEM、自然 GUI、全部继承模式或跨平台验收。原 1.0.25 验证及其精度/平台限制继续按原范围解释，见 `RELEASE_NOTES_1_0_25.md` 和 `VALIDATION_BONES.md`。性能优化、关机及用户明确暂缓事项未重开。
