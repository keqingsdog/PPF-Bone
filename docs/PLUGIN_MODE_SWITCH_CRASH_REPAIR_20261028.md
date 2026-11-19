# 2026-10-28 模式切换与质量定时器修复

本轮只改插件，不改引擎、不重新编译CPU/CUDA。圆锥限位和W8已发功能保留。

## 用户现场与证据

用户报告编辑模式移动骨骼后代理不跟随，返回姿态模式并撤销时Blender崩溃。最新原生日志时间18:18:55，Python栈为core/bone_mass.py:_key → _live → mass_timer；原生pyrna_struct_CreatePyObject读地址0x19，来自py_timer_execute。操作日志记录POSE/EDIT切换与编辑平移，但未记录到Undo完成。栈定位证明质量定时器读取RNA时崩溃，不能单凭此确定唯一失效指针或所有崩溃同因。

## 修改

- bone_mass.py：EDIT阶段不访问普通Bone属性组；Undo/Redo/Load之前暂停并清除索引，之后重新解析；模式及Armature身份变化后等待稳定，再恢复RNA读取和质量测量。相同门覆盖定时器及依赖图回调。
- bone_author_follow.py、ui/bone_overlay.py：EDIT预览、显示签名和弹簧端点名称改用纯值作者元数据及当前EditBone；不跨模式保存RNA引用，不在EDIT读取旧Bone PropertyGroup。安全Object/Pose阶段为旧记录补齐元数据。无模拟缓存时显示随编辑骨骼移动和长度变化，取消后恢复；已有模拟缓存或运行忙碌时保留原权威门。
- 禁用骨骼的显示元数据同样更新，避免进入EDIT后从旧快照重画已关闭的物理代理。

## 测试及范围

真实Blender5.1.2事件循环，定时器实际执行：144代理的已保存工程副本、GPU开启、80步EDIT平移/缩放/POSE切换/Undo/Redo，含中间没有timer间隔的快速连续切换。最终进程退出0，_key 8496次、EDIT阶段读取0、稳定后实际质量测量14次，样本mass_valid有效。不是仅同步调用函数的测试。

旧版相同隔离序列未重现原生AV，但记录EDIT阶段_key读取35712次；因此本轮证明移除日志所指危险读取并通过定向压力序列，不能宣称原生AV或所有Undo崩溃已完全根除。

独立编辑预览测试覆盖旧元数据、平移0.4、取消恢复，以及关闭骨骼不再绘制。已有缓存的编辑跟随仍受原缓存门限制，未静默修改或删除用户模拟缓存。

证据在开发机plugin-mode-switch-crash-20261028：user-181855.crash.txt保留原件；MASS_RESULT.md、combined-final.exit.txt、combined-final.stdout.log、edit-follow.log及禁用快照直接测试。后续测试使用自身临时目录，不写用户恢复位置。用户原始工程没有修改，安装不热重载当前Blender。

正常重启Blender后载入这三个更新模块。若已有模拟缓存，需要先使用插件恢复模拟/清除本地动画后再做作者编辑；不要把已有缓存时不跟随误认为本轮已全面覆盖。
