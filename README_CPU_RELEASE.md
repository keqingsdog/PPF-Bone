# PPF 1.0.33 Windows CPU 交付说明

本次提供 Blender 5.1 插件、Windows CPU 完整运行包及完整源码三个独立 ZIP。请将 CPU 运行包解压到新目录，在插件设置选择该运行包根目录及 CPU 后端；不要覆盖正在运行的旧目录。插件 ZIP 可从 Blender 扩展“从磁盘安装”。运行包 start.bat 是 Jupyter 入口。

CPU 验收包括连续扭转两圈、保存续算、真实旧版 V1 显式升级、启动失败后的同 Blender 直接重试、独立新进程普通 V2 续算。旧 V1 升级从保存姿态开始累计，既有圈数无法找回；升级后 prior_turns_known=false 及迁移时间保存在 V2 中。旧缓存公开升级要求原 CPU 后端及原组件路径，迁移到新运行包目录后不能据此推断旧封存工程可恢复。请先用独立工程验证。

CUDA 本轮暂缓，本运行包不附 CUDA 求解器或 backend DLL；Metal、ROCm 未验收。完整源码保留 CPU/CUDA/Metal/ROCm 后端和平台构建脚本，详见 BUILDING_BONES.md 与 docs/CONTINUOUS_TWIST.md。源码不携带用户私有场景、测试输出或构建缓存。

三包按本轮同一源码构建；运行包提供 matching solver、server 和 Python binding。此说明不代替 CUDA、真实根/FEM 故障或人工视觉验收。
