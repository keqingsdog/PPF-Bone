# PPF-Bone

上游项目 / Upstream · [st-tech/ppf-contact-solver](https://github.com/st-tech/ppf-contact-solver)

## 中文

PPF-Bone 是由 keqingsdog 维护的独立骨骼物理分支与发行版。它基于 9 月 27 日（0927）源码及提交上游的错误修复，相关基线实际相当于 10 月 1 日（1001）的状态。

在 Blender 中为各骨骼创建刚体或软体代理，用父子关节连接骨链，并用跨骨链弹簧构建横向或斜向连接。骨骼代理可以与支持的 PPF 材料发生接触。插件提供场景传输、模拟预览、获取结果和烘焙动画的工作流。更多功能和开发背景见[分支说明](README_BONE_FORK.md)与[构建说明](BUILDING_BONES.md)。

### 安装和使用

**当前下载：** Blender 插件 **1.0.68**；Windows x64 求解器 **2026.11.24**。插件和求解器分别发版。

1. 下载并解压 [Windows x64 求解器 ZIP（CPU / CUDA）](https://github.com/keqingsdog/PPF-Bone/releases/download/solver-2026.11.24/PPF-Bone-Windows-CPU-CUDA-2026.11.24.zip)，其中包含便携 Python。
2. 下载 [Blender 插件 1.0.68 ZIP](https://github.com/keqingsdog/PPF-Bone/releases/download/v1.0.68/PPF-Bone-Blender-1.0.68.zip)。在 Blender 中使用“从磁盘安装”，然后在插件设置中将求解器路径指向上一步解压后的文件夹。
3. 配置骨骼代理和连接，传输场景并运行模拟，再获取结果和烘焙动画。

本分支发布的预编译求解器仅支持 Windows x64，目前没有 Linux 或 macOS 的预编译求解器。

### 本次更新

- [Blender 插件 1.0.68](https://github.com/keqingsdog/PPF-Bone/releases/tag/v1.0.68)：新增动画模式开关、独立骨骼动作捕获和清除；骨骼、Static 和 Pin 可共用时间轴采样，传输可复用有效捕获。蒙皮权重拟合新增「厚度偏移」，可给拟合代理增厚。
- [Windows 求解器 2026.11.24](https://github.com/keqingsdog/PPF-Bone/releases/tag/solver-2026.11.24)：优化骨骼关节求解和碰撞检测。严格模式仍是默认；动画模式在下一次传输后生效，续算沿用会话模式。动画模式的轨迹和耗时可能随场景变化。

[下载 2026.11.24 求解器源码](https://github.com/keqingsdog/PPF-Bone/releases/download/solver-2026.11.24/PPF-Bone-solver-2026.11.24.zip)。Windows 运行包内的 CUDA 后端针对 SM89；其他架构可用 CPU，或从源码构建对应的 CUDA 后端。

## English

PPF-Bone is an independent bone-physics fork and distribution maintained by keqingsdog. It starts from the September 27 (0927) source and a bug fix submitted upstream, with the relevant baseline effectively equivalent to the October 1 (1001) state.

In Blender, create rigid or soft proxies for individual bones, connect bone chains with parent-child joints, and add transverse or diagonal connections using cross-chain springs. Bone proxies can contact supported PPF materials. The add-on provides a workflow for transferring scenes, previewing simulations, fetching results, and baking animation. See the [fork overview](README_BONE_FORK.md) and [build notes](BUILDING_BONES.md) for more details.

### Installation and use

**Current downloads:** Blender add-on **1.0.68** and Windows x64 solver **2026.11.24**. The add-on and solver have separate releases.

1. Download and extract the [Windows x64 solver ZIP (CPU / CUDA)](https://github.com/keqingsdog/PPF-Bone/releases/download/solver-2026.11.24/PPF-Bone-Windows-CPU-CUDA-2026.11.24.zip), which includes portable Python.
2. Download the [Blender add-on 1.0.68 ZIP](https://github.com/keqingsdog/PPF-Bone/releases/download/v1.0.68/PPF-Bone-Blender-1.0.68.zip). In Blender, choose Install from Disk, then point the solver path in the add-on settings to the folder extracted in step 1.
3. Configure bone proxies and connections, transfer the scene and run the simulation, then fetch the results and bake animation.

Prebuilt solver releases from this fork support Windows x64 only. No prebuilt Linux or macOS solver is provided by this fork.

### What's new

- [Blender add-on 1.0.68](https://github.com/keqingsdog/PPF-Bone/releases/tag/v1.0.68): Added an Animation Mode toggle and independent bone-motion capture and clearing. Bones, static meshes, and pins can share a timeline pass, and Transfer can reuse valid captures. Skin-weight fitting now includes a Thickness Offset to thicken fitted proxies.
- [Windows solver 2026.11.24](https://github.com/keqingsdog/PPF-Bone/releases/tag/solver-2026.11.24): Optimized bone-joint solving and collision detection. Strict remains the default. Animation Mode takes effect on the next transfer, and resume keeps the session's mode. Animation Mode can change trajectories and runtime depending on the scene.

[Download the 2026.11.24 solver source](https://github.com/keqingsdog/PPF-Bone/releases/download/solver-2026.11.24/PPF-Bone-solver-2026.11.24.zip). The bundled Windows CUDA backend targets SM89. On other architectures, use the CPU backend or build the corresponding CUDA backend from source.

## License and attribution / 许可与署名

许可 / License: Apache-2.0 — [LICENSE](LICENSE). 原项目署名 / Original project credits: ZOZO, st-tech, Ryoichi Ando. 来源与署名 / Provenance and attribution: [NOTICE](NOTICE_BONE_FORK.md).
