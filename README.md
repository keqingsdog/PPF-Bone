# PPF-Bone

上游项目 / Upstream · [st-tech/ppf-contact-solver](https://github.com/st-tech/ppf-contact-solver)

## 中文

PPF-Bone 是由 keqingsdog 维护的独立骨骼物理分支与发行版。它基于 9 月 27 日（0927）源码及提交上游的错误修复，相关基线实际相当于 10 月 1 日（1001）的状态。

在 Blender 中为各骨骼创建刚体或软体代理，用父子关节连接骨链，并用跨骨链弹簧构建横向或斜向连接。骨骼代理可以与支持的 PPF 材料发生接触。插件提供场景传输、模拟预览、获取结果和烘焙动画的工作流。更多功能和开发背景见[分支说明](README_BONE_FORK.md)与[构建说明](BUILDING_BONES.md)。

### 安装和使用

**当前下载：** Blender 插件 **1.0.67**；Windows x64 求解器 **2026.11.22**。插件和求解器分别发版。

1. 下载并解压 [Windows x64 求解器 ZIP（CPU / CUDA）](https://github.com/keqingsdog/PPF-Bone/releases/download/solver-2026.11.22/PPF-Bone-Solver-Windows-x64-CPU-CUDA-20261122.zip)，其中包含便携 Python。
2. 下载 [Blender 插件 1.0.67 ZIP](https://github.com/keqingsdog/PPF-Bone/releases/download/v1.0.67/PPF-Bone-Blender-1.0.67.zip)。在 Blender 中使用“从磁盘安装”，然后在插件设置中将求解器路径指向上一步解压后的文件夹。
3. 配置骨骼代理和连接，传输场景并运行模拟，再获取结果和烘焙动画。

本分支发布的预编译求解器仅支持 Windows x64，目前没有 Linux 或 macOS 的预编译求解器。

## English

PPF-Bone is an independent bone-physics fork and distribution maintained by keqingsdog. It starts from the September 27 (0927) source and a bug fix submitted upstream, with the relevant baseline effectively equivalent to the October 1 (1001) state.

In Blender, create rigid or soft proxies for individual bones, connect bone chains with parent-child joints, and add transverse or diagonal connections using cross-chain springs. Bone proxies can contact supported PPF materials. The add-on provides a workflow for transferring scenes, previewing simulations, fetching results, and baking animation. See the [fork overview](README_BONE_FORK.md) and [build notes](BUILDING_BONES.md) for more details.

### Installation and use

**Current downloads:** Blender add-on **1.0.67**; Windows x64 solver **2026.11.22**. The add-on and solver have separate releases.

1. Download and extract the [Windows x64 solver ZIP (CPU / CUDA)](https://github.com/keqingsdog/PPF-Bone/releases/download/solver-2026.11.22/PPF-Bone-Solver-Windows-x64-CPU-CUDA-20261122.zip), which includes portable Python.
2. Download the [Blender add-on 1.0.67 ZIP](https://github.com/keqingsdog/PPF-Bone/releases/download/v1.0.67/PPF-Bone-Blender-1.0.67.zip). In Blender, choose Install from Disk, then point the solver path in the add-on settings to the folder extracted in step 1.
3. Configure bone proxies and connections, transfer the scene and run the simulation, then fetch the results and bake animation.

Prebuilt solver releases from this fork support Windows x64 only. No prebuilt Linux or macOS solver is provided by this fork.

## License and attribution / 许可与署名

许可 / License: Apache-2.0 — [LICENSE](LICENSE). 原项目署名 / Original project credits: ZOZO, st-tech, Ryoichi Ando. 来源与署名 / Provenance and attribution: [NOTICE](NOTICE_BONE_FORK.md).
