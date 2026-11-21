# PPF-Bone

上游项目 / Upstream · [st-tech/ppf-contact-solver](https://github.com/st-tech/ppf-contact-solver)

## 中文

PPF-Bone 是由 keqingsdog 维护的独立骨骼物理分支与发行版。它基于 9 月 27 日（0927）源码及提交上游的错误修复，相关基线实际相当于 10 月 1 日（1001）的状态。

在 Blender 中为各骨骼创建刚体或软体代理，用父子关节连接骨链，并用跨骨链弹簧构建横向或斜向连接。骨骼代理可以与支持的 PPF 材料发生接触。插件提供场景传输、模拟预览、获取结果和烘焙动画的工作流。更多功能和开发背景见[分支说明](README_BONE_FORK.md)与[构建说明](BUILDING_BONES.md)。

### 安装和使用

**当前下载：** Blender 插件 **1.0.65**（Blender 4.2–5.1）。下载页的源码标签仍为 `v1.0.36`。Windows x64 求解器是单独的运行包，不使用 Blender 插件的版本号。

1. 从 [最新发布](https://github.com/keqingsdog/PPF-Bone/releases/latest) 下载并解压 `PPF_Bone_Windows_x64.zip`。完整求解器包包含 Python、CPU 和 CUDA 构建。
2. 在 Blender 中通过“从磁盘安装”选择 `PPF_Bone_Blender_4.2-5.1.zip`，然后在插件设置中将求解器路径指向解压后的文件夹。统一安装包支持 Blender 4.2–5.1。插件版本以 ZIP 内的 `blender_manifest.toml` 为准，可能与 Release 的 Git 标签不同。
3. 配置骨骼代理和连接，传输场景并运行模拟，再获取结果和烘焙动画。

本分支发布的预编译求解器仅支持 Windows x64，目前没有 Linux 或 macOS 的预编译求解器。

## English

PPF-Bone is an independent bone-physics fork and distribution maintained by keqingsdog. It starts from the September 27 (0927) source and a bug fix submitted upstream, with the relevant baseline effectively equivalent to the October 1 (1001) state.

In Blender, create rigid or soft proxies for individual bones, connect bone chains with parent-child joints, and add transverse or diagonal connections using cross-chain springs. Bone proxies can contact supported PPF materials. The add-on provides a workflow for transferring scenes, previewing simulations, fetching results, and baking animation. See the [fork overview](README_BONE_FORK.md) and [build notes](BUILDING_BONES.md) for more details.

### Installation and use

**Current downloads:** Blender add-on **1.0.65** (Blender 4.2–5.1). The download page retains the `v1.0.36` source tag. The Windows x64 solver is a separate runtime package and does not use the Blender add-on version number.

1. Download and extract `PPF_Bone_Windows_x64.zip` from the [latest release](https://github.com/keqingsdog/PPF-Bone/releases/latest). The complete solver package includes Python and CPU and CUDA builds.
2. In Blender, use Install from Disk to select `PPF_Bone_Blender_4.2-5.1.zip`, then point the solver path in the add-on settings to the extracted folder. The unified installer supports Blender 4.2–5.1. Check `blender_manifest.toml` in the ZIP for the add-on version, which may differ from the release's Git tag.
3. Configure bone proxies and connections, transfer the scene and run the simulation, then fetch the results and bake animation.

Prebuilt solver releases from this fork support Windows x64 only. No prebuilt Linux or macOS solver is provided by this fork.

## License and attribution / 许可与署名

许可 / License: Apache-2.0 — [LICENSE](LICENSE). 原项目署名 / Original project credits: ZOZO, st-tech, Ryoichi Ando. 来源与署名 / Provenance and attribution: [NOTICE](NOTICE_BONE_FORK.md).
