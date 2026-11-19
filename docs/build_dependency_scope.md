# 依赖约束与实际验证范围

本分支保留上游的平台准备路线；`warmup.py` 是既有 Linux x86_64 环境准备脚本，并非 Windows/macOS 通用安装器。Windows 依照 `build-win-native/README.md`，其他平台依照对应的 `build-*-native/README.md`。不要在已有工作环境中为核对版本重新运行完整 warmup。

## 可选约束，不改变默认安装

`python warmup.py --constraints /path/to/constraints.txt` 仅在用户明确传入时，向 Python 依赖及四面体工具的两次 pip install 添加 `--constraint`。路径先转为绝对路径并检查文件存在；pip 自升级、系统工具安装和后端选择沿用原行为。无参数仍走原依赖列表，既有 cbor2、pytetwild 固定版本仍有效。该选项不会增加或删除任何后端。

`requirements/windows-py311-tested.txt` 记录本机现有 Windows CPython 3.11 运行环境的直接运行依赖版本，供审阅和显式选用，不是所有传递依赖的锁文件，也不是干净机器构建证明。文件的环境标记仅约束 Windows Python 3.11，Linux/macOS 以及其他 Python 版本不会套用这些版本。其他平台应先按官方默认路线准备，再依据其实际测试维护自己的约束文件。

例如在已经准备好的 Windows Python 环境中，只约束所需依赖安装（按实际需求选择包，不把这条短命令当完整 warmup）：

```powershell
python -m pip install -c requirements/windows-py311-tested.txt numpy scipy cbor2 psutil pytetwild tetgen pyvista
```

约束文件不会安装未列在 install 命令中的包；包可用性及传递依赖仍由目标平台 pip 解析。不要使用 `--no-deps` 绕过正式安装依赖。2026-10-28曾对已安装直接包执行无下载／安装的受限dry-run（`--no-index --no-deps`），不是完整依赖解析。本次恢复材料仅核实指定包元数据和安装命令helper，没有重装运行环境或重跑该历史dry-run。

## 2026-10-28 已测和未测

| 项目 | 证据范围 |
| --- | --- |
| Windows x64 Python 3.11.9 | 当前完整运行包：实际 CPU 本地 server 的 Transfer/Build/Run/GetAll、12 帧骨链与软体球、完整/单帧烘焙恢复后重新获取均完成；核心包版本见可选约束文件。不是全平台全功能验收。 |
| Blender 5.1.2 / 内置 Python 3.13 | 插件端真实程序化烘焙、恢复、保存重开及上述网络周期；插件 wheel 与前端 Python 各自匹配，不能把前端约束强用于 Blender Python。 |
| Rust | Cargo 构建使用仓库 Cargo.lock 与 `--locked`；当前主机读到 rustc 1.96.1、cargo 1.96.1。此为当前工具观测，不证明现有运行包由这版编译，不宣称其他 stable 版本已测。 |
| CUDA/C++ | 既有构建证据使用 CUDA 12.8 和 Windows MSVC；详见 `VALIDATION_BONES.md`。本次 PATH 实际 nvcc 为 13.0.88，未用它重新构建，不将它列为通过的构建组合。正式源码构建仍依 `BUILDING_BONES.md` 显式选工具链。 |
| Linux/macOS、其他 Python/Rust/C++ 版本 | 保留源码与官方路线；本轮未运行干净机器编译和物理验证，不用 Windows 版本快照作替代。 |

仓库尚未提供经多平台实编译确认的统一最小/最大 Python 或 Rust 范围；不要把一个本机版本推断成支持区间。发布验证应记录目标平台真实编译器、Python、后端和运行结果，再扩大已测范围。

## 2026-10-30 Windows CPython 3.13 CPU 隔离复建记录

`requirements/windows-py313-cpu-tested.txt` 是一次 Windows x64、CPython 3.13.9 CPU 运行中实际解析的包版本约束，包含当次存在的安装工具；它不是跨平台锁文件、不固定 wheel 工件、不证明其他 Python 版本或系统依赖可复现。使用私有 venv，并按仓库 `warmup.py` 的 `python_packages()` 与 `tetra_packages()` 列表通过该 venv 的 pip 配合 `-c requirements/windows-py313-cpu-tested.txt` 安装；不要执行会设置用户全局环境的 warmup 主流程，也不要用 `--no-deps` 跳过解析。第二个独立 venv 按同一约束实际安装成功，另实际执行受约束的 `pip==25.2` 安装并验证版本，`pip check`通过，运行了以下新编译 CPU 产物。安装工具仅包括当次实际存在的项目。

同一 Windows 主机上以 VS 2022 x64、Rust/Cargo 1.96.1、新源码解压目录、新 Cargo 缓存和空 target、`--jobs 2 --backend cpu --phase all` 构建发布源码快照，release 构建自然退出0（19分03秒）。事后在同一 vcvars64 环境回读到 MSVC cl 19.44.35222、VC Tools 14.44.35207、Windows SDK 10.0.26100.0；这不是构建当时留存的独立工具版本日志。新目标中的 CPU solver 与 Python binding 执行普通前端场景，进程退出0、10个模拟步，后验检查初始第0帧至第10帧共11个输出均为64个有限顶点，首末最大位移约0.08951；另以真实前端 Scene.build 和作者序列化生成两个刚体的跨链输入，直接运行同一新 solver，模拟时长约0.205秒、11个有限帧，原生receipt记录kind 5及41次实际样本。跨链准备在公开 export 遇到 PRODUCT_SELECTION_REQUIRED 后转入直接 native 序列化，因此该例不证明公开 export 或 Blender Build/Transfer 准入；能力声明和编译成功本身均不替代上述实跑。

首次隔离构建使用较深的 target 路径，MSVC 报找不到实际已生成的 include，文件完整路径长262字符；该次失败保留。改为新的空短 target 后成功，不能只凭这组对照确定唯一原因，也不能声称所有长路径均可用。以上为同一主机和当时源码快照的证据；不等同于新机器、全部环境、Linux/macOS、CUDA或其他后端验证。后续源码变更仍需对应验证。

例如从源码根目录准备隔离环境（`<...>` 替换为自己的路径；其中 runpy 那条仅读取 warmup 的依赖列表，不执行其 main）：

```powershell
& '<CPython-3.13-exe>' -m venv '<private-venv>'
& '<private-venv>\Scripts\python.exe' -m pip install -c requirements/windows-py313-cpu-tested.txt 'pip==25.2'
& '<private-venv>\Scripts\python.exe' -c 'import runpy,subprocess,sys; w=runpy.run_path("warmup.py",run_name="ppf_dependency_list"); subprocess.run([sys.executable,"-m","pip","install","-c","requirements/windows-py313-cpu-tested.txt",*w["python_packages"](),*w["tetra_packages"]()],check=True)'
& '<private-venv>\Scripts\python.exe' -m pip check
```

在 VS x64 开发环境中设置隔离的 `CARGO_HOME`、`TEMP`/`TMP`、pip缓存和 `PYTHONNOUSERSITE=1`，然后使用新的空短 target：

```powershell
& '<private-venv>\Scripts\python.exe' tools/build_source.py --phase all --backend cpu --python '<private-venv>\Scripts\python.exe' --jobs 2 --target-dir '<new-short-target>'
```

运行普通前端前应将 `CARGO_TARGET_DIR` 设置为同一个新 target，使 Python binding 与求解器来自本次构建；先清除指向其他产品或旧输出的继承环境变量。源码本身的身份和能力检查不应关闭。


