# 骨骼分支源码构建

本页构建命令不等于功能验收。连续扭转、缓存迁移和历史测试边界见 [候选说明](docs/CONTINUOUS_TWIST.md)。公开源码不附带插件依赖 wheel；请按插件依赖准备流程获取或自行构建适用于目标平台的 wheel。

本文针对本分支新增的骨骼/关节功能。上游各平台的完整运行环境打包说明保留在 `build-*-native/README.md`。无需下载历史开发机的目标文件，也不应从旧运行包复制自有 `.obj` / `.lib` 以使构建通过。

## 源码归档和 Git 执行权限

可在仓库根目录用 `python tools/package_public_source.py --output /path/outside/source.zip` 创建源码归档；输出必须位于源码树外且尚不存在。归档将 `.sh` 文件写为 Unix 100755，普通文件写为 100644。Linux/macOS 可用支持 Unix 文件权限的解压工具解压，再用 `git add` 提交。

Git 工作树只打包索引中的跟踪路径；新公开文件需要先明确加入 Git 索引，并建议从审阅过的干净提交构建。源码 ZIP 解压后缺少 `.git` 时只能按归档附带的 `.ppf-public-source-files.json` 清单重打，不扫描工作目录里的新文件。旧 ZIP 若没有清单，请先把可公开的源文件放到干净 Git 工作树再打包，不要从含缓存或凭据的目录自动生成清单。

本机配置、常见凭据文件、编译产物和三平台构建缓存不会进入公开包；预期来源路径若被换成符号链接或目录联接，打包会失败，不读取其目标。归档清单管控的是路径，提交或公布之前仍应人工核对已跟踪文本中的秘密及许可证。

Windows 文件系统无法可靠保存 Unix 执行位；从 Windows 解压后如需创建 Git 提交，先 `git add`，再在 PowerShell 中显式标记已跟踪的 shell 文件：

```powershell
git ls-files '*.sh' | ForEach-Object { git update-index --chmod=+x -- "$_" }
```

这一步修改 Git 索引，不改变脚本内容。`tests/test_public_product_paths.py` 是公开路径的定向回归，`.gitignore` 显式允许跟踪它；其他根目录 `tests/` 项仍受原有忽略规则约束。

## 依赖

- Rust/Cargo 和仓库的 `Cargo.lock`。
- Python 及前端依赖；Python 绑定必须与实际运行前端的解释器兼容。按 `warmup.py` / `pyproject.toml` 准备依赖。
- CUDA 12.8 与其支持的本机 C++ 工具链。Windows 使用 Visual Studio x64 开发者命令环境；Linux 使用支持的 GCC/Clang、make 及平台构建工具。
- 正式 CUDA 构建还需要可运行、能读取目标产物的 NVIDIA `cuobjdump` 与 `nvdisasm`，并加入 PATH。当前目标表含 sm61，请使用 CUDA 12.8 的二进制工具；本机实际 CUDA 13 工具拒绝 sm61 selector，不能拿其 whole-image 回退当七架构全扫。仅有 `nvcc` 的精简 SDK 不足以完成原 SASS 验收。官方独立组件可从 [12.8.0 redist 索引](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.8.0.json) 获取，无需重下整个 SDK。
- 其余前端和平台构建依赖按各平台 warmup 配方获取。当前 neutral/host 实现不需要 Eigen 头文件。第三方工具包可以另行安装，源码包不包含整套 Python/CUDA 环境。

首次联网需要下载 Cargo 依赖和平台依赖；`--offline` 仅适用于依赖已经备齐的机器。它不会补齐缺失源码或替代工具链。

Python 可选约束、当前实际已测组合和未验证边界见 [依赖约束说明](docs/build_dependency_scope.md)。默认依赖安装不变，不把 Windows 版本快照作为 Linux/macOS 的锁文件。

## Windows CPU

CPU 构建不依赖 CUDA。在仓库根目录、Visual Studio x64 开发者命令环境运行，Python 应为最终运行前端所用的解释器：

```powershell
python tools/build_source.py --phase all --backend cpu --python "C:\path\to\python.exe" --jobs 3 --target-dir target/cpu
```

该命令生成 CPU 求解器、服务和 Python 扩展，分别位于 `target/cpu/release`。CPU 共享 ABI 身份用于普通前端；骨骼恢复力属于单独的可选能力，不应让普通 CPU 场景依赖 CUDA 产品路由。

没有 CUDA SDK、也不在 macOS Metal 路线时，`--backend auto` 会明确报错并提示显式选择 `--backend cpu`，沿用上游不自动降级 CPU 的选择语义。`--phase backend --backend cpu` 实际下发 Cargo CPU 构建；显式选择 CUDA 而未找到 SDK 会直接报错，不改选 CPU，也不会只完成身份检查就返回成功。

普通 CPU 前端有实际运行证据。2026-10-26新增了CPU跨链弹簧的14个隔离案例、一次真实checkpoint恢复，以及32骨裙子4秒运行；这仍不是全部CPU骨骼扩展的验收。FEM根控制与事件等可选扩展以实际选中产物的能力探针为准，不因本轮PDRD跨链案例通过而推断支持。完整保留各后端源码不等于后加扩展在所有后端均已实现或验收。

## Windows CUDA

在仓库根目录、Visual Studio x64 开发者命令环境运行。以下路径是用户填写的示例；不要求原开发机盘符。

```powershell
python tools/build_source.py --phase all --cuda-root "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8" --python "C:\path\to\python.exe" --jobs 3 --target-dir target/cuda
```

该入口先使用树内输入检查生成身份，再从 neutral kernel 生成 CUDA、编译后端和 Rust workspace。首次必须从没有自有构建产物的目录开始；后续正常增量构建可复用本次源码产生的对象。

修改 kernel 表、布局或生成器等身份输入后，先运行 `python coordination/windows-resume-20261004/restoring-curve/native-scalar/generate_identity.py` 重新生成同源身份，再构建。不要手改摘要常量或删除 `--check` 来掩盖输入不匹配。

`--phase backend`、`--phase rust` 和 `--phase services` 用于定位构建问题。单独服务编译成功不代表求解器或 CUDA 后端编译成功。

正式 CUDA 的 `all` / `rust` 路径会在下发构建前执行两种反汇编工具的 `--version`，缺失或不能运行时立即停止，并拒绝 `PPF_ALLOW_FP64`。随后通过 `PPF_FORMAL_CUDA_GUARD=1` 要求实际逐架构审计；工具枚举、架构选择、进程或零指令等无法完成审计的路径都会失败。Cargo 监视该模式，避免复用旧的跳过结果。Windows CUDA 链接分支也调用同一套原架构和 FP64 守卫。原指令规则和空许可表保留。

`--phase backend` 是开发用设备编译阶段，不代表最终原守卫通过；`--print-commands` 也不执行检查。非正式开发模式保留原回退行为，不能把缺工具 warning 或工具不支持部分架构时的 whole-image 回退当成完整验收。修复进度以 [验证记录](VALIDATION_BONES.md) 顶部为准。

### 显卡目标配置

`crates/ppf-cts-compute/cuda/cuda_arch.txt` 是 CUDA 编译目标的统一配置，后端构建和 Rust 的设备门都读取它。发布源码保留上游的目标列表 `61 / 75 / 86 / 89 / 90 / 100 / 120`。

此前内部正式构建的 Windows CUDA 后端从源码 LTO IR 链接出上述七个原生 cubin，并用 cuobjdump 确认存在；当时 Rust CUDA workspace 按完整目标表构建成功。这是历史内部构建证据，不代表本次公开源码的 CUDA 构建结果。该次开发机为 sm89，其他目标硬件尚未逐一实跑。不要为了复现开发机运行而删减发布目标表。

`--jobs N` 限制 Cargo 和独立设备构建的外层并行数。Cargo 内部调用的 make 固定为一个 worker，避免嵌套并行倍增；直接 Cargo 构建也默认使用一个 make worker，可用正整数 `PPF_DEVICE_BUILD_JOBS` 显式调整。Windows CUDA 每条 nvcc 编译及设备链接使用一个线程，外层编译队列遵守 N。Linux CUDA Makefile 的设备链接也使用 `-t 1`；先前 `-t 0` 沿用官方 9/27 默认，现在为履行本入口的并发预算承诺而限制线程，并不表示原版构建必然失败。该调度逻辑仍需在目标平台按实际工具链验证；公开二进制须从公开源码重新构建，不能依赖开发机旧产物。

## Linux CUDA

本次开发机没有 Linux 运行环境。接手方需要实际执行 Linux CUDA 编译，并验证新生成的求解器、后端库、Python 绑定及正常场景；Windows 编译通过不构成 Linux 验收。

在准备好 CUDA 12.8、Rust、make、支持的本机 C++ 编译器和 Python 依赖后，从仓库根运行：

```bash
python3 tools/build_source.py --phase all --cuda-root /usr/local/cuda --python /path/to/python3 --jobs 3 --target-dir target/linux-cuda
```

此入口仅选择 CUDA，由 Cargo 构建脚本从源码生成和编译后端，不强制同时构建 CPU/ROCm。`--print-commands` 可以查看平台命令计划，但不执行编译，不能作为 Linux 构建通过的证据。

不要把上游默认的多后端打包命令当作新增骨骼功能的 CPU/ROCm 验证，也不要复用冻结 Windows 配方中的目标文件。

## 组装 Windows CPU + CUDA 运行包

完成两个后端的源码构建后，以第三方 Python、MinGit 和 CUDA 运行库作为输入；ffmpeg 是可选视频组件，默认不纳入运行包；如确需分发，可显式加 `--include-ffmpeg`，并由分发者提供相应许可和源码材料。以下目录都是使用者自行填写的路径，输出目录必须尚不存在；脚本不会调用会清空输入的 `bundle.bat`。

```powershell
python tools/package_windows_runtime.py --source-root . --third-party-runtime "C:\PPF\official-runtime" --mingit "C:\PPF\MinGit" --cuda-library crates/ppf-cts-compute/cuda/build/lib/libsimbackend_cuda.dll --cuda-bin "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin" --msvc-redist "C:\path\to\x64\Microsoft.VC143.CRT" --output "C:\PPF\BoneRuntime"
```

该脚本复制本次源码生成的 CPU/CUDA 求解器、服务和 Python 扩展，保留 `python/`、`bin/`、`target/{cpu,cuda}/release/`、`frontend/`、`examples/` 等官方结构，并实跑两个求解器的 `--backend`。普通前端仍可选择 CPU/CUDA。源码扩展路径通过自动生成的 `LOCAL_SOURCE_BUILD.json` v2 配置使用 CUDA；配置中的路径均相对运行包根目录，不包含原开发机路径或探针快照。

运行包移动后，把配套插件的求解器目录选为新的运行包根目录，并重新传输场景。以 `start.bat` 打开普通示例；用户自己的 Python 脚本需使用包内 Python，并把包根目录加入 Python 模块路径、`bin` 加入 DLL 搜索路径。

## 启动与可选旧产品配置

常规使用时，在 Blender 插件的求解器路径中选择完整运行包目录，再连接服务；无需设置旧产品变量。源码配置的运行包使用其根目录中的相对路径配置。

仅在显式使用历史固定产品模式时，设置 `PPF_CTS_PRODUCT_ROOT` 指向该产品目录。该模式中 CUDA bin 的优先级为 `PPF_CTS_PRODUCT_CUDA_BIN`，其次产品配置的 `cuda_runtime_bin`；Python 若设置 `PPF_CTS_PRODUCT_PYTHON` 则使用指定可执行文件，否则依次查找产品根下 `runtime/python/python.exe` 与 `python/python.exe`。示例路径应替换成自己的实际路径：

```powershell
$env:PPF_CTS_PRODUCT_ROOT = "C:\path\to\legacy-product"
$env:PPF_CTS_PRODUCT_CUDA_BIN = "C:\path\to\cuda\bin"
$env:PPF_CTS_PRODUCT_PYTHON = "C:\path\to\python.exe"
```

公开 Windows 运行资产不附可选的 GPL `ffmpeg.exe`；视频功能可单独安装和配置 FFmpeg，或依既有脚本取得。解算所需组件不依赖这一可选视频程序。

公开源码树的 Windows CPU 冷构建及 25 帧普通 FEM 测试已退出 0。CUDA 源码保留七种目标架构；公开 Windows 运行资产另行发布，其实际结果以该资产的发行说明为准。

## 相对路径本地配置（v2）与历史配置

若已自行组装实际运行目录（含源码前端、配套 Python、后端二进制和运行库），可在目录根显式生成相对路径配置。Windows CPU 示例：

```powershell
python tools/configure_source_product.py --relative-root . --backend cpu --python-relative python/python.exe --bin-relative bin --build-relative target/cpu/release
```

Linux CPU 的同类目录可以使用 `--python-relative python/bin/python3`，前提是用户已准备真实的本平台产物和自包含 Python 运行环境：解释器入口及其链接最终目标都在组装目录内，标准库、site-packages、扩展模块及其本机动态依赖也须在目标系统可用。普通 `python -m venv` 常将入口链接到目录外系统解释器，会被闭包检查拒绝；`venv --copies` 也不会自动保证标准库、动态依赖齐全或搬迁后可用。保留所配置的入口路径启动前端，以保留其环境语义；搬迁后须在新位置检查真实解释器、依赖导入、后端能力及实际运行，不能仅凭链接通过就宣称可移植。CUDA、Metal、ROCm 通过同一命令的 `--backend` 显式选择；后端库不在构建目录时另传 `--backend-library-relative`。该配置不是构建或打包命令，不会下载运行库，也不会制造扩展能力声明。工具执行既有路径闭包检查和后端可执行文件探测；实际扩展能力仍在 Build 时检查。已有配置不会覆盖，先保留原配置并选择新的组装目录。

源码内历史固定 profile 仅在显式选择相应旧产品时用于兼容。请求骨骼等扩展但没有本地配置或显式产品选择时会清楚拒绝，不会自动读取旧开发机路径；普通场景不激活扩展合同，保留官方路径。新机器不要通过复制旧固定 profile、修改盘符或手写能力字段来建立源码配置。历史产品及源码配置的具体边界见[源码配置说明](docs/SOURCE_PROFILE_BOUNDARIES_20261030.md)。

### 旧版固定产品配置工具（v1）

下述工具仅保留用于旧版 v1 固定产品配置，不适用于上节的可移动 v2 源码运行包。新运行包应使用打包脚本自动生成的相对路径 v2 配置。

本轮新增 `tools/configure_source_product.py` 的本地源码构建配置入口。它要求显式选择源目录、实际编译输出、产品目录、Python、CUDA 运行库目录和后端库；对真实二进制执行功能探针后才写入可用配置。输出是本机配置，不能提交 Git。

Windows 示例：

```powershell
python tools/configure_source_product.py --source-root . --build-dir target/cuda/release --product-root "C:\PPF\LocalBoneProduct" --python "C:\path\to\python.exe" --cuda-bin "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin" --backend-library crates/ppf-cts-compute/cuda/build/lib/libsimbackend_cuda.dll
```

Linux 的后端名为 `libppfbe_cuda.so`，求解器没有 `.exe` 后缀；Python、运行库和路径分隔符也应采用目标平台的实际值。不能只重命名 Windows 文件。

配置成功后，在配套插件连接面板把求解器路径选为这里的 `--source-root` 目录，继续正常连接、传输、构建和运行。工具生成的本机配置不要提交 Git。移动源码目录或重新编译产物后，需要按实际新路径、输出和新的产品目录重新配置，不能继续冒用旧配对。

## 准备插件安装包

从仓库根目录按目标 Blender 内嵌 Python ABI 打包扩展。例如：

```powershell
python blender_addon/wheels/build_package.py --abi cp311 --blender "C:\path\to\blender.exe" --output ppf-blender-cp311.zip
python blender_addon/wheels/build_package.py --abi cp313 --blender "C:\path\to\blender.exe" --output ppf-blender-cp313.zip
```

构建入口在隔离目录中为所选 ABI 准备对应的五个平台官方 cbor2 6.0.1 wheel 和扩展 manifest；公开源码本身不附 wheel。Blender 4.2 使用 cp311，5.1 使用 cp313；其他版本以目标 Blender 的内嵌 Python ABI 为准，不支持的 ABI 会拒绝打包。安装脚本须通过 `PPF_BLENDER_BIN` 指定目标 Blender，以检测其 ABI。构建和下载需要网络与对应平台依赖；具体用法以 [wheel 工具说明](blender_addon/wheels/README.md) 为准。本机 Windows 对 Blender 4.2/cp311 和 5.1/cp313 做过隔离安装、启用与 cbor2 往返测试；其他版本和平台未因此自动获得验证。GitHub 扩展工作流为手动触发的无 secrets 构建并上传两种 ABI artifact，尚未在 GitHub 实际运行。

## 验证状态

2026-10-26跨链弹簧与静态网格接触修正：CPU增量release构建自然退出0（601.531秒）；正式CUDA全阶段自然退出0（2104.657秒），原七架构守卫扫描5,233,686条指令，无放宽许可或删减目标。该构建使用本树原身份生成器，新种类语义需要同时更新求解器、后端库和Python绑定，不能只替换EXE。上述是构建证据；实际GPU功能与交付状态另见[验证记录](VALIDATION_BONES.md)。

2026-10-25恢复力精度修正后，14个受影响CUDA编译单元和七架构链接通过；原规则逐七架构共4,858,272条指令检查退出0，无回退。配套CPU05、CUDA Rust05均自然构建0；完整运行包两后端新进程启动和各自绑定加载0。正式入口的原守卫真实运行，不只做工具预检。已有服务需正常重启才载新文件。详见[本轮数值修正](docs/NATIVE_RESTORING_F32_20261025.md)；该增量构建不改称又做了一次全冷构建。

恢复后的 117 个 neutral 模块、366 个入口已接回代码生成链。Windows sm89 从空后端输出和空 Cargo target 构建实际退出0，1254.531秒；后续只补 Python 返回字段并定向重编，42.735秒。正常插件路径已实际构建并运行1秒，自动31帧缓存及同次回放/Clear通过分段检查。原观察器超时的整轮外层退出1保留，不能写成整轮全0。详见 [验证记录](VALIDATION_BONES.md)。Linux、其他显卡及新增骨骼的其他后端没有本轮实机验证。

Windows CPython 3.13 的一次隔离 CPU 构建、普通场景和直接跨链求解，以及独立第二 venv 的依赖约束重建范围，见 [依赖约束说明](docs/build_dependency_scope.md)。短 target 的成功不排除较深路径的 MSVC include 问题；Linux/macOS 与其他后端仍须各自验证。
