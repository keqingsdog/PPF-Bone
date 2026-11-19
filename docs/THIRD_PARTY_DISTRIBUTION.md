# 第三方依赖与分发边界

历史快照：2026-10-28。除下方 2026-10-30 定向观察外，所列版本和安装路径均为历史快照，不代表当前发行依赖或完整传递依赖。本文说明当前能直接观察到的依赖及发行准备事项，不是整包许可审计、法律意见或发布授权。本次只读源码、指定安装 ZIP 的 wheel 元数据和本地运行包元数据；没有安装、删除或替换依赖。当前源码与交接材料仍保持原有私有状态。

## 四种交付物要分别处理

|交付物|当前内容与入口|适用边界|
|---|---|---|
|PPF 及本分支自有源码|根 [LICENSE](../LICENSE)、各文件原版权/许可声明；基线为官方 2026-09-27|根 Apache-2.0 不会把第三方代码、wheel、解释器、数学库或工具重新许可为 Apache。保留原版权与许可；修改文件应标明修改，存在适用 NOTICE 时保留其要求。|
|由用户执行安装脚本下载的依赖|`warmup.py`、`build-win-native/warmup.bat`、插件 `ui/install_ops.py`|这是安装流程，不等同于源码包已包含这些二进制；但不能据此推定集成、链接、修改或运行服务没有许可责任。执行 pip 得到的具体版本及 native 内容仍需辨明。|
|Blender 插件安装 ZIP|插件 Python 源码和五个 cbor2 平台 wheel|这些 wheel 是实际再分发，不只是声明依赖。其原许可文件须随载荷保留；合并双 ABI 不会改变第三方许可。|
|完整 Python/求解器运行包|Python、site-packages、Rust/原生求解器及选中后端所需库，可能另含工具|适用 Python、各 Python 包及内嵌库、Rust/C++ 依赖、CUDA/ROCm 等各自条件；不能把整个运行包标成“纯 Apache”。本表不是完整传递依赖清单。|

Apache 的要求以 [Apache License 2.0 官方正文](https://www.apache.org/licenses/LICENSE-2.0) 为准。根 LICENSE 未作改写，也没有新建一个替代第三方原许可的总 LICENSE。

## 2026-10-30 候选 ZIP 与运行包定向观察

候选 `formal-candidate-20261030-01/addon.zip` 中五个 cbor2 6.0.1 wheel 的 METADATA 均声明 MIT，且各自包含原 LICENSE.txt。指定运行包中仅重新查看 tetgen 0.8.4、triangle 20250106、plyfile 1.1.5 的本地 metadata 与许可文件：tetgen 有 wrapper LICENSE 和 src/tetgen-license；triangle 有 wrapper LICENSE；plyfile metadata 含 GPL 文本，其 dist-info 未发现独立 LICENSE 文件。此观察不能证明二进制或全部传递依赖许可完备。

本轮也核对以下上游资料：[cbor2 6.0.1 PyPI](https://pypi.org/project/cbor2/6.0.1/) 声明 MIT；[tetgen 0.8.4 PyPI](https://pypi.org/project/tetgen/0.8.4/) 说明 wrapper 使用 libigl/tetgen 的 TetGen 1.6.0；[WIAS TetGen 许可说明](https://www.wias-berlin.de/software/tetgen/download2.jsp) 列出 1.6 的 AGPLv3 与另行付费授权；[Triangle 作者页面](https://www.cs.cmu.edu/~quake/triangle.html) 对未经许可销售或纳入商业产品设有限制，不能泛化成禁止所有商业使用。这些事实不构成对整体发行方式的法律结论。

## 2026-10-28 安装 ZIP 的历史随包观察

读取的是正式文件 `<运行包根目录>\PPF骨骼插件_源码运行包v2_20261024.zip`，没有比较其他 ZIP、计算摘要或全目录核对。在其 `wheels/` 中实际发现五个 wheel，均为 `cbor2 6.0.1`：

- `cbor2-6.0.1-cp311.cp313-cp311.cp313-macosx_11_0_arm64.whl`
- `cbor2-6.0.1-cp311.cp313-cp311.cp313-manylinux_2_28_aarch64.whl`
- `cbor2-6.0.1-cp311.cp313-cp311.cp313-manylinux_2_28_x86_64.whl`
- `cbor2-6.0.1-cp311.cp313-cp311.cp313-win_amd64.whl`
- `cbor2-6.0.1-cp311.cp313-cp311.cp313-win_arm64.whl`

每个实际 wheel 的 `cbor2-6.0.1.dist-info/METADATA` 均声明 `License-Expression: MIT`；每个均含 `cbor2-6.0.1.dist-info/licenses/LICENSE.txt`，署名 Alex Grönholm。官方 [cbor2 6.0.1 PyPI 页面](https://pypi.org/project/cbor2/6.0.1/) 同样声明 MIT，并说明其实现含 Rust。这里确认的是包层许可声明和许可文本确实存在；没有据此审完 native 扩展的全部传递组件。

五个 wheel 是同一个包的不同平台/ABI 载荷，不是五种不同库。`blender_manifest.toml` 的 Apache-2.0 标记描述插件项目，不应被解释为这五个 wheel 或所有运行依赖的总许可。`blender_addon/wheels/fetch.py` 与 `merge_abi.py` 是下载/合并入口；发行时不得丢弃 wheel 内的原许可证。

SSH 的 `paramiko` 和 Docker 的 Python SDK 在 `ui/install_ops.py` 中按需 pip 安装，本次该 ZIP 的 wheel 区没有这两种包；它们也未安装在本次检查的完整运行包 Python 环境中。[Paramiko 官方 PyPI](https://pypi.org/project/paramiko/) 标记 LGPL-2.1，[Docker SDK 官方 PyPI](https://pypi.org/project/docker/) 标记 Apache-2.0。这不包含其日后安装版本的全部传递加密库/HTTP 库审查，也不等于 Docker Desktop 本体的许可。

## TetGen：Python wrapper 与 C++ 内核必须分开

本地完整运行包实际安装 `tetgen 0.8.4`。它的 METADATA 标记 MIT，但同时保留了两份不同层次的文件：

- `python/Lib/site-packages/tetgen-0.8.4.dist-info/licenses/LICENSE`：Python wrapper 的 MIT 文本。
- `python/Lib/site-packages/tetgen-0.8.4.dist-info/licenses/src/tetgen-license`：TetGen 内核的 AGPL v3 或更高版本，以及 WIAS 另行商业许可途径。

这是本地实际文件，不是仅凭项目名推测。[tetgen 官方 PyPI](https://pypi.org/project/tetgen/0.8.4/) 明确区分 MIT wrapper 和 AGPL 内核；其当前说明指出使用 TetGen 1.6.0 的 C++ 来源。[WIAS 官方下载与许可说明](https://www.wias-berlin.de/software/tetgen/download2.jsp) 明确列出 AGPLv3，并提示组合软件/网络服务的源码义务和另行许可选项；对应内核的 [libigl/tetgen LICENSE](https://github.com/libigl/tetgen/blob/master/LICENSE) 提供完整文本。

因此，不能将该 wheel 或链接它的完整运行包仅按 MIT/Apache 处理。发行者需要结合实际调用/链接、修改、再分发和服务部署方式，确认适用的 AGPL 履行方案或是否持有覆盖该方式的另行授权。本文未判定这些条件已满足，也没有把“AGPL”简单解释为禁止商业使用。仅删除其许可文件、改包名或只写 wrapper 的 MIT 都不能解决这项。

## Triangle：wrapper 的 LGPL 不替代内核作者条件

本地完整运行包实际安装 `triangle 20250106`；其 METADATA 和 `triangle-20250106.dist-info/LICENSE` 标记/包含 LGPL-3.0 wrapper 许可。[官方 PyPI 项目页面](https://pypi.org/project/triangle/20250106/) 说明这是 Jonathan Richard Shewchuk 的 Triangle C API 包装；[wrapper 上游 LICENSE](https://github.com/drufat/triangle/blob/master/LICENSE) 为 LGPLv3。

但 [Triangle 作者的 CMU 官方页面](https://www.cs.cmu.edu/~quake/triangle.html) 另有内核分发条件：未经许可不得销售或纳入商业产品。该措辞不等于禁止所有商业场景中使用 Triangle，也不能因 wrapper 标记 LGPL 就忽略它。作者页面还明确区分另行公开的 robust predicates 与 Triangle 本体，不能把前者的 public domain 说法套到整个 Triangle。

发行者须针对所用内核版本、源码内原始说明、wrapper 与内核组合方式，以及是否销售/纳入商业产品确认条件。当前没有声明已取得额外许可。本次只观察到本地 dist-info 中的 wrapper LICENSE；没有将它当成内核版权说明已完整随包保留的证明。Windows ARM64 的源码构建路线也不豁免该责任。

## 直接 Python 依赖：2026-10-28 历史环境快照

下面版本来自指定完整运行包 `python/Lib/site-packages` 的包元数据，不是新的版本锁，也不是 PyPI“最新版本”。来源为各包同版本官方 PyPI 页面及其包内许可文本。`warmup.py` 中除 cbor2/pytetwild 等固定项外仍有浮动项；Windows 配方另有固定版本及 ARM64 差异。某包缺装不代表其功能被删除，本次不改变安装配方。

下表“许可”是包层声明/原文本的记录，**不覆盖 wheel 中额外编译的库或所有传递依赖**。

|直接依赖与官方来源|观察版本|包层声明/注意项|
|---|---|---|
|[numpy](https://pypi.org/project/numpy/2.4.4/)|2.4.4|实际表达式为 BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0；保留该版本完整许可。|
|[scipy](https://pypi.org/project/scipy/1.17.1/)|1.17.1|BSD 文本；其 wheel 的 BLAS/LAPACK/编译运行库不能只按 SciPy 项目名归类。|
|[Pillow](https://pypi.org/project/Pillow/12.3.0/)|12.3.0|MIT-CMU；图像库等 native 载荷另按实际文件。|
|[numba](https://pypi.org/project/numba/0.65.1/)|0.65.1|BSD；LLVM/llvmlite 等另核。|
|[plyfile](https://pypi.org/project/plyfile/1.1.5/)|1.1.5|GPLv3 或更高版本；是发行专项确认项，不应漏作普通 MIT 工具。|
|[requests](https://pypi.org/project/requests/2.34.2/)|2.34.2|Apache-2.0。|
|[trimesh](https://pypi.org/project/trimesh/4.12.2/)|4.12.2|MIT。|
|[PyWavefront](https://pypi.org/project/PyWavefront/1.3.3/)|1.3.3|BSD。|
|[matplotlib](https://pypi.org/project/matplotlib/3.11.2/)|3.11.2|Matplotlib 自有、PSF 风格的许可文本；不是根 Apache。|
|[tqdm](https://pypi.org/project/tqdm/4.70.1/)|4.70.1|MPL-2.0 AND MIT。|
|[pythreejs](https://pypi.org/project/pythreejs/2.4.2/)|2.4.2|BSD-3-Clause；前端资源另看所带 notices。|
|[ipywidgets](https://pypi.org/project/ipywidgets/8.1.9/)|8.1.9|BSD-3-Clause。|
|[fast-simplification](https://pypi.org/project/fast-simplification/0.1.13/)|0.1.13|MIT。|
|[tabulate](https://pypi.org/project/tabulate/0.10.0/)|0.10.0|MIT。|
|[triangle](https://pypi.org/project/triangle/20250106/)|20250106|LGPL-3.0 wrapper；内核作者条件见上节。|
|[cbor2](https://pypi.org/project/cbor2/6.0.1/)|6.0.1|MIT；插件还实际捆绑五个平台 wheel。|
|[psutil](https://pypi.org/project/psutil/7.2.2/)|7.2.2|BSD-3-Clause。|
|[pytetwild](https://pypi.org/project/pytetwild/0.4.2/)|0.4.2|MPL-2.0；官方页面同时将 wrapper 与 fTetWild 描述为 MPL-2.0，其他内嵌构建依赖仍需另核。|
|[tetgen](https://pypi.org/project/tetgen/0.8.4/)|0.8.4|MIT wrapper + AGPL 内核/另行许可，见上节。|
|[pyvista](https://pypi.org/project/pyvista/0.48.4/)|0.48.4|MIT；VTK 等传递依赖另核。|
|[certifi](https://pypi.org/project/certifi/2026.7.22/)|2026.7.22|MPL-2.0；Windows 配方显式安装。|
|[gdown](https://pypi.org/project/gdown/6.4.0/)|6.4.0|MIT；Windows 配方显式安装。|
|[ruff](https://pypi.org/project/ruff/0.16.9/)、[black](https://pypi.org/project/black/26.5.1/)、[isort](https://pypi.org/project/isort/9.0.1/)|0.16.9 / 26.5.1 / 9.0.1|MIT；开发工具，若完整运行包带出也属于发行内容。|
|[jupyterlab](https://pypi.org/project/jupyterlab/4.6.4/)|4.6.4|BSD 文本；随带前端/传递依赖另核。|
|[jupyterlab-code-formatter](https://pypi.org/project/jupyterlab-code-formatter/3.1.0/)|3.1.0|本地 License 字段为 MIT 文本，但 classifier 写 BSD；发行者应以该实际版本原许可文件确认，不把矛盾静默归并。|
|[nbconvert](https://pypi.org/project/nbconvert/7.17.1/)|7.17.1|BSD-3-Clause。|

`warmup.py` 还声明 `jupyterlab-lsp`、`python-lsp-server`、`python-lsp-ruff`；本次 Windows 运行包未安装这三项（Windows 配方与通用配方不同），没有为它们虚构已分发版本或完成许可核验。若发行时安装/捆绑，须补入该目标的记录。以上不是对现场所有 site-packages 作全树枚举。

## 完整运行包的其他层次与发行待确认项

- [Python 官方许可说明](https://docs.python.org/3.11/license.html) 为 PSF License Agreement，并带其他历史/内含组件声明；保留解释器随附文本。
- `Cargo.toml` 的 workspace `license = Apache-2.0` 只声明本项目 crate，不覆盖 `Cargo.lock` 中所有依赖及其 native 链接；Rust/C++ 对应源码、许可与 notices 仍须按具体发行目标补齐。
- `build-win-native/bundle.bat` 会生成 `THIRD_PARTY_LICENSES.txt`，但脚本里有说明不等于本次目标的许可证集合已经完整；本地运行包实际 `licenses/` 观察到 `PPF-LICENSE.txt` 和 `CUDA-LICENSE.txt`，不能凭此认定其余层已审完。
- NVIDIA 库遵循所分发版本的 [CUDA EULA](https://docs.nvidia.com/cuda/eula/index.html) 及其可再分发范围，而非 Apache。ROCm、Eigen、MinGit、编译运行库、数学库等以目标真正携带的文件及其上游许可为准；本次未做完整二进制依赖闭包或重新授权判断。
- 若实际分发 MPL/GPL/LGPL/AGPL 组件，发行者应针对各自条款落实需要的许可/版权文本、对应源码及获取方式、修改说明或其他适用义务；是否构成组合工作及具体范围需结合实际产品评估，不能用本文表格代替。
- TetGen/Triangle/plyfile、native 内嵌库和全部传递依赖仍是发行专项确认项。必要时取得适用的另行许可或专业法律复核；本次不以该待办擅自删功能或换依赖，也不宣称“完整运行包已获法律许可”。

公开源码、发布插件 ZIP、发布完整运行包是三个不同决定。本文可随源码作为依赖说明使用；并未执行公开发布、共享权限修改或外发私有材料。
