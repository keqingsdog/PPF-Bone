# 源码构建与历史产品配置的边界

本分支保留上游普通场景入口，也保留旧产品的兼容代码。新机器不应复制历史产品配置、修改其中盘符，或根据旧产品名称推断本机求解器能力。

## 普通场景

没有显式产品选择，且输入不请求骨骼等扩展时，`frontend/build_worker.py` 以空构建合同启动普通路径。`frontend/_r14_product_adapter.py` 的 `activated_profile()` 在没有 `PPF_CTS_PRODUCT_ROOT` 时返回空值；普通后端由原有设备和产物发现逻辑选择。源码根目录里的 v2 source locator 本身不会把普通场景强制路由到旧产品。

`BUILDING_BONES.md` 中“没有本地配置或显式产品选择时会清楚拒绝”应理解为**实际请求需要产品合同的扩展时**：普通场景不因此被一律拒绝。无效的显式旧产品选择仍会报错；v2 source locator 不激活普通场景产品合同，其完整有效性在实际扩展选择时验证。

## 新源码构建与扩展

按 [BUILDING_BONES.md](../BUILDING_BONES.md) 构建所需后端与配套运行目录；为该目录显式生成 `LOCAL_SOURCE_BUILD.json` v2 相对路径配置。路径及 Python 解释器应指向使用者实际准备的文件。Windows CPU 的示例：

```text
python tools/configure_source_product.py --relative-root . --backend cpu --python-relative python/python.exe --bin-relative bin --build-relative target/cpu/release
```

这是目录结构示例，并不是下载、编译或自动补齐运行环境的命令。Linux、macOS 或其他目录布局应使用本机实际 Python、构建目录及平台产物名称。配置文件是本机生成的运行配置，不提交源码仓库。命令及路径选项以本树工具和构建说明为准。

真正请求扩展时，`load_profile()` 优先使用明确传入的根或 `PPF_CTS_PRODUCT_ROOT`；未指定时仅在当前源码／运行根存在有效 v2 source locator 时选择该根，否则报 `PRODUCT_SELECTION_REQUIRED`。本地 v2 选择实际后端后，`backend_routes` 指定相应的构建目录、程序和后端库；选中目录与配置不符或后端未配置时明确拒绝，不猜测切换另一个后端。旧 v2 没有 `backend_routes` 时，仅允许其原配置后端。实际扩展仍需匹配真实产物与能力检查；写入配置本身不是能力证明。

## 显式旧产品

`frontend/_r14_product_profile.py` 的历史路径、产品名称及对应固定身份规则是旧发布产品兼容数据；`DEFAULT_PRODUCT_ROOT` 的名字不代表默认选择它。当前 frontend 选择入口使用明确根或上述 v2 source locator，不以这个常量自动启用历史路径。

显式选择已登记的旧发布产品时，仍走其固定产品合同及对应验证。旧 published launcher 的解释器检查也绑定了历史 Windows Python 路径，这是一项真实的旧产品兼容限制，不应复制到新机器当作源码构建配方。源码本地配置走它自己的 Python／实际产物路径。

## 验证范围

本说明基于所列生产入口的源码路径及现有 E07/E09 定向证据：普通 CPU 场景曾经实际 Build/Run；E09 的配置及路由为 host fixture 测试。Linux/macOS 原生构建、打包、签名和运行未因本文得到验证。本文没有改动选择逻辑或产品合同；历史兼容实现的去机器化若需进一步改造，应单独设计并用真实旧产品验证。
