# 统一诊断反馈处理，2026-10-28

本次输入仅报告及AST替身脚本，没有修改版源码。基线为W8 09:55UTC分享包；保留之后三文件模式切换/编辑预览修复，不用旧树覆盖当前源码。以下状态不代表公开许可、全部平台或全部场景验收。

## 本轮具体修复

| 编号 | 处理与证据范围 |
|---|---|
| P01 | 多骨完整预设手动关节帧重绑循环修正；真实Blender双骨位置/旋转基线失败、修后通过。 |
| P03 | SET_COLLISION只传碰撞字段；不同颜色的两连接真实操作后颜色不变。 |
| P04 | 选择性预设候选按incoming父链接设置验证；正负控、失败保全通过。 |
| P05 | 5.1.2实际GPU事件循环24步、9次刷新，EDIT写入0；未证实报告的风险，不擅自关闭定时器。 |
| P02 | full/single按共享Action身份先收集原fake-user再提升；同Action两对象False/True顺序反例复现，修后还原原状态。 |
| P06 | 整体data恢复的作者保护覆盖Mesh UV/attributes/materials/weights、Curve参数、Key/KeyBlock可写属性/数组/custom与relative关系；真实修改拒绝保全及无编辑恢复通过。 |
| P07 | full恢复记录只暂停其目标Curve，无关Curve真实PC2继续播放。 |
| P08 | begin中途失败清理私有备份并恢复Action元数据，注入第二目标失败通过。 |
| N01 | 稳定缩放范数消除合法normal小数平方下溢；梯度effort中间溢出处理及FTZ能量分摊保全。源helper基线反例复现，修后CPU常规及极端回归、实际CUDA sm89 FTZ五例通过。正式配套构建及原生短程状态见下。 |
| E03 | frontend/插件镜像区分dll/so/dylib，darwin使用POSIX sh/LF及对应库路径；跨平台路由是host fixture验证，不是macOS实机。 |
| E07 | 无显式产品配置的普通场景保留optional contract入口；真实生产编码→build_worker.main→生成脚本→CPU Run完成0.31s/10有限帧，扩展门及无效显式选择仍拒绝。 |
| E08 | Linux/macOS包精确包含两个共享FEM helper；隔离host运行缺helper失败及补齐后通过，不宣称实际POSIX bundle验收。 |
| E09 | 显式相对v2配置按后端记录实际 build_dir、程序和库路径；两个 consumer、配置CLI及 Linux/macOS bundle 已接入，保留旧v2原后端兼容，未配置后端明确拒绝。路由／生成配置的 host fixture 通过，配置仍不等于实际能力；Linux/macOS原生构建和 bundle 运行尚未实测。 |

CPU原生短程：PASS；CUDA原生短程：PASS。每个case和构建的实际退出、时长与阶段见handoff证据；没有把替身、Build READY或helper执行等同完整求解。原生基线短程仅覆盖RR/RF/FF和旧分轴，不代表用户长裙整段。

## 2026-10-30 源码交接进展与保留事项

- E01：补回[第三方分发说明](THIRD_PARTY_DISTRIBUTION.md)，当前候选五个cbor2 wheel的MIT声明与原许可文件存在已直接读取；TetGen、Triangle与plyfile另作定向说明，保留wrapper和内核的许可区别。历史依赖表与本次观察已分开。完整运行包及所有传递依赖的发行许可仍未审完，不给整个ZIP的法律保证。
- E02：补回原构建说明已引用却缺失的[依赖约束说明](build_dependency_scope.md)和Windows Python 3.11直接依赖约束快照；实际11个指定包元数据及已有可选`--constraints`命令helper验证通过。默认安装配方未改，这不是全传递锁文件或干净机器构建证明；其他平台／工具链实编仍未完成。
- E04：[源码配置边界](SOURCE_PROFILE_BOUNDARIES_20261030.md)、构建说明和README现区分普通场景、相对v2扩展配置与显式历史产品。旧published launcher的Windows解释器绑定是保留的历史兼容限制，不能用于新的通用源码入口。
- E05：[CI授权边界](CI_FORK_POLICY_20261030.md)记录主发布与独立Blender扩展发布的不同策略；主`version`前置门已排除未授权AWS的真发布反例。没有因此修改工作流；远端环境审批、仓库变量及OIDC信任未访问／验证，不能用YAML环境名冒充审批已生效。
- E06：私有原报告迁出公开源码后，README和两篇使用说明的引用改为[公开技术摘要](USER_REPORTED_PUBLIC_SUMMARY_20261030.md)，修复了这三处具体断链并保留私有原件。最终选定公开候选的其他隐私／二进制资源与真实Git历史仍须按实际交付范围判断，不把定向链接测试当全包审计。

上述文档与针对性检查已获有范围复审；没有发布GitHub、改共享权限或删改官方功能，也未整删构建依赖的coordination生成器。

N02：现有source_stamp只覆盖特定restoring相关文件，不是所有内核和整个引擎来源的摘要。本轮配套编译、实际选择路径、ABI布局准入及实际运行是各自证据，不能用其中一个替代其他项。没有新增例行哈希/全树一致性审查。

N03：W7历史报告已标适用版本。其当时圆锥未实现的记录保留；W8及本轮报告说明后续进度，不把不同时间状态混成同一版结论。

旧完整烘焙恢复记录缺少新作者数据签名时将保守拒绝，避免猜测性覆盖；不会把旧记录假装无损迁移成功。Linux/macOS/Metal/ROCm未获本轮实机验收。任意Ctrl-Z原生崩溃仍按用户决定暂缓；Outliner自动收起已有5.1.2生产回调的目标root/Empty与无关展开状态保护测试，范围见BONE_PLAYBACK_AND_OPEN_ISSUES_20261028.md，不外推所有版本和交互。
