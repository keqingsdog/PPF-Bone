# Linux 插件增量合并与作者态跟随修复（2026-10-25）

> **当前有效创建规则（用户最新明确要求，覆盖下文旧间距修整记录）**：创建、编辑和切换代理形状属于作者阶段，不要求预先证明碰撞间距，不以粗包围界拒绝，不为避碰自动裁剪输入尺寸。保留合法网格/面积/体积/质量/归属校验。碰撞组、相邻排除以及真实build/native检查不变。下文早期coarse planner、分离证明及自动收短仅是已被取代的诊断历史，后续移植不要接回创建入口。

本轮以当前 portable-source02 为基线，增量合入用户提供的 b0e12e8 插件补丁，保留本地之后的 Undo、连接票据、缓存归属及恢复力默认值。没有整包覆盖，也没有修改原生求解器、ABI、FP64 规则或重编 CPU/CUDA。

## Linux 交付的合并

交付的 28 文件增量涉及捕获与烘焙事务、Pin 清理、Profile 预检、预设确认目标、删除后的列表选择身份、恢复力作者库及插件入口。另修正直接测试发现的独立捕获对象身份、曲线烘焙取消丢失作者工作 Action、DeleteGroup 确认期间复用组槽的问题。

Windows Blender 5.1.2 的私有后台测试：捕获/烘焙 16 项、Profile/预设 10 组、入口/删除身份 13 项通过。确认弹窗部分采用方法截获；没有据此声称真实跨窗口弹窗、网络、GPU 物理或整个插件全部通过。Linux 附带的普通 CPU 前端参考补丁独立保留，没有混入本轮插件增量。

## 静态均匀姿态缩放

用户保存的场景中，Bone.005 继承 Bone.001 约 1.363726 倍的真实均匀姿态缩放；对象缩放为 1。作者跟随此前能预览该变换，创建/编码却只接收刚性帧。

现将刚性帧与静态均匀倍率分开：创建几何使用实际世界长度，关节帧保持刚性；作者倍率只写入绑定回执，显示映射据此保留初始头尾，重复播放不累积。单位倍率保留旧回执字段形状，旧回执缺字段默认 1。两个 frontend 适配器是 Python 元数据处理，不涉及原生编译。

用户文件副本的 Bone.005 实际 Add、collect、encode 通过；刚体与 FEM 映射/commit、FEM 形变缩放开关、旧回执和非法倍率、共享 Action slot、祖先动画、已烘网格及 170°/179.9°胶囊路径有定向证据。原生帧拟合及接受帧验证在显示测试中使用替身，未执行新的物理解算。非均匀缩放、剪切、镜像及相关动画缩放不能静默冻结。

## 混合模式错位

最短函数路径：姿态旋转未结算时切到物体模式并移动骨架。旧逻辑仅在代理世界矩阵不变时保持旧偏移，误把随骨架继承的移动当作手动代理编辑，并学习错偏移。改用骨架局部变化判断后，同一序列中心误差由 1.614405 米降至约 5.96e-8 米，第二次结算稳定；长混合模式对照约 7.16e-7 米。

真实 MCP 现场进一步确认：骨架处于 Edit，Edit 与 Pose 域不同，且跟随模块基线为空；作者场景没有求解缓存或忙状态。新增两项修复：

- Edit 时不清空可信基线并假装初始化完成。可信 raw 作者偏移写入代理的 `ppf_bone_author_follow_v1` JSON，以稳定 UUID 解析；Undo/Redo 重建后不使用旧 RNA 指针。
- GPU 在 Edit 时只读投影到 Edit rest 域，退出回到 Pose/Object；不把显示矩阵写回真实代理、网格或作者偏移。

持久偏移测试包含真实 Undo/Redo，延迟清理回调由测试显式调用；Edit 显示 8 项后台几何/绘制批次测试通过，没有 GPU 像素验收。旧代理没有历史偏移记录时，只能兼容采用当前关系，不能据此宣称恢复了历史错误；本轮用户已另外授权将当前四个代理重新对齐，保留网格、材质和尺寸。

## 运行与回归

需要正常重启 Blender 才能加载替换后的插件；使用已经启动的 Python 求解服务时，也需正常重启服务才能加载两个回执适配器。不会自动关闭、热重载或改连接选择。

源码回归入口：

- `tests/blender/test_static_uniform_pose_scale.py`
- `tests/blender/test_author_follow_durable.py`

本机详细证据保留在 `<历史证据根目录>\plugin-cloud-merge-20261025` 及最后一轮 `plugin-only-delivery-20261025/bone005-scale`。原始失败结果仍保留。当前完整点击历史未保存，MCP 中只读到近期选择记录；不能声称恢复了用户本次完整操作序列。历史原生 AV 的唯一原因仍未证实，不保证任意场景无崩溃。

## 本轮实际交付

2026-10-25已部署34个插件文件和2个Python回执适配器，现有两个安装ZIP及桌面源码分享ZIP已更新。最终v2安装ZIP在私有Blender5.1.2通过真实扩展安装、启用、骨面板/操作注册、cbor2原生wheel编解码及新增API导入；进程正常退出0。

用户授权的4代理已在MCP真实场景重新对齐，原网格和材质保留、世界轴向尺度保留，最大中心误差约1.284e-6米，随后再次读取保持稳定，持久record和4条内存基线已建立。原始blend未保存覆盖，执行前变换数据已备份。未热重载当前Blender；用户保存现场后正常重启加载新版。

## 同日后续反馈：Edit 骨长、物体缩放及 543954a

用户重启后又复现 Edit 缩放骨骼错位。EditBone/PoseBone 的矩阵不包含骨长，旧路径只比较矩阵，尾点沿原轴拉长也可能不触发跟随。现以可信作者参考骨长检测变化：中心沿骨局部 Y 加半个长度差，源网格沿代理本地 X 按长度比伸缩，横向尺寸、手工偏移及旋转保留。伸缩绕源网格局部 X 包围中点，避免放大作者在 Edit Mesh 中的整体平移。

Edit 预览与退出 Edit 的实际提交共用参考和伸缩矩阵；只在整批几何/矩阵提交成功后推进参考。重复结算不累积，失败恢复原网格、矩阵、记录和参考。Apply Object Scale 仅重定义坐标但世界端点不变时，不重复拉伸。私有真实 Edit S2、尾点拉长、手工偏移、两种 scale 表示、局部网格平移、失败回滚与 Apply Scale 的针对性证据保留在 `edit-scale-followup`；新增源码回归 `tests/blender/test_author_follow_length.py`。

另一个报错来自 Armature.001 自身约 1.441 倍的静态均匀物体缩放，而非前轮父骨姿态缩放。作者路径将实际 `A @ Pose` 分解为刚性帧与世界均匀倍率；回传以真实 Armature 逆变换还原本地姿态。文档中 `armatures.matrix_world_blender` 保留真实 A，两个 Python frontend 仅对该作者元数据接纳正均匀 similarity，骨/代理/关节的原生帧仍严格刚性。没有自动 Apply 用户物体缩放，也没有修改 native wire 或编译原生求解器。非均匀、剪切、镜像和相关动画缩放继续明确拒绝。

“刚体网格存在面积为零或过小的面”现附骨架、骨名、代理名及尺寸指引。原面积/体积门保持不变；约 2e-6 米的微小 Bone.002 属于真实过小几何，用户已自行删除。错误包装不删除骨、不扩大整个场景、不吞掉受损作者网格。

云端保存修复以用户补充的 543954a（含其 5bee465 基础）为准，增量合入 `ui/dynamics/bake_ops.py` 与 `core/bone_bake.py`：成功烘焙后仅对真正零用户且原未 Fake User 的源 Action 设置保留标志，干净回滚只撤销本事务新增标志。保留本地作者编辑后的保守取消保护，未整体覆盖插件。Windows 私有 5.1.2 验证包括 11 个生产端用例、7 个独立新进程重开及 2 个完成后 Abort helper 边界，进程退出均为 0。骨姿态样本是本地夹具，不冒称原生物理；云端 GUI 证据不冒称 Windows GUI 验收。

**保留限制：Undo 能恢复作者状态，但不会恢复烘焙时删除的磁盘 PC2 缓存；重要工程须保留工程和 data 缓存副本。** 两个云端旧 ZIP 留作原始交付备份。本次不猜测或改写用户当前代理在旧路径下的历史尺寸。

## 用户 14:35 撤销原生 AV：绘制生命周期隔离

用户在已重启的 Blender5.1.2 撤销后仍发生原生访问违规，旧结论不覆盖此崩溃。已封存原 `Untitled.crash.txt`；Python栈精确位于 `ui/bone_overlay.py::geometry` 对原始 `mesh.vertices` 的遍历，原生栈经过 `AttributeStorage::lookup`、`Mesh::vert_positions_for_write` 和 `Mesh_vertices_begin`。

本次只修改 `ui/bone_overlay.py`：region Draw 只消费不可变 primitive 快照，不解析绑定、读取网格或触发依赖图求值。Undo/Redo/load pre 使generation失效、清空快照并取消排队producer；post只排队。后续稳定主循环完成新鲜view-layer求值、通过重建屏障、取临时evaluated mesh，释放临时mesh后才原子发布纯tuple。发布前后核对generation，不持RNA对象跨事件。不是捕获异常或按任意秒数等待来掩盖原生AV。关闭全部GPU显示时停止轮询，由属性变化重新唤醒。

6项后台隔离测试通过。真实私有Window以截图明确的Armature.001/Bone.004、2.08326倍缩放做固定对照：Resize真实完成、自然FramePump替换网格，Undo/Redo确实恢复/重做骨长，稳定后重新按UUID读取原mesh成功。旧版实际Draw内geometry672次，其中重建屏障期间72次；修复版Draw内与屏障期间geometry均0，实际Draw129次、GPU批次1808次、producer83次，未靠停绘制或停采样通过。两进程正常退出0。

**原始AV在新旧私有重放中均未复现，故这次只证明危险绘制路径隔离与重建后正常恢复，不证明唯一根因或全部撤销崩溃已消灭。** 后台独立producer探针在真实copy/swap/remove旧mesh后Undo/Redo可稳定重读，因此没有凭推测改core删除策略或永久保留旧mesh。原始无选骨的首轮重放Resize取消，另行保留，不称缩放复现通过。

本机证据：`plugin-cloud-merge-20261025/undo-crash-followup/WINDOW_REPORT.md`、`WINDOW_COMPARISON.json`、`overlay-snapshot/REPORT.md`及原始crash文件。本轮不操作用户崩溃窗口、不覆盖原blend、不改引擎、不计算哈希。正常重启后加载新的绘制路径，继续以具体反馈跟进原AV。

补充原生键盘事件测试未完成目标动作：本机event_simulate API存在，私有Window收到4个原生事件（含Tab PRESS/RELEASE），但未进入Edit，故S/Enter/原生CtrlZ均未执行。自然退出0不是功能PASS；保留原Window测试显式undo_push分组的边界。未尝试OS激活或操作用户窗口。单文件补丁已实际部署且最终ZIP安装测试PASS；原AV闭合仍待具体复现或用户5.1复测。

随后已完成一个有界替代对照（native-modal-step/REPORT.md）：直接invoke真实Transform modal，原生数字/Enter确认2.08326倍；保持Edit时mesh=M0，退出后自然FramePump替换M1，期间仅setup初始push，无后置push。一次Undo恢复fresh EditBone原长，一次Redo回放大长度；新旧均自然退出0且未复现AV。旧Draw内geometry520/重建期120，新Draw和重建期均0；新GPU批次1648/producer84。默认INVOKE调用已由实际Undo/Redo证明自动组成立，无需为建议bool参数重复。此关闭自动Resize UndoStep先于派生mesh替换的测试时序缺口，仍不等同完整键盘Tab/CtrlZ重放，也不证明AV唯一因果。Undo时派生mesh仍M1且可读，不能声称整个派生网格还原；未新增产品修改。

## Linux 作者动画保护增量：26f8560 与 251bbb9

两个云端更新已按目标文件增量合并，仅修改 `ui/dynamics/bake_ops.py`，保留本地 Curve 作者编辑后的保守取消逻辑。没有整包替换，本地 GPU 快照、缩放、骨长与其他修复不会被旧云端快照覆盖；旧交付 ZIP 保留。

26f8560：Curve 取消时保留作者改绑/清空/删除源 Action 的意图，Mesh 保守取消时为否则会孤立丢失的旧源 Action 设置保留标志。在修改作者数据和删除 PC2 前拒绝活跃 Curve 坐标/手柄驱动器；Animation 也拒绝精确坐标通道的未静音 F-curve modifier，不删除/静音它们来通过。不泛拒非坐标属性、真正静音通道和已验证的加法 NLA。

251bbb9：已包含用户随后补交的 Curve Single Frame 坐标动画保护。单对象和批量入口先验证整个目标集，遇到会覆盖烘焙坐标的 Action/NLA 就提前拒绝，保留原动画、组、旁边目标和缓存。修复方式是无损拦截冲突配置，不是声称可直接覆盖任意已有坐标动画。此前“26尚未包含”的限制仅属于历史26单包，当前已由251守卫覆盖。

本地 Windows5.1 layered Action 另有实测反例：仅 ActionGroup.mute=True、坐标FCurve自身仍未静音时，Single 原先假完成，切帧从烘焙0.2回到动画0.4。最小兼容适配不把 layered 组静音当有效豁免，仍保留真正 FCurve.mute 和 legacy 组静音语义，不改作者标志。

Windows 私有5.1.2针对性证据：26增量14生产端、9独立重开、1双Action-slot用例通过；251增量10生产端、10独立重开通过。当前slot无modifier允许、另一slot同名co通道有GENERATOR则拒绝，复用已有绑定slot解析，不遍历所有无关slot。拒绝路径保留作者和PC2，允许对照实际完成并切帧/重开保持。Linux证据和Windows证据分别记录，不冒称全部GUI或物理求解通过。

本机统一证据入口：`plugin-cloud-merge-20261025/cloud251-merge/REPORT.md` 与 `cloud26f-merge/REPORT.md`。Undo不会恢复已经删除的磁盘PC2的限制继续保留；原14:35原生AV的唯一归因/用户复测仍开放。全四边面、五档精细度和形状快捷键已在后续独立生成器里程碑完成交付，其证据见下节。

## 默认全四边面、生成精细度和删除重建

新增生成器里程碑（最终审查PASS，11文件增量）：内置胶囊、球、盒默认全部四边面，包含经用户授权调整的极点连接；可选三角面。自定义网格保留作者拓扑。五档为极低、低、中、高、极高，新刚体默认低，新软体默认中。旧LOW/HIGH及形状的枚举序号保留，旧HIGH几何不降级；已有作者网格不因升级自动重建。

| 档位 | 胶囊顶点 | 四边面 | 输出三角面 |
| --- | ---: | ---: | ---: |
| 极低 | 34 | 32 | 64 |
| 低 | 74 | 72 | 144 |
| 中 | 178 | 176 | 352 |
| 高 | 642 | 640 | 1280 |
| 极高 | 962 | 960 | 1920 |

四边面减少编辑显示中的多边形数，原生编码仍输出三角面，不宣称解算开销减半。代理编辑模式按BoneX约定使用T单键循环胶囊/球/盒，当前范围所有目标都有有效自定义源网格时才加入自定义；缺少或无效源时循环跳过，不产生缺源报错。数字键全部释放；正在G/S/R变换时T和Esc也交给变换操作，带Ctrl/Alt/Shift/Win或Command的T不切形状。生成选项支持当前范围批量应用，并进入跨工程预设及MCP GENERATE入口。

原报错并不等同用户手工损坏网格：插件自身轴向伸缩、烘入缩放和不同拓扑也会不再符合旧胶囊模板。现在从实际求值后的世界网格建立保守胶囊包围界；只有包围界证明足够间隙时才允许新邻接代理，无法证明仍明确拒绝，原始面积、闭合、体积门不放宽。90°和170°的原错误路径均有修复前失败、修复后真实生成成功的证据。

旧逻辑将自动避让收缩写成普通尺寸，保留UUID的删除重建又把它误认成手工尺寸。现在成功自动生成后记住尺寸来源，仍匹配自动值的重建按当前骨长重算。后续按用户要求补齐了旧版默认生成网格的显式Delete→Create语义（见下节），无需额外点击“重算默认尺寸”。该按钮仅供主动重置已有代理，保留形状、材质、恢复力、拓扑、精细度和作者世界位置/旋转；自定义网格整批不适用。普通更新与重开不猜测迁移未知旧尺寸。

证据：plugin-proxy-generation-20261025/REPORT.md（60个生成组合），ui-hotkeys/HANDOFF.md（5组UI/预设/MCP入口、2组尺寸重算），clearance-followup/REPORT.md（包围、实际BVH、90/170、自动与手工/legacy来源）。均为私有Windows5.1.2程序化测试，不冒称自然GUI或新的原生物理解算验收。测试观察器持有跨Edit旧RNA导致的两次停留已保留，不计通过。引擎未改、未编译。

最终审查另关闭两处具体接点：质量辅助统一使用编码器同源loop triangles，避免低精度非平面四边面的另一种三角化导致质量校验误报表面相交或体积不同；9个Sphere/Capsule、QUAD/TRI、正仿射与Edit对照和canonical体积吻合，镜像、负体积仍拒绝。基础网格仍匹配胶囊模板但有modifier/shape keys时，不再采用较小的基础包围界，而是转入实际求值网格；扩大表面的unsafe候选拒绝，safe对照通过。旧基本几何不被修改。

11文件已增量部署到正式源码、Windows5.1安装目录、两个安装ZIP和桌面源码分享ZIP。最终v2安装包在私有Blender5.1.2真实安装/启用后，实际创建RIGID LOW与SOFT MEDIUM全QUAD胶囊、切换Sphere QUAD/Box TRI并通过质量体积计算；正常退出0（installer-generation/RESULT.json）。未热重载用户Blender、未修改用户场景、未重编引擎。

## 最新 Linux d95db9a：Single Frame 综合安全

已按目标文件增量接续26f8560、251bbb9和d95db9a，最终审查PASS，仅修改ui/dynamics/bake_ops.py，不整包覆盖本地作者跟随、GPU快照、缩放、生成器或其他修复。原始三个云端ZIP继续保留。

Single及All Single先对全部目标做只读准入并冻结通过验证的实际姿态，然后才执行骨烘焙和PC2清理。Mesh的形态键（包括仅Basis）、共享/跨Scene/链接数据、求值后顶点数量变化、非有限坐标、坐标动画/驱动器、缓存后的启用或可见性动画修饰器会提前拒绝。Curve的形态键Action/NLA/driver及绝对形态键eval_time动画也纳入保护。静态/真正静音且已经验证可保留结果的配置仍允许。

保留Windows5.1 layered Action组静音的兼容守卫，并共用到坐标、形态键和modifier可见性通道；不修改作者flag来通过。批量在任一目标准入失败时，前面的正常Mesh及骨阶段也不会被提前消费。拒绝不会删除原动画或PC2。此为全目标准入改进，并非承诺提交阶段任意I/O或回调失败都有通用回滚。

私有Windows5.1.2证据：16个生产用例PASS、独立进程重开17条对象记录PASS（含一个批量用例的正常相邻对象），两进程正常退出0。3个允许复合配置使用真实evaluated.to_mesh对照，几何最大误差0，切帧及重开保持。没有重跑云端170套用例，不将Linux结果冒称Windows GUI或物理解算验收。证据入口plugin-cloud-merge-20261025/cloudd95-merge/REPORT.md。

磁盘PC2删除的Undo仍不恢复，重要工程应保留工程与data副本；复杂META/transition/tweak与任意作者回调不在本轮定向证据范围。此前原生Undo AV也未据此被宣称消除。

## 删除后重建自动重算：无需额外按钮

默认“删除代理→创建代理”现在对未编辑的内置生成网格记录一次性DEFAULTS意图，并按当前骨骼世界长度重新计算半径、长度比例、盒宽和盒深，再执行原有间隙适配。不要求用户再点“重算默认尺寸”。这条显式重建语义同时覆盖无来源记录的旧canonical默认网格；历史手工输入如果与该几何完全相同也不可区分，会按本次重新创建意图重算，不能冒称恢复了旧尺寸来源。普通更新和重开不触发该策略。

Custom、明确编辑或有求值变形的网格、已有AUTO记录证明手改过的尺寸优先PRESERVE；无可信几何历史的旧非canonical拉伸网格也不猜来源。删除后修改尺寸或形状会取消DEFAULTS，即便又改回旧数值也不落回旧AUTO重算；同值赋值不会误取消。标记绑定骨架/骨/体/关节身份、形状和四尺寸，创建成功才消费，失败恢复状态和标记，可正常重试。材质、恢复力等非尺寸参数不重置。

3文件增量经senior审查PASS，私有Blender5.1.2的8组测试自然退出0。其中实际默认bpy.ops删除→默认创建，不传新参数：旧无来源R0.03、ratio0.2，骨长改为2后自动得到R0.4、ratio1；另覆盖手工/Custom、编辑网格与旧AUTO优先级、删后修改与改回、普通更新、真实小几何拒绝、提交后注入失败回滚和重试、外来对象所有权拒绝。证据：plugin-proxy-generation-20261025/legacy-recreate-auto/REPORT.md。未执行新的自然GUI或物理解算。

## 按用户反馈改为BoneX的T单键循环

旧1/2/3/4及小键盘形状拦截已删除；T沿用本地BoneX参考panel/op.py:198的切换约定。仅当前代理编辑器视图区、无修饰键、无正在处理的变换时切换。多选以当前骨形状决定整批下一项，自定义源必须在全部目标上有效、闭合且非owned代理，否则循环静默跳过自定义；显式手动选择无源Custom仍保留原代理并报出具体缺源。

2文件增量senior PASS，私有5.1.2的2组测试退出0：真实modal方法调用生产operator完成多选切换、混合缺源/坏源跳过、全部有效custom独立复制；数字及G/S/R、Esc、修饰T透传。Win/Command+T另有不生成和shape/mesh不变断言。事件/context为替身，不称自然GUI验收。证据single-key-cycle/HANDOFF.md。

## 当前现场间距误拒：固定网格投影证明与碰撞策略

已用MCP只读确认当前Blender5.1.2的Untitled.blend，并以libraries.write导出私有Scene副本；未修改live场景或覆盖原文件。捕获时Bone.004缺代理且有PRESERVE意图，Bone.003为旧非canonical胶囊，Bone.005已切换Sphere。私有副本先复现Bone.003/Bone.004同文拒绝；原反馈004/005因现场形状已变，不冒称逐字重现原状态。

旧保守胶囊过松：当前默认Create在实际网格的全顶点单位轴投影可证明间距，但旧外包胶囊无法证明。现仅对不会被其他pair继续拟合修改的固定pair，使用最多504个有限单位候选轴，核验双方所有实际求值世界顶点的支撑区间。间隔需大于原offset和gap、32-epsilon坐标舍入，再加128-epsilon余量。任意多边面处于其顶点凸包内，因而这是充分的分离证明；BVH无相交仅作诊断，不能直接放行。无证明仍走原fit或拒绝。Sphere纳入固定障碍物，原网格面积、体积和密度门不变。

创建planner还补齐原漏接的组过滤双向membership/mask AND规则，关闭过滤视为全组全mask；之后再施加子骨相邻排除，兄弟骨不误用父子排除。相邻选项不会重新允许已被组过滤禁止的碰撞。本次现场全部允许碰撞，所以实际修复依靠网格证明，而非偷偷关闭碰撞。

source副本实际Create返回1，旧003/005 mesh身份、全部坐标和worldmatrix不变。003/004支撑区间分离0.00566810，大于required0.00335030+reserve0.000134858；004/005 Sphere分离0.361555，大于0.00368403+0.000163916。真实重叠114对三角形的反例：组允许且相邻允许仍拒且状态保留；组禁/相邻允、组允/相邻禁、两者禁均按策略成功。两脚本自然退出0，2文件senior PASS。证据clearance-policy-followup/REPORT.md。

最终安装包另用libraries.load导入同一个私有Scene，只选Bone.004调用默认创建验证；该隔离重放关闭作者跟随回调、不触load/save handlers、不修改live用户场景。自然GUI和原生物理解算不由这些测试宣称通过。新收到f740610的完整动画失败保护仍独立合并，未由本几何里程碑宣称完成。
## f740610 完整动画烘焙失败保护（已合并审查PASS）

基于用户新交付f740610，只增量修改ui/dynamics/bake_ops.py、core/pc2.py、core/bone_bake.py；保留本地5.1 layered静音判断、Curve保守Abort、作者跟随、T循环和实际网格间距修复。未整包替换，也没有引擎修改。

完整Animation Bake遇Curve形态键（包括仅Basis）先拒绝，保护作者键；Single Frame先前允许的静态Curve键仍可用。全队列只读清理预检发生在提交前，但它仅识别已知不可用输入，不是Windows删除权限保证。缓存必须实际unlink成功后才拆修改器/内存引用和上游对象状态；unlink拒绝明确报错保留引用。晚发错误区分可回退、已提交、结果不确定：保留已完成或不确定输出及原Action，只撤回仍可安全恢复的未提交目标；重复finalize和终态Abort不重复提交/回退。采样后文件消失明确报错并保留需检查的输出和源Action；骨回退失败标明不确定，避免盲重试。

Windows5.1.2新增8个生产用例和9条独立重开记录通过，两退出0。真实Windows ACL夹具中PC2仍可写且非readonly、父os.access(W|X)也True，但同时拒绝文件DELETE与父DELETE_CHILD时os.unlink抛WinError5；Curve/Mesh生产strict cleanup均保持PC2、Curve内存/句柄/控制点、Mesh修改器/可见性动画与相关状态。Receipt标明phase=unlink、input_preserved=True、disk_removed=False。finally恢复自身私有夹具权限，恢复后可写可删除，Blender自然退出0。未改用户目录权限。

证据：plugin-cloud-merge-20261025/cloudf740-merge/REPORT.md及cloudf740-permission/REPORT.md。Linux4.3.2附带266用例结果独立保留，不当作Windows运行结果；未重跑旧170/266矩阵，不以源码测试冒称安装测试。

本修复不提供整批原子事务、任意I/O/RNA/回调失败的通用回滚。若报已完成或状态不确定，应检查并另存工程，不要反复点击试图强行成功；外部磁盘PC2的原生Undo仍未实现。骨rollback测试使用真实本地finish加异常替身，不冒称新原生求解。正常重启加载插件即可，无需重新编译引擎。
## 自由网格编辑显示与常驻GPU控件

按本地BoneX E编辑原网格的操作方式，PPF进入MESH编辑时给当前目标临时SOLID、wire/all_edges和in_front展示，暂时链接当前Scene root，使隐藏/排除的代理集合不会遮掉可编辑顶点边；其他骨架和集合隐藏/排除旗标不被打开。正常返回、取消和进入失败恢复目标原display/隐藏/链接状态，并回到原骨架及模式。

ui/bone_overlay.py仅在原有安全bindings枚举中排除实际EDIT且有EDITING marker的目标，避免真实网格与GPU重复遮挡；其他代理照常显示。没有在Draw增加RNA读取，没有改producer、Undo barriers、generation或timer快照机制。GPU显示总开关、透明度、面/边和in_front等核心控件放在代理编辑区旁及真实代理编辑页常驻，不恢复已删除的进阶/专业物理面板或长分组列表。

3文件增量经senior PASS，私有Blender5.1.2四组测试自然退出0：实际Select→EDIT/BMesh全选→返回与隐藏集合恢复、只剔除目标GPU而其他骨架继续、入口异常回滚、cancel恢复目标/Pose、draw只读。首轮观察器缺view_layer.update及真实cancel丢active的缺口日志保留；已修并重测。证据free-mesh-edit-followup/HANDOFF.md。没有自然GUI像素或Undo崩溃已消除的声明。
## 最新9888现场：盒体切胶囊的真实相交与自动收短

用户重报间距错误后，9876已切到另一4.3 SDF窗口；未在该窗口操作。用户明确提供9888后，只读确认5.1实际loaded bytecode包含上一轮fixed-support证明，不是以磁盘源码冒充加载版。最新Scene已不同：004有Capsule、005有Box。libraries.write导出唯一私有Scene副本live_scene_9888_20261025_171644.blend，再后台实际默认Generate把005由BOX切到CAPSULE，复现004/005同文错误。

本轮是新胶囊真实相交：15对BVH候选经既有narrowphase逐对证实都有surface交线，不能放宽上一轮正确的分离证明。现仅对明确本次实际改变形状、旧BOX/SPHERE网格可验证且目标为CAPSULE的代理，传入一次性shape_changed_from旧形状映射，允许新候选只收短长度。半径、世界位置旋转、盒宽深、材质、恢复力、UUID及旧邻体保持；普通Update、同形状Generate、作者编辑网格和CUSTOM不获得此许可。长度独自不足仍拒绝并回滚，不偷偷减半径。

length-only目标不会复用fixed-pair证明，最终所有相关pair按最终尺寸重新验证，带float32数值余量。旧无来源尺寸不因这次自动修整被伪写为全尺寸AUTO；已有全AUTO匹配记录才更新。UI旧形状在修改前事务快照取出，许可仅本次调用有效。

最新私有场景实际operator通过：length_ratio从0.96669209只减到0.936637878（约收短3.1%），radius保持0.300449699；空AUTO来源、width/depth、physics和UUID保持，world矩阵误差约4.77e-7，旧004 mesh及placement不变。15处真实交线变0；最终capsule surfaceclearance0.00363942，大于actualrequiredgap0.00356142加rounding0.0000399494。长度不足、编辑旧Box、same-shape/普通Update边界各自然退出0，拒绝路径保原数据。3文件senior PASS。证据clearance-second-feedback/REPORT.md。未修改live用户场景、热重载或执行新原生求解。

f740最终runtime ZIP已在私有Blender5.1.2实际安装/启用及既有创建/切换/旧私有Scene重放通过，退出0（installer-f740/RESULT.json）。新自由网格编辑和本长度修整另在最终ZIP做私有真实EDIT/Return与最新9888副本shape转换重放，测试边界为程序化，非自然GUI/Undo AV验收。
## 第三次反馈：默认添加入口的新胶囊长度适配漏接

用户明确本轮操作是“添加/更新代理”。重启后9888起初未监听，用户重新启动后只读确认5.1.2 frame46/POSE、只选005；实际loaded bytecode包含上一轮shape-change长度修整，不能归因旧代码。005此时缺proxy、enabled=False、CAPSULE、R0.6849630475、ratio0.779626429，AUTO snapshot与PRESERVE删除意图并存；004已有Capsule。唯一新副本为clearance-third-feedback/live_scene_9888_20261025_174112.blend。

私有副本默认bpy.ops.ppf.bone_proxy_update()准确复现004/005同文拒绝。根因是之前length_adjustable只开放existing重建+shape-change入口；PRESERVE让新建计划返回{}，缺proxy实际新建因此仍被当固定不可缩。现只在共用生成器一处补齐：本批选中、缺proxy、实际新生成的内置CAPSULE自动获得length-only安全适配，保留原shape-change规则。已有proxy的metadata-only更新、Custom和自由编辑原网格不扩权，半径自动调整仍仅原AUTO资格，不伪造legacy全尺寸AUTO。

PRESERVE继续阻止默认四尺寸重新初始化，但不再阻止新生成胶囊为初始间隙只收短长度。最终所有相关pair按最终尺寸复验，不复用会拟合目标的fixed-proof；长度无法独自容纳时明确拒绝并事务回滚。没有更改碰撞开关或要求用户再点重算。

最新副本默认Add已返回FINISHED：R、width/depth、physics、身份保持，ratio0.779626429→0.538264692，新frame误差约7.15e-7，旧004全部mesh/placement不变，实际重叠0；最终gap0.003365755大于required0.003305221加rounding0.0000335143。紧接第二次普通Update保持同mesh、全部顶点、worldmatrix和尺寸不变。独立无空间反例拒绝且完整恢复。两个after进程自然退出0，单文件senior PASS。证据clearance-third-feedback/REPORT.md。

本轮未修改live场景或热重载。最终runtime ZIP以独立私有配置实际安装后，重放此同一frame46 Scene的默认Add，并保留原T、E真实网格编辑与返回验证；正常重启Blender加载，无需原生重编译。未据此宣称自然GUI或新的原生物理解算验收。

## 最新作者阶段规则：移除创建clearance硬门与自动裁剪

用户明确要求允许先创建和编辑代理，再由碰撞组和真实场景构建决定物理接触。此前胶囊外包界与分离面证明是保守充分条件，无法证明并不等于实际穿插，不能用它阻止作者生成。仅ops/bone_proxy.py移除创建私有planner/专属fit helpers、生产调用和post-mesh contact-budget硬门；没有捕获并吞掉ValueError来假通过，也没有修改碰撞组来放行。

新代理按用户输入或正常首次默认尺寸生成，不做clearance驱动的自动缩短/缩半径；shape_changed_from内部参数保留调用兼容但不再裁剪。Delete来源分类所需_existing_generated_capsule、新建默认尺寸、作者frame、closed/triangle-area/positive-volume、质量换算、ownership/UUID及事务回滚保留。官方build/native与CPU/CUDA/Metal/ROCm路线不动。普通作者创建允许初始重叠，不代表实际物理构建或求解已经接受。

两个私有Blender5.1.2定向进程自然退出0：旧失败frame46默认Add保R0.6849630475、ratio0.779626429原样生成，旧004mesh/placement不变；独立Head Sphere+同头Hair胶囊有30个BVH初始重叠仍可创建，Capsule/Sphere/Box切换保持输入尺寸和frame、组record/相邻参数不变。Open Custom与零面积闭合Custom仍准确拒绝并回滚。单文件senior PASS。证据creation-clearance-authoring/REPORT.md。以上覆盖此前同日关于创建时证明gap和自动收短的方案，不再继续叠加这些作者门。

## 编码导入契约修复与最小发布依赖闭包

用户实际安装版encoder调用core.connection._selected_product_profile，但其旧connection缺该函数；r14_product_profile也缺has_local_source，local_source_profile文件缺失，属于之前增量发布漏同步的依赖链。现在公开真实selected_product_profile并保留private同函数兼容别名，encoder/params改用公开API；部署同时带真实r14_product_profile与local_source_profile现有实现，不塞stub、不绕过产品校验，不把普通CPU/Metal/ROCm强行当CUDA固定产品。

安装式bl_ext.user_default命名空间先准确复现原ImportError，四依赖补齐后三次真实encode_param→CBOR→decode通过：普通source CPU、指定完整运行包v2 GPU/CUDA、已连接backend的canonical CPU优先于旧UI路径。各Param v2为559 bytes，用户SCHWARZ参数保留。普通CPU/CUDA/Metal/ROCm与v1 CUDA locator不接管其他backend对照通过，共5组自然退出0。证据plugin-encode-import-followup-20261025/REPORT.md。

当前canonical完整运行包的LOCAL_SOURCE_BUILD为scope=source，按原语义selector=None，不走旧R14默认产品，不触二进制checksum；测试没有spy掉生产验证，也未计算哈希、执行native或声明跨硬件运行验收。发布仅五文件合并增量（作者创建1+编码依赖4），最终ZIP另经真实安装、实际CPU/CUDA参数编码和原失败Scene原尺寸创建验证，正常重启Blender加载，无需原生重编译。
## 原生固定场景Python接口：排除对与碰撞组参数实接

用户真实构建错误为scene_fixed_scene_assemble不接受excluded_object_pairs。实际canonical内置Python默认加载target/cpu/release/_ppf_cts_py.dll；其旧签名也缺collision_filter、static_collision_filter、static_tri_offset。源码crates/ppf-cts-py/src/scene_py/fixed_init.rs同样漏了这4参数的PyO3接线，虽Rust core已具备mesh collision-filter装配与组过滤后再排除的逻辑。

现仅该Rust绑定文件补4个末尾可选参数，真实传给core，旧无参/None调用保持兼容。mask/offset/pairs严格检查类型、维度、连续性、长度、范围和finite，不把负数/过大mask用cast截断；排除对规范顺序、去重、拒绝同体/负索引/不合法形状。没有从frontend删参数来躲错，没有关闭碰撞或改GPU求解内核。

分别构建CPU feature与无core/cpu feature的CUDA host-wrapper，独立artifact需要NVIDIA驱动标志为False/True，绝非DLL改名。各自真实内置Python及canonical frontend加载36项检查通过，自然退出0：FixedScene构造有/无排除的接受与拒绝、三对象局部排除不放过第三体、组掩码、静态组/offset及非法输入。CPU和CUDA现有solver的实际--probe/既有format gate也通过，未执行求解。

证据fixed-scene-api-followup-20261025/REPORT.md、CPU_ARTIFACT_RESULT.json、CUDA_CANDIDATE_RESULT.json、BOOTSTRAP_RESULT.json。现有构建脚本按必要格式生成stamp，没有另做哈希审计/手填身份。发布为canonical runtime的两份对应host DLL和桌面源码分享包的可移植Rust文件；S仍完整源码入口，旧R9仅保持现有插件兼容镜像，不凭空创建不属于它的crates目录。用户当前打开的本地服务如已加载旧模块需正常重启，新进程实际加载路径及装配结果另作发布后核验。
## 多选直接调参、相邻默认关闭与真实装配发布

用户反馈此前只改活动骨骼属实：旧面板直接绑定active RNA，批量要另点Apply。现在新增core/bone_selection_parameters.py，以临时WindowManager get/set镜像把GUI显式编辑路由到同一选中作者事务；内部Update/Preset/MCP的原setattr单目标语义不被改成全局广播。注册由ops/bone_proxy_editor.py管理，骨面板共用SelectionLayout；models/bone_bindings.py仅将collide_connected未显式设置的默认改False，老工程显式True不被覆盖。

Pose只取当前真实selected_pose_bones及已选owner，Object只解析本次selected owned proxies，可显式跨骨架；不扫其他骨架残留Bone.select。无选择不暴露可写active字段，E真网格编辑维持原路径。公共标题简洁显示“多选N项，显示活动项；修改即统一”。白名单覆盖36个非动画Bone作者字段，包括碰撞flag/masks、生成尺寸LOD、质量材质、root/软限位及预览色；身份、proxy、AUTO来源、参与开关、物理模型转换及嵌套events/draft保留专用路径，不声称所有字段无条件批量。

批量先冻结目标/原值/显式set状态与group，再验证ownership/独占material/idle/有效动画通道。中间guard不丢必要尺寸来源取消、质量/材质同步，只把昂贵Clear/retire合并一次；失败整批恢复且释放guard。GUI XYZ或mask只把实际改变的分量写到各target，保留每target其余不同分量，同active输入no-op；directAPI显式整vector赋值语义保留。

摩擦在无实际Action/NLA/driver且AutoKey OFF时也直接多选同步；镜像值旁prop_decorator指向真实material RNA，原group面板保留。实际作者动画/AutoKey ON保原RNA单项编辑并提示；后面的选中目标有实际动画则全批拒不部分覆盖。NLA按strip.action_slot读取，muted strip仍保护作者通道。WM slider的hover-I/Driver上下文不冒称等同原RNA，动画仍用真实decorator/原面板。

4文件senior PASS，私有5.1.2分阶段证据：9组基础选择/回调/回滚/动画路径，6组摩擦/真实keyframe/AutoKey/E/0rows，2组mask/XYZ分量与回滚，均自然退出0；不重复旧矩阵。证据multi-selection-parameters/HANDOFF.md。未作自然GUI/Undo AV验收。

原生双wrapper已实际发布到完整运行包target/cpu及target/cuda/release/_ppf_cts_py.dll；发布后canonical内置Python默认CPU与显式CUDA加载路径分别确认为各自该文件，needs_nvidia_driver False/True，36项真实assemble/FixedScene检查各自通过exit0，包含真实group+excluded关系且未排除的交叉仍拒。RUNTIME_CPU_RESULT.json、RUNTIME_CUDA_RESULT.json保留证据；没有把候选DLL测试冒充已发布路径。运行中的旧Blender类及已载旧扩展的本地服务需正常重启，未自动杀进程、热重载或改用户场景，未重新跑solver动画。
### 2026-10-25 当前材质默认与骨长问题（最新要求）

- 用户再次确认新建默认应为RIGID；SOFT新建默认硬度提高，默认代理分辨率提高。执行目标：RIGID/LOW；显式选择新建SOFT时HIGH、Young模量100 MPa（原20 MPa的5倍）。已有显式模型、材质参数、分辨率与用户保存预设不自动覆盖。面板必须显示实际模型，软体材质预设不得因STIFF_RIGID内部枚举误标为刚体。
- 9888现场Untitled的8个参与骨骼实际均SOFT/SOLID；前景4体native质量约724.9/1805.6/3866.7/4049.6 kg，有限软锚刚度均1e5 N/m。frame140附近精确native关节锚间距约1.9725/1.7780/1.4043/2.1834 m，初帧<2.3e-6 m；显示重锚为了连父级跨过这些实际间距而拉长骨骼。单tet局部形变与整体代理轴向形变也有差异。不得仅称刚体材质问题、面数问题或显示问题。
- 当前不转换、不覆盖用户原始blend或磁盘缓存。默认参数修改不等于旧场景物理拉长已解决；刚体自身刚性不等于有限软关节不会拉开。按实际测试与后续决定分别收口，不重启已撤回的硬关节方案。
- 上轮stamp路径问题由用户正常重连后恢复，本轮无路由产品补丁、无GPU重编；旧连接的具体环境来源未获现场证实。

### 2026-10-25 新建默认材质／软体硬度和分辨率已修正

- 5个插件文件实现完成，senior STATUS PASS。新建默认RIGID＋LOW；主动选择新建SOFT时Young模量100 MPa（原20 MPa的5倍）＋HIGH，全四边面胶囊642顶点／640四边面／1280原生三角。网格等级可手调，不更改密度、恢复力和锚刚度。
- 已有绑定、显式E／分辨率／模型及完整用户预设保持，不迁移原Untitled的8个SOFT。面板紧凑显示真实owned PDRD/SOLID类型；STIFF_RIGID存储ID保留但人类标签改为高硬度软体，材质选择标为待应用，不把枚举当实际模型。Draw只读。
- 私有Blender5.1.2实际RNA/operator五组PASS、自然exit0。来源physical-model-preset-followup/HANDOFF.md；最终安装包复用installed_defaults_probe.py的run(addon_name)，只测本轮新增默认／保留／标签／预设功能，不重跑旧矩阵。
- 当前场景拉长的只读证据见plugin-proxy-generation-20261025/bone-stretch-user-scene-20261025/REPORT.md。真实软锚分离与显示桥接并存；更硬或更多面数不等于老场景的关节间隙已修复。本轮无引擎修改、无GPU重编、无原工程或缓存写入、无热重载。

- 发布后最终v2 ZIP已在全新私有Blender5.1.2通过真实ExtensionManager安装/启用及同一run(addon_name)五组检查，natural_exit_code=0。新建默认刚体与软体100 MPa/HIGH、显式旧值、真实类型只读UI、真实Apply及完整用户预设保留均通过；证据plugin-cloud-merge-20261025/installer-physics-defaults/RESULT.json。当前用户Blender未热重载，保存并正常重启后加载。

### 2026-10-25 500 MPa／密度1与首次重建偏细实现完成（待集中发布）

- 默认材质仅new_binding_defaults.py：SOFT首次未显式E为5e8 Pa，RIGID和SOFT首次未显式密度均1 kg/m³；raw旧绑定fallback不改，显式值、旧绑定、完整预设和专用材质配方保留。私有5.1.2真RNA/operator五组PASS、exit0，证据density-500mpa-followup。
- 首次细、次次粗根因是旧网格已按作者uniform缩放，RNA半径未同幅更新；旧Delete仅按RNA拟合模板误判PRESERVE，首Create用旧小半径，第二次新canonical又被认DEFAULTS。ops/bone_proxy.py增加实际local L/R的完整网格/拓扑识别，真实manual/mismatch优先，不伪造旧AUTO；PRESERVE来源写重建代理持久标记，显式reset才清除，事务失败恢复。
- 用户Untitled私有Scene真实Delete/Create×3首轮即稳定；手工/preset四尺寸与局部edited各三轮、普通Update、reset失败回滚与成功清来源均PASS，exit0。证据first-recreate-stability/REPORT.md。源文件已实现，尚未据此宣称当前安装目录已更新；将与中段关闭修复集中发布。

### 2026-10-25 本轮定稿：首次重建、500 MPa／密度1、关闭链中段

- 新建默认RIGID/LOW维持；SOFT新建500 MPa/HIGH，两模型首次未显式密度1 kg/m³。创建前密度控件也只读显示有效默认1，不改旧RNA fallback1000，不写Draw状态；旧绑定、显式参数和完整用户预设保持。实际5.1.2六组PASS、exit0。
- 代理首次重建偏细已定位并修正实际网格尺寸与RNA旧尺寸的识别差异；真实Untitled私有作者Scene三轮Delete/Create首轮即稳定，手改/preset尺寸和edited来源持续保留，普通Update不动，失败回滚与显式reset语义保持。只读历史模板精确几何匹配不代表哈希/安装一致性审计。
- BoneX1.2.3 Python会把未参与的直接父骨作隐藏运动学端点，并在每次交互帧重读Blender世界姿态；不擅自断言其闭源exe内部算法。旧PPF在A物理/B关闭/C物理情况下把B世界轨迹预采样，实际原生短链A降0.09903 m而C仅6.62e-6 m，证实下段被世界端点留住。
- 修复静态非物理中段：复用既有采样检查整条中间链，原生C接最近物理祖先A，保初始世界锚及手工局部帧，D继续接C；不造B模拟体，不修改作者骨架/参与设置。实际前端导出与手工frame保持通过；同0.15 s生产CUDA短链A/C/D下降约0.10221 m，A/C差4.75e-7 m，10有限帧，自然exit0，未编引擎。
- 显示以本帧realized祖先、冻结basis与convert_local_to_pose逐级虚拟FK，避免直接读旧B.matrix的一帧滞后。无物理祖先仍保原运动学根路径；关闭自动跟随仍为独立自由根。相对祖先变化的中间动画、特殊继承/约束/驱动当前接口无法正确表达，限定分支明确提示，不静默丢动画。端点相抵但内部变化也在编码阶段拒绝，唯一新增实测恢复frame3/subframe.25且exit0。
- 本轮7产品文件：new_binding_defaults.py、bone_selection_parameters.py、ops/bone_proxy.py、core/encoder/bones.py、core/encoder/automatic_parent_anchors.py、core/rigid_pose.py、frontend/_automatic_parent_anchor_build_.py。无引擎修改/原生重编/用户原blend或磁盘缓存写入。短链证据不是用户场景或复杂长链整体验收。
- 证据分别为density-500mpa-followup、first-recreate-stability、middle-bone-participation-20261025。用户剩余13分钟要求收口，停止扩展审查/测试；最后仅真实ZIP安装后复用新增直接探针一次，现有安装包与源码分享包原位更新。

- 本轮7文件已原位发布到5.1安装目录、canonical完整运行包前端、原v2安装ZIP与桌面既有源码分享包。最终ZIP在全新私有Blender5.1.2真实ExtensionManager安装启用后：默认6项、中段producer/FK8项及真实Untitled副本删建/手改尺寸/来源回滚检查全部PASS，natural_exit_code=0（6.72秒）；证据plugin-cloud-merge-20261025/installer-recreate-middle-density-20261025/RESULT.json。整轮最终senior PASS，未追加大矩阵。用户保存后正常重启Blender与本地PPF服务并重新传输运行，现有旧模拟输出不自动重算。
