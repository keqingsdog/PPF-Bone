# Linux累计修复的Windows接入（2026-10-27）

输入为用户提供的`PPF_返还源码后累计修复_20261026.zip`。按其中BASE／FINAL对当前源码进行函数级三方迁移，没有覆盖旧的完整插件快照。当前双栏／经典界面、语言偏好、组内16层碰撞、跨链、e9 Capture、材质默认值和父级映射均保留。

## 功能变化

- CUSTOM操作拒绝后恢复恢复力界面的tier／compat值及显式／未设置状态，不全局抑制RNA回调。
- 摩擦批量输入与活动材质的动画关键帧装饰器分开，两个入口有中文名称。
- 插件增加纯序列化跨链作者契约helper，插件及前端在允许`cross_links_v1`时仍严格检查字段、身份、端点及数值。
- 骨骼缓存刷新先正常评估当前作者Action再回放缓存；无缓存的持久记录恢复后也评估当前Action，保留静态回退。忙碌／暂停时保留待刷新项并稍后重试，不提前丢弃队列。
- 碰撞基线使用`Scene.session_uid`跟踪Scene身份，仍保留真正碰撞变化的输出失效和失败重试。
- Linux本地源码启动脚本使用LF，Windows与旧发布路线保留CRLF；没有在Windows上宣称Linux实际启动通过。
- 原生修复仅限raw／PSD Hessian分量写入binary32的边界：正常主尺度的张量中，不超过最小subnormal一半的有限分量按最近偶数舍入到带符号零；raw和PSD分别检查自己的尺度。能量、梯度、主特征值和通用转换仍严格，NaN／Inf／溢出继续拒绝。不保证逐分量舍入后数学意义上的严格PSD。

## 实际验证

Windows Blender5.1.2私有配置已验证CUSTOM恢复、正常／拒绝操作、跨链作者契约、摩擦批量／作者关键帧、持久记录保存重开以及忙碌刷新队列。真实磁盘CBOR缓存的同帧Undo／Redo及碰撞变化负控另有回归；并非完整Transfer／Run后的所有缓存Undo验收。

必须纠正的一处早期归因：初次cache-window07清除缓存的直接触发是测试初始化留下的`None`碰撞基线，不是已观测到的Scene指针变化。本轮没有实测Undo分配新指针的分支；`session_uid`作为API提供的稳定身份接线保障。旧原生AV崩溃及完整PC2磁盘Undo仍未据此关闭。

最终插件候选以现有正式安装包仅替换13个命名文件，在独立Blender中真实安装。20项代表检查通过，自然退出0：新增helper从实际安装目录加载，无源码fallback；中文摩擦两行和组内表可见，无旧摩擦decorator警告，骨入口可见且独立碰撞模块隐藏。

前端4项定向测试覆盖合法／非法输入及16种启动器换行组合。原生portable host回归含22项及真实`evaluate_binding_f32`与错误尺度负控；Windows使用`/O2 /fp:precise /fp:contract /arch:AVX2`匹配生产FMA配置。回归源码在`crates/ppf-cts-solver/tests/restoring_tensor_rounding/run_host.py`及对应`tests/kernels/restoring_tensor_rounding.cpp`。

本机CPU和CUDA均通过官方`tools/build_source.py --phase all`增量构建；CUDA保留sm61／75／86／89／90／100／120及原单精度守卫。没有删除CPU／Metal／ROCm路线，也没有抄入Linux生成身份；必要身份由当前源码的原生成器重新生成。

CPU中心及非对称两例均自然结束，约0.205秒物理时间、11有限帧、41接受步，kind5非零装配与history推进均有真实receipt。CUDA对应两例也自然结束，RTX4070TiSUPER实际设备、相同帧／时间／装配／history判据通过；单独一次自有进程观察确认加载私有新CUDA后端。短程两例不等于复杂场景或连续无穿透的普遍保证。

## 交付边界

本轮只更新命名插件／前端／原生源文件及配套CPU／CUDA自产物，沿用已有运行环境和安装包。已知旧Undo崩溃、多缓存Scene、全部历史场景和其他平台没有在本轮重新宣称通过。源码继续是9/27官方结构上的增量分支；新机器应按`BUILDING_BONES.md`构建，不依赖本机target或历史对象文件。

界面紧急修正详见`PLUGIN_UI_DISPLAY_FIX_20261027.md`。本轮集中最终审查为PASS，结论限于上述源码与直接功能范围。总控按已审命名覆盖与配对产物方案发布；部署和同名云盘替换各记录实际执行结果，未执行的更新不能算完成。
