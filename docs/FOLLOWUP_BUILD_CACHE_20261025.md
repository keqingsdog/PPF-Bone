# 构建、缓存与切场景补修（2026-10-25）

本页对应Linux端随后发来的四项静态反馈；结论来自当前源码的定向检查，不把静态审查当原生验收。上一轮绑定/真拖帧修复见[公开摘要](USER_REPORTED_PUBLIC_SUMMARY_20261030.md)。

## 构建调度

- 官方9/27 Linux CUDA链接原本使用`-t 0`，即按全部CPU并行；本分支新增`--jobs`后未一起约束它，确实会突破外层预算。现链接改为`-t 1`，Windows原本已是1，不重复修改。
- 无CUDA、非macOS时，旧`--phase backend --backend auto`能只检查身份就返回0，没有实际构建。隔离CLI已复现。
- 保留官方auto选择语义：无可用CUDA/Metal时明确非0并提示显式`--backend cpu`；显式CPU真实调度Cargo，显式CUDA缺SDK明确拒绝，不静默切后端。
- macOS Metal、Linux ROCm和原平台入口保留；允许的backend阶段执行实际命令或明确报错。新增`tools/test_build_backend_dispatch.py`三组定向测试通过，链接参数用无害编译器替身执行验证。
- 这些是CLI、命令及替身测试，没有Linux实际编译或新GPU求解；调度修复不需要重编已完成的Windows物理产物。

## 已保存工程的Clear All

- 私有Blender5.1.2保存一个包含A/B场景的工程，原“清理所有变形”在A删除了B的静态变形与Pin文件和内存，已实证。
- 现在只处理当前Scene有确切归属的捕获key；新捕获记录实际owner/key，已删除owner可按确证记录清理。逐对象入口与Clear All共用归属门，内存缓存也按key处理。
- 共享对象、复制/重复归属和未知孤儿不猜删；文件删除失败时保留相应内存与归属供重试。
- 同一个保存工程定向修后：A正常清理，B的文件、内存、骨骼索引/控制与会话状态保留。测试未运行新物理解算，不外推为任意多工程组合验收。

## 提交后切Scene

- 切Scene取消的是当前本地异步操作，原本没有发送服务器Stop，旧提示容易被误解为远端任务已停止。
- 现在区分提交前和提交后：尚未提交则明确取消；已发起请求则提示本地操作结束、未发送停止请求，应返回原场景查看状态。异步排队不冒充远端已确认运行。
- 用户切Scene不等于请求终止计算；需要真正停止时使用明确的Stop操作。后续编码、上传、清理与完成回调不会交给新场景。
- 真实Transfer入口/阶段及UpdateParams直接提交路径，在编码/网络替身下测试通过：未向B执行后续回调或清理，未调用Stop。本项不是新的远端求解验收。

三个生命周期文件为`blender_addon/core/pc2.py`、`core/async_op.py`、`ui/solver.py`，已定向部署至正式R9、安装插件及旧ABS04插件包；未热重载Blender、未改用户场景或连接配置。源码分享及配套v2插件由同一当前源码打包。同一senior对构建及生命周期两个限定里程碑最终PASS，不称整个项目无bug。

可从仓库直接复现的回归脚本已保留为`tests/blender/test_capture_scene_ownership.py`和`test_scene_submission.py`。它们用仓库相对导入和私有临时输出，在独立factory Blender5.1.2中实际退出0；第二项的编码与网络仍是替身，不冒称实际服务器测试。
