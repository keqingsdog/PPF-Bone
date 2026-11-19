# 来源与分支说明

本项目基于 ZOZO / st-tech 的 `ppf-contact-solver` 开发。原项目地址：<https://github.com/st-tech/ppf-contact-solver>。

本分支维护者为 `keqingsdog`。保留 Ryoichi Ando 及上游项目贡献者的原有署名。

本地工程的基线来自 2026-09-27 官方源码包。骨骼/关节、相关求解扩展、Blender 骨骼代理与映射、预设、显示和操作流程包含本分支后续修改；这些改动不是上游官方发行承诺。上游原文档、文件头和 `LICENSE` 保留。

根目录 `LICENSE` 为项目携带的 Apache License 2.0 文本。第三方依赖继续使用各自随附的许可说明。源码仓库不包含整套 CUDA、Python、Blender 或 BoneX 插件；BoneX 仅作为操作流程参考，不作为本仓库可重新编译的依赖。

本机路径配置、测试工程、运行缓存和凭据不属于发布源码。平台工具链及生成产物应由接手方按构建说明准备。

对不能安全添加注释的既有上游文件，相邻 `.license` 文件记录原项目许可来源及本分支修改；这些旁注不改变原文件格式，也不表示整个项目已通过 REUSE 认证。
