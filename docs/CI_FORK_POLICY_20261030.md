# Fork CI 的授权与验证边界

本文记录2026-10-30对当前工作流源码的定向检查，没有触发GitHub Actions、创建标签、上传发布或访问AWS。源码条件不能证明仓库外部设置已经配置正确。

## 主运行包与AWS工作流

`cards-win.yml`和`release.yml`仅由`workflow_dispatch`触发，顶层令牌权限为`contents: read`。AWS任务同时要求默认分支、显式`allow_aws=true`、仓库变量`PPF_ENABLE_AWS='true'`，并进入`ppf-cloud`环境；OIDC写权限仅在相应任务内授予。默认手工输入不授权AWS。

主`release.yml`的写标签／发布路径另要求`publish_release=true`、`PPF_ENABLE_RELEASES='true'`、`dry_run=false`及默认分支，进入`ppf-release`环境。`version`任务先检查显式发布意图；选择All、Linux或Windows真正发布时，没有AWS授权会先失败。`create-tag`依赖该任务成功，不能把“关闭AWS后仍会发布缺失平台”作为已证实可达故障。

当前创建标签不等待构建与验证任务完成。已授权运行若后续构建失败，可能遗留没有发布的标签；这是现有流程的维护边界，不代表未经授权的写入。未为此改动平台构建依赖图。

## 本次定向收紧的独立入口

示例AWS模板与已存在生成物、unit-tests的CUDA任务均增加默认false的allow_aws、默认分支、PPF_ENABLE_AWS和ppf-cloud门；CPU unit不受AWS门限制。重新运行示例生成器保留模板门控。

当前 Blender 扩展工作流只以 contents:read 构建并上传两种 ABI 的工作流产物，不执行发布。主 release 的可选扩展步骤只分派该构建工作流，不传发布参数；主 release 自身的授权门保持不变。

Docker入口的dry_run默认true，allow_aws、publish_image和allow_private_publish默认false。两个互斥调用共享完整EC2构建、验证和清理；只读构建调用不授packages:write，发布调用另需显式发布意图、PPF_ENABLE_PACKAGES=true及ppf-release环境，再在ppf-cloud执行。私有仓库以GitHub仓库属性判断，必须额外选allow_private_publish。以上为本地YAML与针对性正负控，尚未远程执行Actions，不代表外部环境审批、令牌与AWS信任设置已配置。

## 仓库维护者需要配置的外部条件

- 将AWS仓库变量和实际云角色配置为自己的账户；不要照抄原项目的授权关系。
- 为`ppf-cloud`和`ppf-release`设置需要的环境审批与允许分支。YAML写出环境名称不等于审批规则已生效。
- 将OIDC角色信任限制到实际仓库、分支或环境及需要的云操作，检查所用账户的最终权限。
- 实际发行前运行所选平台的构建、产物验证及发布演练。这里的静态检查不替代这些结果。

本次没有读取仓库变量值、环境保护设置或AWS信任策略，也没有访问令牌内容。上面的配置项不以本地源码检查冒充已验证。

## 后续定向入口

`getting-started.yml` 的AWS headless及两个run-all-once批量生成器各批使用显式allow_aws、默认分支、PPF_ENABLE_AWS与ppf-cloud；自托管headless任务保留原路径。两个生成器正常生成各五批。actionlint对这三个实际workflow定向检查通过，检查配置声明了原有自托管runner标签；这是本地静态检查，并未运行云CI。此前47项指42个Linux/Windows示例生成物加build-docker、docker-core、release-extension、release、unit-tests五项，不代表全仓53项均已测试。Blender云入口由独立工作处理，不根据这些测试外推。

## 检查范围说明

当前顶层共有53个工作流 YAML。上一轮 actionlint 定向47项为21个示例的 Linux 与 Windows 生成物共42项，加 build-docker、docker-core、release-extension、release、unit-tests 五项；本轮新增检查 getting-started、run-all-once、run-all-once-win、blender 四项。其余 cpu-acceptance 无 AWS／OIDC，make-docs 的 OIDC 用于既有 main／github-pages 发布而非 AWS；这两项本轮未重跑 lint。上述各轮检查不能表述为本轮重新检查全部53项。生成脚本实际位于 `.github/workflows/scripts`。本地静态检查不替代远程 CI、环境审批或云角色信任配置。
