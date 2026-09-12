# uvx 的环境与升级

uvx 创建的临时运行环境不会改变 Vexor 全局配置和模型的存储位置，它们仍保存在用户的 .vexor 目录中。

uvx 会缓存解析后的环境，不会自动升级。需要最新版本时可使用 vexor@latest，或使用 uvx --refresh vexor 显式刷新。vexor update --upgrade 面向 PATH 安装，不用于 uvx 环境。

来源：docs/mcp.md。
