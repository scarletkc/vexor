# Vexor CI/CD 流程图

## 完整流程
```
                    ┌────────────────────┐
                    │  Push/PR to main   │
                    └──────────┬─────────┘
                               │
                               ▼
                    ┌─────────────────────────┐
                    │ any *.py files changed? │
                    └──────────┬──────────────┘
                               │
                     ┌─────────┴─────────┐
                   yes                   no
                     │                    │
                     ▼                    ▼
          ┌────────────────────┐    ┌─────────────────┐
          │   pytest (linux)   │    │ skip pytest job │
          └──────────┬─────────┘    └─────────────────┘
                     │
                     ▼
          ┌─────────────────────────┐
          │ push to main & codecov  │
          │   tests pass?           │
          └──────────┬──────────────┘
                     │
                     ▼
          ┌─────────────────────────┐
          │prepare_release (detect) │
          │   version changed?      │
          └──────┬──────────────┬───┘
                 │              │
               yes            yes
                 │              │
                 ▼              ▼
       ┌──────────────────┐  ┌──────────────────┐
       │  build_release   │  │  publish_pypi    │
       │ (pyinstaller +   │  │ (build + upload) │
       │  GUI zip/reuse)  │  │                  │
       └────────┬─────────┘  └──────────────────┘
                │
                ▼
       ┌───────────────────────────┐
       │  publish_release          │
       │  (gh-release)             │
       └───────────────────────────┘
```

## 流程说明

**阶段 1: 代码检查与测试**
- 触发: Push 或 PR 到 main 分支
- 检查 Python 文件是否有变更
- 有变更 → 运行测试 (Ubuntu)
- 无变更 → 跳过测试

**阶段 2: 发布准备**
- 条件: push 到 main 且测试通过
- 检测版本号是否变化 (以 `vexor/__init__.py` 的 `__version__` 与上一提交相比为准)

**阶段 3: 构建与发布**
- 版本变化时同时进行:
  - 构建多平台发行版 (Windows + Linux)
  - GUI 版本号相对上一版 Release 有变化时构建 Electron GUI ZIP (Windows + Linux)
  - GUI 版本号未变化时复用上一次 Release 的 GUI 产物并跳过 Electron 构建
  - 发布到 PyPI
- 最后创建 GitHub Release

> `pytest` 仅在 diff 中包含 `.py` 文件时才会运行；纯文档 / 资源改动会跳过。
> 只有 push 到 main 且版本号改变时，才会进入构建可执行 / 发布 GitHub Release /
> 推送 PyPI 的阶段；否则直接结束。GUI 产物命名使用 GUI 版本号。
