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
       │  electron zip)   │  │                  │
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
- 检测版本号是否变化

**阶段 3: 构建与发布**
- 版本变化时同时进行:
  - 构建多平台发行版 (Windows + Linux)
  - 构建 Electron GUI ZIP (Windows + Linux)
  - 发布到 PyPI
- 最后创建 GitHub Release

> `pytest` 仅在 diff 中包含 `.py` 文件时才会运行；纯文档 / 资源改动会跳过。
> 只有 push 到 main 且 `vexor/__init__.py` 的版本号改变时，才会进入
> 构建可执行 / 发布 GitHub Release / 推送 PyPI 的阶段；否则直接结束。
