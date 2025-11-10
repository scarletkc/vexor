# HEAD 模式推进计划

## 1. 内容提取基座
- [x] 新建 `vexor/services/content_extract_service.py` 并提供 `extract_head(path: Path, char_limit: int) -> str | None` 接口。
- [x] 定义常量 `HEAD_CHAR_LIMIT`（暂定 1000）和公共的编码探测/清洗工具。
- [x] 设计扩展名到解析器的注册表（例如 `.txt`, `.md`, `.py` 默认走纯文本读取；未匹配则返回 `None`）。

## 2. 策略/模式接入
- [x] 在 `vexor/modes.py` 中新增 `HeadStrategy`，调用 `extract_head`，并在缺失内容时回退到 `NameStrategy`。
- [x] 更新 `available_modes()` 输出包含 `head`，CLI `--mode` 校验允许该值。
- [x] 在索引流程中（`index_service`) 使用 `HeadStrategy` 的标签构建 embeddings。

## 3. 解析器实现
- [x] 纯文本/代码：实现通用 `_read_text_file()`（带编码探测/去 BOM），用于 `.txt`, `.md`, `.py`, `.js`, `.json`, `.yaml` 等。
- [ ] Markdown：可复用纯文本读取，只需去掉多余 markdown 语法（后续可接入 `markdown-it`）。
- [ ] HTML：引入 `beautifulsoup4` + `lxml` 解析，提取 `<body>` 前若干文字。
- [ ] PDF：使用 `pypdf` 或 `pdfplumber` 读取前 1-2 页。
- [ ] DOCX：使用 `python-docx` 提取前几段；后续视需求扩展 PPTX/XLSX。
- [ ] 统一在超出 `HEAD_CHAR_LIMIT` 时截断，并保留简单清洗（去多余空行、控制符）。

## 4. CLI/文档
- [x] README “Workflow/Commands/Tips” 中说明 `--mode head` 行为和成本考量。
- [x] 在 `vexor/text.py` 加入针对 `head` 模式的帮助文本提示。

## 5. 测试
- [x] 单元测试：`tests/unit/test_modes.py` 验证 `HeadStrategy` 对纯文本/空内容的处理和 fallback 逻辑。
- [x] 内容提取测试：针对每种文件类型提供小样本，确认 `extract_head` 输出是否符合预期（当前覆盖文本/代码）。
- [ ] 集成测试：创建包含头部文本的临时文件，运行 `vexor index --mode head` + `vexor search --mode head`，确保可命中。

## 6. 发布前检查
- [x] 确认新的依赖写入 `pyproject.toml`/`requirements.txt` 并在 README 中注明。
- [x] 执行 `pytest` 全量回归，确认 `name` 模式不受影响。
- [ ] 若 `CACHE_VERSION` 需再次变更，同步更新迁移说明（目前为 3）。
