# 期刊论文语义匹配系统

## 系统简介

这是一个能够**抓取指定期刊论文**、并通过**语义匹配**判断用户输入主题是否存在相似论文的系统。

**核心功能**：
- 从多个学术数据库（OpenAlex、CrossRef、Semantic Scholar）获取国内外期刊论文
- 内置 UTD 24 本国际顶级期刊 + 5 本中文经管类期刊配置
- 使用多语言向量化模型（BGE-M3）实现中英文跨语言检索
- 通过 FAISS 向量索引实现快速语义匹配
- **年份过滤**确保检索结果时效性
- 支持 RESTful API 和命令行两种使用方式

## 面试场景使用说明

### 题目背景

现场 AI 编程实现：抓取国内外各5本期刊的论文，然后用户输入想写的论文主题，需要系统后台通过语义匹配，告知用户这10本期刊是否存在相似主题的论文信息。

### 核心难点与解决方案

| 难点 | 解决方案 |
|------|----------|
| 国外期刊访问困难 | 使用 OpenAlex、CrossRef、Semantic Scholar 等开放 API，不直接爬取期刊网站 |
| 中文期刊国际数据库收录有限 | Semantic Scholar（AI驱动）+ OpenAlex名称搜索 + 手动导入 |
| API 限流 (429错误) | 请求间隔控制 + 重试机制 + 多数据源降级 |
| 跨语言检索（中查英/英查中） | BGE-M3 多语言向量化模型（无需翻译） |
| 检索结果不相关 | 年份过滤（近5年）+ 未来日期过滤 + 相似度阈值 |
| 论文数据不完整 | 多源 API 交叉补全、降级策略 |
| 面试现场快速实现 | 模块化架构、降级策略、本地缓存 |

### 面试现场使用流程

```
1. 面试官公布10本期刊名称和ISSN
2. 输入期刊配置（或使用默认配置）
3. 运行初始化命令
4. 用户输入研究主题进行检索
```

#### Step 1: 初始化系统

```bash
python main.py init
```

初始化过程：
1. 连接数据库（SQLite）
2. 加载向量化模型（BGE-M3）
3. 配置期刊（UTD 24 本国际 + 5 本中文）
4. 抓取各期刊论文
5. 对论文进行向量化索引

#### Step 2: 检索相似论文

```bash
python main.py search "深度学习在医学影像中的应用"
```

#### Step 3: 导入中文论文（可选）

```bash
# 生成示例数据
python -m journal_matcher.services.import_papers sample

# 导入真实数据
python -m journal_matcher.services.import_papers import chinese_papers.json
```

#### Step 4: 启动 Web 服务（可选）

```bash
python main.py serve
# 访问 http://localhost:8000/docs 查看API文档
```

## 项目结构

```
journal_matcher/
├── __init__.py              # 包初始化
├── config.py                # 配置文件（含UTD 24期刊）
├── cli.py                   # 命令行界面
├── main.py                  # 主入口
├── models/
│   ├── __init__.py
│   └── schemas.py           # Pydantic 数据模型
├── services/
│   ├── __init__.py
│   ├── database.py          # 数据库服务（SQLite + FAISS）
│   ├── embedding.py         # 向量化服务（BGE-M3）
│   ├── journal.py           # 期刊论文获取服务（国际）
│   ├── chinese_journal.py   # 中文期刊获取服务（新增）
│   ├── search.py            # 语义检索服务
│   ├── translation.py       # 翻译服务（可选）
│   └── import_papers.py     # 手动导入论文（新增）
└── api/
    ├── __init__.py
    └── main.py              # FastAPI 应用入口
```

## 快速开始

### 1. 安装依赖

```bash
conda activate mian
pip install -r requirements.txt
```

### 2. 初始化系统

```bash
python main.py init
```

### 3. 执行检索

```bash
python main.py search "深度学习在医学影像中的应用"
```

### 4. 启动 API 服务（可选）

```bash
python main.py serve
```

## 技术架构

### 检索流程

```
用户输入主题
    │
    ▼
┌─────────────────────────────────────────┐
│  步骤1: 语言检测                        │
│  - 自动检测中文/英文                    │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  步骤2: 向量召回 (FAISS)                │
│  - BGE-M3 向量化（中英文统一向量空间）   │
│  - 检索 Top-50 候选                     │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  步骤3: 过滤                            │
│  - 年份过滤：只保留近5年论文            │
│  - 未来日期过滤：排除预发表论文         │
│  - 相似度阈值：低于0.1的结果被过滤     │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  步骤4: 按期刊分组，返回 Top-5          │
└─────────────────────────────────────────┘
```

### 中文期刊数据获取策略

```
┌─────────────────────────────────────────┐
│  Layer 1: Semantic Scholar (优先)        │
│  - AI驱动学术搜索                      │
│  - 收录范围广，包括中文期刊            │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  Layer 2: OpenAlex 名称搜索             │
│  - 扩大年份范围                        │
│  - fuzzy 匹配期刊名称                  │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  Layer 3: CrossRef                      │
│  - 按期刊名查询ISSN                    │
│  - 按ISSN获取论文                      │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  Layer 4: 手动导入（兜底）              │
│  - 支持JSON/CSV格式                    │
└─────────────────────────────────────────┘
```

### 核心论文参考

| 论文 | 关键贡献 | 应用 |
|------|----------|------|
| M3-Embedding (arXiv:2402.03216) | 多语言、多功能、多粒度向量化 | 跨语言检索基础 |
| CLIRudit (ACL 2024) | 稠密检索无需翻译 | 验证方案可行性 |

详见 `00_研究调研报告.md`

## UTD 24 本国际顶级期刊

系统内置以下 UTD 24 本期刊配置：

| 领域 | 期刊 |
|------|------|
| **金融 (4)** | Journal of Finance (JF), Journal of Financial Economics (JFE), Review of Financial Studies (RFS), Journal of Financial and Quantitative Analysis (JFQA) |
| **会计 (4)** | Accounting Review (AR), Journal of Accounting and Economics (JAE), Journal of Accounting Research (JAR), Accounting, Organizations and Society (AOS) |
| **管理科学 (4)** | Management Science (MS), Administrative Science Quarterly (ASQ), Academy of Management Journal (AMJ), Strategic Management Journal (SMJ) |
| **运营管理 (2)** | Operations Research (OR), Manufacturing & Service Operations Management (M&SOM) |
| **信息系统 (2)** | MIS Quarterly (MISQ), Information Systems Research (ISR) |
| **市场营销 (3)** | Journal of Marketing (JM), Journal of Marketing Research (JMR), Journal of Consumer Research (JCR) |
| **组织行为 (2)** | Academy of Management Review (AMR), Organization Science (OS) |
| **经济学 (3)** | American Economic Review (AER), Quarterly Journal of Economics (QJE), Journal of Political Economy (JPE) |

## 面试应答要点

### Q: 如何解决访问国外期刊困难的问题？

**答**：不直接爬取期刊网站，使用开放学术 API：
- **OpenAlex API**：覆盖2亿+学术文献，免费访问
- **CrossRef API**：DOI 官方注册机构，数据权威
- **Semantic Scholar API**：AI驱动学术搜索，收录范围更广

### Q: 如何解决中文期刊数据获取困难？

**答**：采用多层降级策略：
1. **Semantic Scholar**：AI驱动，收录大量中文期刊
2. **OpenAlex 名称搜索**：扩大年份范围，模糊匹配期刊名
3. **CrossRef**：按期刊名查询ISSN，再获取论文
4. **手动导入**：支持JSON/CSV格式导入真实数据

### Q: 如何实现中英文跨语言检索？

**答**：使用 BGE-M3 多语言向量化模型：
- 原生支持100+语言，在统一向量空间中编码中英文
- 无需翻译，直接在向量空间匹配语义
- 例如：中文查询"深度学习"可以直接匹配英文论文"Deep learning takes on protein folding"

### Q: 如何确保检索结果的相关性？

**答**：多层过滤保障：
- **年份过滤**：只保留近5年论文
- **未来日期过滤**：排除预发表论文
- **相似度阈值**：低于0.1的结果被过滤
- **向量语义匹配**：BGE-M3 确保语义相关性

### Q: 系统的局限性？

**答**：
1. 依赖外部 API 可用性
2. 中文期刊在国际数据库中数据较少（已通过多层降级策略缓解）
3. 部分论文数据不完整（无摘要）
4. 学术检索不等于查重

## 相关文档

- `00_研究调研报告.md` - 详细的技术调研和论文参考
- `01_系统设计文档.md` - 完整的系统架构设计
- `02_实现总结.md` - 代码实现总结

---

*作者：AI Assistant | 日期：2026-05-16*
