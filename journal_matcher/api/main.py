"""
期刊论文语义匹配系统 - FastAPI 应用入口
====================================

功能说明：
- 定义RESTful API接口
- 处理HTTP请求和响应
- 提供系统初始化、检索、健康检查等功能

API端点：
- GET  /health              - 健康检查
- GET  /api/journals        - 获取期刊列表
- POST /api/journals        - 添加期刊
- GET  /api/journals/{issn} - 获取期刊详情
- GET  /api/journals/{issn}/papers - 获取期刊论文
- POST /api/init            - 初始化系统（抓取论文+向量化）
- POST /api/search          - 执行语义检索

作者：AI Assistant
日期：2026-05-16
"""

from typing import List, Optional
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ..models.schemas import (
    Journal, JournalCreate, Paper, PaperSummary,
    SearchQuery, SearchResponse, InitRequest, InitResponse,
    JournalListResponse, PaperListResponse, HealthResponse, ErrorResponse
)
from ..services.database import DatabaseService
from ..services.journal import JournalService, get_journal_service
from ..services.embedding import get_embedding_service
from ..services.search import SearchService, get_search_service, reset_search_service
from ..config import config


# =============================================================================
# 初始化状态
# =============================================================================

# 全局服务实例（延迟初始化）
_db: Optional[DatabaseService] = None
_journal_service: Optional[JournalService] = None
_search_service: Optional[SearchService] = None

# 初始化状态标志
_initialized = False
_initializing = False


def get_db() -> DatabaseService:
    """获取数据库服务实例"""
    global _db
    if _db is None:
        _db = DatabaseService()
    return _db


def get_journal_svc() -> JournalService:
    """获取期刊服务实例"""
    global _journal_service
    if _journal_service is None:
        _journal_service = get_journal_service()
    return _journal_service


def get_search_svc() -> SearchService:
    """获取检索服务实例"""
    global _search_service
    if _search_service is None:
        _search_service = get_search_service()
    return _search_service


# =============================================================================
# FastAPI 应用
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理

    在应用启动时执行初始化：
    1. 加载向量化模型
    2. 连接数据库
    3. 验证系统状态
    """
    global _initialized, _db, _search_service

    print("=" * 60)
    print("期刊论文语义匹配系统启动中...")
    print("=" * 60)

    try:
        # 初始化数据库
        _db = DatabaseService()
        print("[OK] 数据库连接成功")

        # 预加载向量化模型（可选，加速首次检索）
        # 这会占用一些内存，但可以减少首次检索的延迟
        # embedding = get_embedding_service()
        # print(f"[OK] 向量化模型加载成功: {embedding.model_name}")

        _initialized = True
        print("=" * 60)
        print("系统启动完成！")
        print("=" * 60)

    except Exception as e:
        print(f"[ERROR] 系统启动失败: {e}")
        raise

    yield

    # 应用关闭时清理资源
    print("系统关闭中...")


# 创建FastAPI应用
app = FastAPI(
    title="期刊论文语义匹配系统",
    description="""
## 系统介绍

这是一个能够**抓取指定期刊最新一期论文**、
通过**语义匹配**判断用户输入主题是否存在相似论文的系统。

## 核心功能

1. **期刊管理**：添加和管理要检索的期刊
2. **论文抓取**：从OpenAlex、CrossRef等开放API获取论文
3. **语义检索**：使用多语言向量化模型实现跨语言检索
4. **结果展示**：按期刊分组展示相似论文

## 技术架构

- **向量化模型**：BGE-M3（多语言向量化，支持中英文跨语言检索）
- **向量索引**：FAISS（高效相似度检索）
- **数据存储**：SQLite（论文元数据）+ 向量缓存

## 使用流程

1. 调用 `/api/init` 初始化系统（抓取论文+向量化）
2. 调用 `/api/search` 输入主题进行检索
3. 查看结果判断是否存在相似论文
    """,
    version="1.0.0",
    lifespan=lifespan
)

# 添加CORS中间件（允许前端跨域访问）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# 健康检查接口
# =============================================================================

@app.get("/health", response_model=HealthResponse, tags=["系统"])
async def health_check():
    """
    健康检查接口

    返回系统的运行状态，包括：
    - 数据库连接状态
    - 向量索引状态
    - 向量化模型状态
    """
    try:
        db = get_db()
        embedding = get_embedding_service()

        return HealthResponse(
            status="healthy",
            version=config.VERSION,
            database_connected=True,  # 如果连接失败会抛出异常
            vector_index_ready=db.is_index_ready(),
            embedding_model_ready=embedding.is_ready()
        )
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            version=config.VERSION,
            database_connected=False,
            vector_index_ready=False,
            embedding_model_ready=False
        )


# =============================================================================
# 期刊管理接口
# =============================================================================

@app.get("/api/journals", response_model=JournalListResponse, tags=["期刊管理"])
async def list_journals():
    """
    获取期刊列表

    返回所有已配置的期刊
    """
    db = get_db()
    journals = db.get_all_journals()
    return JournalListResponse(journals=journals, total=len(journals))


@app.post("/api/journals", response_model=Journal, tags=["期刊管理"])
async def create_journal(journal: JournalCreate):
    """
    添加新期刊

    Args:
        journal: 期刊配置（ISSN、名称等）
    """
    db = get_db()

    # 检查是否已存在
    existing = db.get_journal(journal.issn)
    if existing:
        raise HTTPException(status_code=400, detail=f"期刊 {journal.issn} 已存在")

    # 创建期刊对象
    new_journal = Journal(
        issn=journal.issn,
        name=journal.name,
        country=journal.country,
        description=journal.description
    )

    db.save_journal(new_journal)
    return new_journal


@app.delete("/api/journals/{issn}", response_model=dict, tags=["期刊管理"])
async def delete_journal(issn: str):
    """
    删除期刊

    Args:
        issn: 期刊ISSN
    """
    db = get_db()
    journal = db.get_journal(issn)

    if not journal:
        raise HTTPException(status_code=404, detail=f"期刊 {issn} 不存在")

    # 注意：这里只是从期刊表中删除，不删除关联的论文
    with db._get_connection() as conn:
        conn.execute("DELETE FROM journals WHERE issn = ?", (issn,))
        conn.commit()

    return {"message": f"期刊 {issn} 已删除"}


# =============================================================================
# 论文获取接口
# =============================================================================

@app.get("/api/journals/{issn}/papers", response_model=PaperListResponse, tags=["论文管理"])
async def get_journal_papers(issn: str):
    """
    获取指定期刊的论文列表

    Args:
        issn: 期刊ISSN

    Returns:
        论文列表
    """
    db = get_db()
    journal = db.get_journal(issn)

    if not journal:
        raise HTTPException(status_code=404, detail=f"期刊 {issn} 不存在")

    papers = db.get_papers_by_journal(issn)
    paper_summaries = [PaperSummary.from_paper(p) for p in papers]

    return PaperListResponse(
        journal=journal,
        papers=paper_summaries,
        total=len(paper_summaries)
    )


# =============================================================================
# 系统初始化接口（核心接口）
# =============================================================================

class InitStatus(BaseModel):
    """初始化状态"""
    status: str
    message: str
    progress: float = 0.0


@app.post("/api/init", response_model=InitResponse, tags=["系统"])
async def initialize_system(request: InitRequest, background_tasks: BackgroundTasks = None):
    """
    初始化系统

    这是核心接口，执行以下步骤：
    1. 添加期刊配置
    2. 从API抓取论文
    3. 对论文进行向量化
    4. 构建FAISS索引

    **建议的使用方式**：
    - 面试开始时调用此接口
    - 系统在后台完成初始化（约需1-2分钟）
    - 之后可以反复调用 `/api/search` 进行检索

    Args:
        request: 初始化配置
            - use_default: 是否使用默认期刊列表
            - journal_configs: 自定义期刊列表

    Returns:
        初始化结果统计
    """
    global _initialized, _search_service

    if _initialized and get_db().get_papers_count() > 0:
        return InitResponse(
            status="already_initialized",
            message="系统已经初始化，跳过",
            journals_count=len(get_db().get_all_journals()),
            papers_count=get_db().get_papers_count(),
            vectorized_count=get_db().get_index_size()
        )

    db = get_db()
    journal_svc = get_journal_svc()

    errors = []
    papers_count = 0
    vectorized_count = 0

    # 确定要初始化的期刊列表
    if request.use_default or not request.journal_configs:
        # 使用默认期刊列表
        journal_configs = []
        for country, journals in config.DEFAULT_JOURNALS.items():
            for j in journals:
                journal_configs.append({
                    "issn": j["issn"],
                    "name": j["name"],
                    "country": country,
                    "description": j.get("description", "")
                })
    else:
        journal_configs = request.journal_configs

    # 保存期刊配置
    journals = []
    for cfg in journal_configs:
        journal = Journal(
            issn=cfg["issn"],
            name=cfg["name"],
            country=cfg.get("country", "INT"),
            description=cfg.get("description", "")
        )
        db.save_journal(journal)
        journals.append(journal)

    print(f"已配置 {len(journals)} 本期刊")

    # 抓取论文
    all_papers = []
    for i, journal in enumerate(journals):
        print(f"正在抓取期刊 {i+1}/{len(journals)}: {journal.name}...")

        try:
            papers, source = journal_svc.fetch_papers(journal, max_papers=50)
            print(f"  - 获取到 {len(papers)} 篇论文 (来源: {source})")

            # 保存论文到数据库
            if papers:
                db.save_papers_batch(papers)
                all_papers.extend(papers)
                papers_count += len(papers)

            # 更新期刊抓取时间
            db.update_journal_fetch_time(journal.issn)

        except Exception as e:
            error_msg = f"抓取期刊 {journal.name} 失败: {str(e)}"
            print(f"[ERROR] {error_msg}")
            errors.append(error_msg)

    print(f"论文抓取完成，共 {papers_count} 篇")

    # 向量化索引
    if all_papers:
        print("开始向量化...")
        try:
            # 确保向量化模型已加载
            embedding = get_embedding_service()
            if not embedding.is_ready():
                raise RuntimeError("向量化模型加载失败")

            # 创建检索服务并索引论文
            _search_service = SearchService(db, embedding)
            vectorized_count = _search_service.index_papers(all_papers)
            print(f"向量化完成，共 {vectorized_count} 篇")

        except Exception as e:
            error_msg = f"向量化失败: {str(e)}"
            print(f"[ERROR] {error_msg}")
            errors.append(error_msg)

    _initialized = True

    return InitResponse(
        status="success" if not errors else "partial",
        message="初始化完成" if not errors else "部分失败",
        journals_count=len(journals),
        papers_count=papers_count,
        vectorized_count=vectorized_count,
        errors=errors
    )


@app.post("/api/init/reset", response_model=dict, tags=["系统"])
async def reset_system():
    """
    重置系统

    清空所有数据，重新开始
    """
    global _initialized, _search_service

    db = get_db()
    db.clear_all_data()

    reset_search_service()
    _initialized = False

    return {"message": "系统已重置"}


# =============================================================================
# 语义检索接口（核心接口）
# =============================================================================

@app.post("/api/search", response_model=SearchResponse, tags=["检索"])
async def search_papers(query: SearchQuery):
    """
    语义检索

    这是核心功能接口。用户输入检索主题，系统返回各期刊中是否存在相似论文。

    **技术原理**：
    1. 将用户查询转换为向量
    2. 在FAISS向量索引中检索相似论文
    3. 按期刊分组，应用相似度阈值
    4. 返回每本期刊的检索结果

    **跨语言检索**：
    - 系统使用BGE-M3多语言向量化模型
    - 中文查询可以直接匹配英文论文
    - 无需翻译，直接在统一向量空间中检索

    Args:
        query: 检索查询
            - query: 检索主题（如"深度学习在医学影像中的应用"）
            - top_k: 每期刊返回的最大论文数（默认5）
            - threshold: 相似度阈值（默认0.5，范围0.0~1.0）

    Returns:
        检索结果，按期刊分组
    """
    global _search_service

    if not _initialized:
        raise HTTPException(
            status_code=400,
            detail="系统未初始化，请先调用 /api/init 接口"
        )

    db = get_db()

    # 检查是否有索引数据
    if not db.is_index_ready() or db.get_index_size() == 0:
        raise HTTPException(
            status_code=400,
            detail="没有可检索的论文数据，请先调用 /api/init 初始化"
        )

    # 确保检索服务已初始化
    if _search_service is None:
        _search_service = get_search_service()

    # 执行检索
    response = _search_service.search(
        query=query.query,
        threshold=query.threshold,
        top_k=query.top_k
    )

    return response


@app.get("/api/statistics", response_model=dict, tags=["系统"])
async def get_statistics():
    """
    获取系统统计信息

    返回系统当前状态的各种统计指标
    """
    if not _initialized:
        return {
            "initialized": False,
            "message": "系统未初始化"
        }

    db = get_db()
    journals = db.get_all_journals()

    # 各期刊论文数量
    journal_stats = []
    for journal in journals:
        count = db.get_journal_papers_count(journal.issn)
        journal_stats.append({
            "issn": journal.issn,
            "name": journal.name,
            "papers_count": count
        })

    return {
        "initialized": True,
        "total_journals": len(journals),
        "total_papers": db.get_papers_count(),
        "indexed_papers": db.get_index_size(),
        "journals": journal_stats
    }


# =============================================================================
# 错误处理
# =============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """HTTP异常处理器"""
    return ErrorResponse(
        error="HTTPError",
        message=str(exc.detail),
        details={"status_code": exc.status_code}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """通用异常处理器"""
    return ErrorResponse(
        error="InternalError",
        message="服务器内部错误",
        details={"exception": str(exc)}
    )


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    # 启动FastAPI服务
    uvicorn.run(
        "journal_matcher.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True  # 开发模式：代码修改后自动重载
    )
