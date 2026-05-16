"""
期刊论文语义匹配系统 - 数据库服务
====================================

功能说明：
- 负责SQLite数据库的初始化和操作
- 管理论文元数据的增删改查
- 管理FAISS向量索引的构建和更新

核心设计思路：
- 论文元数据存储在SQLite中，便于查询和过滤
- 论文向量存储在FAISS索引中，支持高效相似度检索
- 两个存储系统通过DOI作为主键关联

作者：AI Assistant
日期：2026-05-16
"""

import json
import sqlite3
from datetime import datetime, date
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from contextlib import contextmanager

import numpy as np
import faiss

from ..models.schemas import Paper, Journal
from ..config import config


# =============================================================================
# 数据库初始化
# =============================================================================

# SQLite表创建语句
CREATE_TABLES_SQL = """
-- 期刊表：存储期刊的基本信息
CREATE TABLE IF NOT EXISTS journals (
    issn TEXT PRIMARY KEY,           -- ISSN是期刊的唯一标识
    name TEXT NOT NULL,             -- 期刊名称
    country TEXT DEFAULT 'INT',      -- 国家代码
    description TEXT,               -- 期刊描述
    last_fetched DATETIME           -- 最后抓取时间
);

-- 论文表：存储论文的元数据
CREATE TABLE IF NOT EXISTS papers (
    doi TEXT PRIMARY KEY,           -- DOI是论文的唯一标识
    title TEXT NOT NULL,            -- 论文标题
    abstract TEXT,                   -- 摘要
    authors TEXT,                   -- 作者列表（JSON格式存储）
    journal_issn TEXT,              -- 所属期刊ISSN
    journal_name TEXT,              -- 所属期刊名称（冗余存储方便展示）
    published_date DATE,            -- 发表日期
    url TEXT,                       -- 论文链接
    keywords TEXT,                  -- 关键词列表（JSON格式存储）
    language TEXT DEFAULT 'en',     -- 论文语言
    indexed_at DATETIME DEFAULT CURRENT_TIMESTAMP,  -- 索引时间
    FOREIGN KEY (journal_issn) REFERENCES journals(issn)
);

-- 论文向量缓存表：避免重复计算向量
CREATE TABLE IF NOT EXISTS paper_vectors (
    doi TEXT PRIMARY KEY,
    vector BLOB,                    -- 向量数据（二进制格式存储）
    indexed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (doi) REFERENCES papers(doi)
);

-- 创建索引加速查询
CREATE INDEX IF NOT EXISTS idx_papers_journal ON papers(journal_issn);
CREATE INDEX IF NOT EXISTS idx_papers_date ON papers(published_date DESC);
"""


class DatabaseService:
    """
    数据库服务类

    负责所有数据库相关的操作，包括：
    1. 数据库连接管理
    2. 期刊数据管理
    3. 论文数据管理
    4. 向量缓存管理
    5. FAISS向量索引管理
    """

    def __init__(self, db_path: str = None):
        """
        初始化数据库服务

        Args:
            db_path: SQLite数据库文件路径，默认使用配置文件中的路径
        """
        self.db_path = db_path or config.database.DB_PATH
        self._ensure_db_dir()  # 确保数据库目录存在
        self._init_database()  # 初始化数据库表
        self.index = None  # FAISS索引对象
        self._load_index()  # 尝试加载已有的FAISS索引

    def _ensure_db_dir(self):
        """确保数据库目录存在"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _get_connection(self):
        """
        获取数据库连接的上下文管理器

        使用with语句自动管理连接的开启和关闭，
        确保操作完成后正确释放资源

        Yields:
            sqlite3.Connection: 数据库连接对象
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # 支持按列名访问数据
        try:
            yield conn
        finally:
            conn.close()

    def _init_database(self):
        """
        初始化数据库表

        执行CREATE TABLE语句，创建所有必要的表
        """
        with self._get_connection() as conn:
            conn.executescript(CREATE_TABLES_SQL)
            conn.commit()

    # =========================================================================
    # 期刊管理方法
    # =========================================================================

    def save_journal(self, journal: Journal) -> bool:
        """
        保存期刊信息

        Args:
            journal: Journal对象

        Returns:
            bool: 是否保存成功
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT OR REPLACE INTO journals (issn, name, country, description, last_fetched)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    journal.issn,
                    journal.name,
                    journal.country,
                    journal.description,
                    journal.last_fetched
                )
            )
            conn.commit()
            return cursor.rowcount > 0

    def save_journals_batch(self, journals: List[Journal]) -> int:
        """
        批量保存期刊信息

        Args:
            journals: Journal对象列表

        Returns:
            int: 成功保存的数量
        """
        count = 0
        with self._get_connection() as conn:
            for journal in journals:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO journals (issn, name, country, description, last_fetched)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        journal.issn,
                        journal.name,
                        journal.country,
                        journal.description,
                        journal.last_fetched
                    )
                )
                count += 1
            conn.commit()
        return count

    def get_journal(self, issn: str) -> Optional[Journal]:
        """
        根据ISSN获取期刊信息

        Args:
            issn: 期刊ISSN

        Returns:
            Journal对象或None
        """
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM journals WHERE issn = ?", (issn,)
            ).fetchone()

            if row:
                return Journal(
                    issn=row["issn"],
                    name=row["name"],
                    country=row["country"],
                    description=row["description"],
                    last_fetched=datetime.fromisoformat(row["last_fetched"]) if row["last_fetched"] else None
                )
            return None

    def get_all_journals(self) -> List[Journal]:
        """
        获取所有已保存的期刊

        Returns:
            Journal对象列表
        """
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM journals ORDER BY country, name").fetchall()
            return [
                Journal(
                    issn=row["issn"],
                    name=row["name"],
                    country=row["country"],
                    description=row["description"],
                    last_fetched=datetime.fromisoformat(row["last_fetched"]) if row["last_fetched"] else None
                )
                for row in rows
            ]

    def update_journal_fetch_time(self, issn: str):
        """更新期刊的最后抓取时间"""
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE journals SET last_fetched = ? WHERE issn = ?",
                (datetime.now(), issn)
            )
            conn.commit()

    # =========================================================================
    # 论文管理方法
    # =========================================================================

    def save_paper(self, paper: Paper) -> bool:
        """
        保存单篇论文

        Args:
            paper: Paper对象

        Returns:
            bool: 是否保存成功
        """
        with self._get_connection() as conn:
            # JSON序列化authors和keywords列表
            authors_json = json.dumps(paper.authors, ensure_ascii=False)
            keywords_json = json.dumps(paper.keywords, ensure_ascii=False)

            cursor = conn.execute(
                """
                INSERT OR REPLACE INTO papers
                (doi, title, abstract, authors, journal_issn, journal_name,
                 published_date, url, keywords, language, indexed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    paper.doi,
                    paper.title,
                    paper.abstract,
                    authors_json,
                    paper.journal_issn,
                    paper.journal_name,
                    paper.published_date,
                    paper.url,
                    keywords_json,
                    paper.language,
                    paper.indexed_at
                )
            )
            conn.commit()
            return cursor.rowcount > 0

    def save_papers_batch(self, papers: List[Paper]) -> int:
        """
        批量保存论文

        Args:
            papers: Paper对象列表

        Returns:
            int: 成功保存的数量
        """
        count = 0
        with self._get_connection() as conn:
            for paper in papers:
                authors_json = json.dumps(paper.authors, ensure_ascii=False)
                keywords_json = json.dumps(paper.keywords, ensure_ascii=False)

                conn.execute(
                    """
                    INSERT OR REPLACE INTO papers
                    (doi, title, abstract, authors, journal_issn, journal_name,
                     published_date, url, keywords, language, indexed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        paper.doi,
                        paper.title,
                        paper.abstract,
                        authors_json,
                        paper.journal_issn,
                        paper.journal_name,
                        paper.published_date,
                        paper.url,
                        keywords_json,
                        paper.language,
                        paper.indexed_at
                    )
                )
                count += 1
            conn.commit()
        return count

    def get_paper(self, doi: str) -> Optional[Paper]:
        """
        根据DOI获取论文

        Args:
            doi: 论文DOI

        Returns:
            Paper对象或None
        """
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM papers WHERE doi = ?", (doi,)
            ).fetchone()

            if row:
                return self._row_to_paper(row)
            return None

    def get_papers_by_journal(self, issn: str) -> List[Paper]:
        """
        获取指定期刊的所有论文

        Args:
            issn: 期刊ISSN

        Returns:
            Paper对象列表，按发表日期降序排列
        """
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM papers
                WHERE journal_issn = ?
                ORDER BY published_date DESC NULLS LAST
                """,
                (issn,)
            ).fetchall()
            return [self._row_to_paper(row) for row in rows]

    def get_all_papers(self) -> List[Paper]:
        """
        获取所有论文

        Returns:
            Paper对象列表
        """
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM papers ORDER BY journal_issn, published_date DESC"
            ).fetchall()
            return [self._row_to_paper(row) for row in rows]

    def get_papers_count(self) -> int:
        """获取论文总数"""
        with self._get_connection() as conn:
            result = conn.execute("SELECT COUNT(*) FROM papers").fetchone()
            return result[0] if result else 0

    def get_journal_papers_count(self, issn: str) -> int:
        """获取指定期刊的论文数量"""
        with self._get_connection() as conn:
            result = conn.execute(
                "SELECT COUNT(*) FROM papers WHERE journal_issn = ?", (issn,)
            ).fetchone()
            return result[0] if result else 0

    def _row_to_paper(self, row: sqlite3.Row) -> Paper:
        """
        将数据库行转换为Paper对象

        Args:
            row: sqlite3.Row对象

        Returns:
            Paper对象
        """
        # 解析JSON格式的authors和keywords
        authors = json.loads(row["authors"]) if row["authors"] else []
        keywords = json.loads(row["keywords"]) if row["keywords"] else []

        # 解析日期
        pub_date = None
        if row["published_date"]:
            try:
                pub_date = date.fromisoformat(row["published_date"])
            except ValueError:
                pass

        return Paper(
            doi=row["doi"],
            title=row["title"],
            abstract=row["abstract"],
            authors=authors,
            journal_issn=row["journal_issn"],
            journal_name=row["journal_name"],
            published_date=pub_date,
            url=row["url"],
            keywords=keywords,
            language=row["language"] or "en",
            indexed_at=datetime.fromisoformat(row["indexed_at"]) if row["indexed_at"] else datetime.now()
        )

    # =========================================================================
    # 向量缓存管理方法
    # =========================================================================

    def save_vector(self, doi: str, vector: np.ndarray):
        """
        保存论文向量到缓存

        Args:
            doi: 论文DOI
            vector: numpy向量数组
        """
        # 将numpy数组转换为二进制格式存储
        vector_bytes = vector.astype(np.float32).tobytes()

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO paper_vectors (doi, vector, indexed_at)
                VALUES (?, ?, ?)
                """,
                (doi, vector_bytes, datetime.now())
            )
            conn.commit()

    def get_vector(self, doi: str) -> Optional[np.ndarray]:
        """
        从缓存获取论文向量

        Args:
            doi: 论文DOI

        Returns:
            numpy向量数组或None
        """
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT vector FROM paper_vectors WHERE doi = ?", (doi,)
            ).fetchone()

            if row and row["vector"]:
                # 将二进制格式转换回numpy数组
                return np.frombuffer(row["vector"], dtype=np.float32)
            return None

    def get_vectors_batch(self, dois: List[str]) -> Tuple[List[str], List[np.ndarray]]:
        """
        批量获取论文向量

        Args:
            dois: DOI列表

        Returns:
            (存在的DOI列表, 对应的向量列表)
        """
        existing_dois = []
        vectors = []

        with self._get_connection() as conn:
            placeholders = ",".join("?" * len(dois))
            rows = conn.execute(
                f"SELECT doi, vector FROM paper_vectors WHERE doi IN ({placeholders})",
                dois
            ).fetchall()

            for row in rows:
                if row["vector"]:
                    existing_dois.append(row["doi"])
                    vectors.append(np.frombuffer(row["vector"], dtype=np.float32))

        return existing_dois, vectors

    # =========================================================================
    # FAISS向量索引管理
    # =========================================================================

    def _load_index(self):
        """加载FAISS索引（如果存在）"""
        # 尝试原始路径
        index_path = Path(config.VECTOR_INDEX_PATH)
        if not index_path.exists():
            # 尝试安全路径
            index_path = config.get_safe_index_path()

        if index_path.exists():
            try:
                self.index = faiss.read_index(str(index_path.resolve()))
            except Exception as e:
                print(f"加载FAISS索引失败: {e}")
                self.index = None

    def build_index(self, papers: List[Paper], vectors: np.ndarray):
        """
        构建FAISS向量索引

        Args:
            papers: Paper对象列表（顺序与vectors对应）
            vectors: numpy矩阵，每行是一篇论文的向量

        说明：
        - 使用IndexFlatIP（内积索引）+ L2归一化实现余弦相似度
        - 这种方式简单高效，适合小规模数据集（<100万）
        """
        if len(papers) == 0 or vectors.shape[0] == 0:
            print("警告：没有论文数据，跳过索引构建")
            return

        dim = vectors.shape[1]  # 向量维度

        # 创建L2归一化的内积索引
        # 为什么用内积+归一化：
        # 归一化后的向量，内积 = 余弦相似度
        # 比直接用余弦相似度计算更快
        self.index = faiss.IndexFlatIP(dim)

        # 对向量进行L2归一化
        # 这样内积就直接等于余弦相似度
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # 防止除零
        normalized_vectors = vectors / norms

        # 添加向量到索引
        self.index.add(normalized_vectors.astype(np.float32))

        # 保存索引到磁盘
        self._save_index()

        print(f"FAISS索引构建完成：{len(papers)} 篇论文，维度 {dim}")

    def _save_index(self):
        """保存FAISS索引到磁盘"""
        if self.index is not None:
            # 使用安全的路径，避免中文路径问题
            index_path = config.get_safe_index_path()
            index_path.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self.index, str(index_path.resolve()))

    def search_index(self, query_vector: np.ndarray, top_k: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """
        在向量索引中搜索相似论文

        Args:
            query_vector: 查询向量（1D numpy数组）
            top_k: 返回的最相似论文数量

        Returns:
            (相似度分数数组, 论文索引数组)
        """
        if self.index is None:
            return np.array([]), np.array([])

        # 归一化查询向量
        norm = np.linalg.norm(query_vector)
        if norm > 0:
            query_vector = query_vector / norm

        # 搜索
        query_vector = query_vector.reshape(1, -1).astype(np.float32)
        similarities, indices = self.index.search(query_vector, top_k)

        return similarities[0], indices[0]

    def is_index_ready(self) -> bool:
        """检查向量索引是否就绪"""
        return self.index is not None and self.index.ntotal > 0

    def get_index_size(self) -> int:
        """获取索引中的向量数量"""
        if self.index is None:
            return 0
        return self.index.ntotal

    def clear_all_data(self):
        """清空所有数据（用于测试或重置）"""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM paper_vectors")
            conn.execute("DELETE FROM papers")
            conn.execute("DELETE FROM journals")
            conn.commit()

        # 删除FAISS索引文件
        if Path(config.VECTOR_INDEX_PATH).exists():
            Path(config.VECTOR_INDEX_PATH).unlink()

        self.index = None
