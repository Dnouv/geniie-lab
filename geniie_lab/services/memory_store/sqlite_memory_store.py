import json
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional


@dataclass
class MemoryRecord:
    memory_id: int
    memory_note: str
    memory_timestamp: str
    stage: str
    created_at: str
    memory_kind: Optional[str] = None
    confidence: Optional[float] = None
    tags: list[str] = field(default_factory=list)
    payload: Optional[dict[str, Any]] = None


class SQLiteMemoryStore:
    def __init__(self, db_path: str):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS session_memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                topic_id TEXT NOT NULL,
                stage TEXT NOT NULL,
                memory_note TEXT NOT NULL,
                memory_timestamp TEXT,
                memory_kind TEXT,
                confidence REAL,
                tags_json TEXT,
                payload_json TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_session_memories_scope
            ON session_memories (run_id, topic_id, created_at DESC)
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS session_memory_sql_reads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                topic_id TEXT NOT NULL,
                stage TEXT NOT NULL,
                step_num INTEGER NOT NULL,
                raw_sql_query TEXT NOT NULL,
                executed_sql_query TEXT NOT NULL,
                row_count INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_session_memory_sql_reads_scope
            ON session_memory_sql_reads (run_id, topic_id, stage, step_num)
            """
        )
        self._ensure_column("session_memories", "memory_timestamp", "TEXT")
        self._conn.commit()

    def _ensure_column(self, table_name: str, column_name: str, definition: str) -> None:
        cur = self._conn.cursor()
        cur.execute(f"PRAGMA table_info({table_name})")
        existing = {str(row[1]) for row in cur.fetchall()}
        if column_name in existing:
            return
        cur.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}")

    def write_memory(
        self,
        *,
        run_id: str,
        topic_id: str,
        stage: str,
        memory_note: str,
        memory_timestamp: Optional[str] = None,
        memory_kind: Optional[str] = None,
        confidence: Optional[float] = None,
        tags: Optional[list[str]] = None,
        payload: Optional[dict[str, Any]] = None,
    ) -> int:
        clean_note = memory_note.strip()
        if not clean_note:
            raise ValueError("memory_note must be non-empty.")
        created_at = datetime.now(UTC).isoformat()
        if not memory_timestamp:
            memory_timestamp = created_at
        tags_json = json.dumps(tags or [], ensure_ascii=False)
        payload_json = json.dumps(payload or {}, ensure_ascii=False)
        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT INTO session_memories (
                run_id, topic_id, stage, memory_note, memory_timestamp, memory_kind,
                confidence, tags_json, payload_json, created_at
            ) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                topic_id,
                stage,
                clean_note,
                memory_timestamp,
                memory_kind,
                confidence,
                tags_json,
                payload_json,
                created_at,
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def query_memories(
        self,
        *,
        run_id: str,
        topic_id: str,
        query_text: str,
        top_k: int,
    ) -> list[MemoryRecord]:
        limit = max(1, min(int(top_k), 50))
        query_text = (query_text or "").strip()
        cur = self._conn.cursor()

        if not query_text or query_text == "*":
            cur.execute(
                """
                SELECT * FROM session_memories
                WHERE run_id = ? AND topic_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (run_id, topic_id, limit),
            )
        else:
            like_expr = f"%{query_text}%"
            cur.execute(
                """
                SELECT * FROM session_memories
                WHERE run_id = ? AND topic_id = ?
                  AND (
                    memory_note LIKE ? COLLATE NOCASE
                    OR IFNULL(memory_kind, '') LIKE ? COLLATE NOCASE
                    OR IFNULL(tags_json, '') LIKE ? COLLATE NOCASE
                    OR IFNULL(payload_json, '') LIKE ? COLLATE NOCASE
                  )
                ORDER BY id DESC
                LIMIT ?
                """,
                (run_id, topic_id, like_expr, like_expr, like_expr, like_expr, limit),
            )
        rows = cur.fetchall()
        return [self._row_to_record(row) for row in rows]

    def delete_memory_by_id(self, *, run_id: str, topic_id: str, memory_id: int) -> bool:
        cur = self._conn.cursor()
        cur.execute(
            """
            DELETE FROM session_memories
            WHERE run_id = ? AND topic_id = ? AND id = ?
            """,
            (run_id, topic_id, int(memory_id)),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def delete_memories_by_text(
        self,
        *,
        run_id: str,
        topic_id: str,
        match_text: str,
        max_delete: int = 5,
    ) -> int:
        match_text = (match_text or "").strip()
        if not match_text:
            return 0
        limit = max(1, min(int(max_delete), 50))
        cur = self._conn.cursor()
        like_expr = f"%{match_text}%"
        cur.execute(
            """
            SELECT id FROM session_memories
            WHERE run_id = ? AND topic_id = ?
              AND (
                memory_note LIKE ? COLLATE NOCASE
                OR IFNULL(memory_kind, '') LIKE ? COLLATE NOCASE
                OR IFNULL(tags_json, '') LIKE ? COLLATE NOCASE
                OR IFNULL(payload_json, '') LIKE ? COLLATE NOCASE
              )
            ORDER BY id DESC
            LIMIT ?
            """,
            (run_id, topic_id, like_expr, like_expr, like_expr, like_expr, limit),
        )
        ids = [int(row["id"]) for row in cur.fetchall()]
        if not ids:
            return 0
        cur.executemany(
            "DELETE FROM session_memories WHERE run_id = ? AND topic_id = ? AND id = ?",
            [(run_id, topic_id, memory_id) for memory_id in ids],
        )
        self._conn.commit()
        return len(ids)

    def close(self) -> None:
        self._conn.close()

    def log_sql_read(
        self,
        *,
        run_id: str,
        topic_id: str,
        stage: str,
        step_num: int,
        raw_sql_query: str,
        executed_sql_query: str,
        row_count: int,
    ) -> None:
        created_at = datetime.now(UTC).isoformat()
        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT INTO session_memory_sql_reads (
                run_id, topic_id, stage, step_num,
                raw_sql_query, executed_sql_query, row_count, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                topic_id,
                stage,
                int(step_num),
                raw_sql_query,
                executed_sql_query,
                int(row_count),
                created_at,
            ),
        )
        self._conn.commit()

    def execute_read_sql(
        self,
        *,
        run_id: str,
        topic_id: str,
        sql_query: str,
    ) -> tuple[list[dict[str, Any]], str]:
        cleaned = self._validate_select_sql(sql_query)
        cur = self._conn.cursor()
        self._create_scoped_views(cur, run_id=run_id, topic_id=topic_id)
        cur.execute(cleaned)
        columns = [str(item[0]) for item in (cur.description or [])]
        rows = []
        for raw_row in cur.fetchall():
            rows.append({columns[idx]: raw_row[idx] for idx in range(len(columns))})
        return rows, cleaned

    @staticmethod
    def _validate_select_sql(sql_query: str) -> str:
        cleaned = (sql_query or "").strip()
        if not cleaned:
            raise ValueError("sql_query must be non-empty.")
        cleaned = cleaned.rstrip(";").strip()
        lower = cleaned.lower()
        if not (lower.startswith("select ") or lower.startswith("with ")):
            raise ValueError("Only SELECT queries are allowed for memory reads.")
        if ";" in cleaned:
            raise ValueError("Only a single SELECT statement is allowed.")
        allowed_views = ("topic_memories", "topic_memory_sql_reads", "db_relations", "db_columns")
        if not any(view_name in lower for view_name in allowed_views):
            raise ValueError("Read SQL must query one or more scoped views: topic_memories, topic_memory_sql_reads, db_relations, db_columns.")
        blocked_tokens = (" insert ", " update ", " delete ", " drop ", " alter ", " pragma ", " attach ", " detach ")
        if any(token in f" {lower} " for token in blocked_tokens):
            raise ValueError("Read SQL contains disallowed tokens.")
        return cleaned

    def _create_scoped_views(self, cur: sqlite3.Cursor, *, run_id: str, topic_id: str) -> None:
        escaped_run_id = run_id.replace("'", "''")
        escaped_topic_id = topic_id.replace("'", "''")
        cur.execute("DROP VIEW IF EXISTS topic_memories")
        cur.execute("DROP VIEW IF EXISTS topic_memory_sql_reads")
        cur.execute("DROP VIEW IF EXISTS db_relations")
        cur.execute("DROP VIEW IF EXISTS db_columns")
        cur.execute(
            f"""
            CREATE TEMP VIEW topic_memories AS
            SELECT
                id,
                memory_note AS content,
                COALESCE(memory_timestamp, created_at) AS memory_timestamp,
                stage,
                created_at
            FROM session_memories
            WHERE run_id = '{escaped_run_id}' AND topic_id = '{escaped_topic_id}'
            """
        )
        cur.execute(
            """
            CREATE TEMP VIEW db_relations AS
            SELECT
                'topic_memories' AS object_name,
                'view' AS object_type,
                'Stored memory content scoped to the current run and topic.' AS description
            UNION ALL
            SELECT
                'topic_memory_sql_reads' AS object_name,
                'view' AS object_type,
                'Audit log of prior SQL read queries scoped to the current run and topic.' AS description
            UNION ALL
            SELECT
                'db_relations' AS object_name,
                'view' AS object_type,
                'Catalog of scoped views exposed to the model.' AS description
            UNION ALL
            SELECT
                'db_columns' AS object_name,
                'view' AS object_type,
                'Catalog of columns exposed by the scoped views.' AS description
            """
        )
        cur.execute(
            """
            CREATE TEMP VIEW db_columns AS
            SELECT 'topic_memories' AS object_name, 'id' AS column_name, 'INTEGER' AS data_type, 'Memory row identifier.' AS description
            UNION ALL
            SELECT 'topic_memories', 'content', 'TEXT', 'Free-text stored memory content.'
            UNION ALL
            SELECT 'topic_memories', 'memory_timestamp', 'TEXT', 'Runtime-assigned timestamp for the memory entry.'
            UNION ALL
            SELECT 'topic_memories', 'stage', 'TEXT', 'Stage that wrote the memory.'
            UNION ALL
            SELECT 'topic_memories', 'created_at', 'TEXT', 'Runtime creation timestamp.'
            UNION ALL
            SELECT 'topic_memory_sql_reads', 'id', 'INTEGER', 'SQL read audit row identifier.'
            UNION ALL
            SELECT 'topic_memory_sql_reads', 'stage', 'TEXT', 'Stage that issued the SQL read.'
            UNION ALL
            SELECT 'topic_memory_sql_reads', 'step_num', 'INTEGER', 'Tool-loop step index within the stage.'
            UNION ALL
            SELECT 'topic_memory_sql_reads', 'raw_sql_query', 'TEXT', 'Raw SQL query emitted by the model.'
            UNION ALL
            SELECT 'topic_memory_sql_reads', 'executed_sql_query', 'TEXT', 'Validated SQL query executed by the runtime.'
            UNION ALL
            SELECT 'topic_memory_sql_reads', 'row_count', 'INTEGER', 'Number of rows returned by the SQL read.'
            UNION ALL
            SELECT 'topic_memory_sql_reads', 'created_at', 'TEXT', 'Runtime timestamp for the read audit event.'
            UNION ALL
            SELECT 'db_relations', 'object_name', 'TEXT', 'Scoped relation name.'
            UNION ALL
            SELECT 'db_relations', 'object_type', 'TEXT', 'Relation type.'
            UNION ALL
            SELECT 'db_relations', 'description', 'TEXT', 'Human-readable description.'
            UNION ALL
            SELECT 'db_columns', 'object_name', 'TEXT', 'Scoped relation name.'
            UNION ALL
            SELECT 'db_columns', 'column_name', 'TEXT', 'Column name.'
            UNION ALL
            SELECT 'db_columns', 'data_type', 'TEXT', 'Logical data type.'
            UNION ALL
            SELECT 'db_columns', 'description', 'TEXT', 'Human-readable column description.'
            """
        )
        cur.execute(
            f"""
            CREATE TEMP VIEW topic_memory_sql_reads AS
            SELECT
                id,
                stage,
                step_num,
                raw_sql_query,
                executed_sql_query,
                row_count,
                created_at
            FROM session_memory_sql_reads
            WHERE run_id = '{escaped_run_id}' AND topic_id = '{escaped_topic_id}'
            """
        )

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> MemoryRecord:
        tags = []
        payload = None
        tags_raw = row["tags_json"]
        if isinstance(tags_raw, str):
            try:
                parsed = json.loads(tags_raw)
                if isinstance(parsed, list):
                    tags = [str(item) for item in parsed]
            except json.JSONDecodeError:
                tags = []
        payload_raw = row["payload_json"]
        if isinstance(payload_raw, str):
            try:
                parsed_payload = json.loads(payload_raw)
                if isinstance(parsed_payload, dict):
                    payload = parsed_payload
            except json.JSONDecodeError:
                payload = None
        return MemoryRecord(
            memory_id=int(row["id"]),
            memory_note=str(row["memory_note"]),
            memory_timestamp=str(row["memory_timestamp"] or row["created_at"]),
            stage=str(row["stage"]),
            created_at=str(row["created_at"]),
            memory_kind=row["memory_kind"],
            confidence=row["confidence"],
            tags=tags,
            payload=payload,
        )
