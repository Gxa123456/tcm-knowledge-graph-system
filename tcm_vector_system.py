#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中医知识图谱系统 - Neo4j 向量检索版本（FastEmbed + Setup + Fuzzy + Drill）

目标：
- 向量检索只做“模糊匹配/召回候选”
- 最终返回通过 Cypher 得到的：疾病/证型/症状（含 code），并可下钻到症状列表

功能：
1) 测试 Neo4j 连接
2) 测试 fastembed embeddings 是否可用，并拿到向量维度
3) setup：创建向量索引 + 批量写入 embedding（Disease / SyndromeNode / SymptomNode）
4) ask：用户输入任意（疾病/证型/症状/口语） -> 返回三类候选（Cypher） + 自动下钻结果
5) cand：只看三类候选（Cypher 返回的统一结构）
"""

import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from fastembed import TextEmbedding
from neo4j import GraphDatabase


# =========================
# 配置
# =========================
@dataclass
class Config:
    neo4j_url: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_pass: str = "12345678"
    neo4j_db: str = "neo4j"

    # ✅ AUTO = 启动时从 fastembed 支持列表里自动选择
    fastembed_model: str = "AUTO"

    # 索引名称
    idx_disease: str = "disease_embedding_idx"
    idx_syndrome: str = "syndrome_embedding_idx"
    idx_symptom: str = "symptom_embedding_idx"

    # 写入批次
    batch_size: int = 64
    sleep: float = 0.05

    # 下钻关系类型（如果你的关系名不是 HAS_SYMPTOM，就改这里）
    rel_has_symptom: str = "HAS_SYMPTOM"


CFG = Config()


# =========================
# 向量检索系统
# =========================
class TCMVectorSystem:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.driver = None

        # 由 test_embeddings() 自动填充
        self.embed_dim: Optional[int] = None

        # ✅ 自动选择 fastembed 支持的模型
        self.cfg.fastembed_model = self._pick_fastembed_model(self.cfg.fastembed_model)
        print(f"🧠 fastembed model = {self.cfg.fastembed_model}")

        # ✅ 初始化 embedder（只做一次）
        self.embedder = TextEmbedding(model_name=self.cfg.fastembed_model)

    @staticmethod
    def _pick_fastembed_model(model_name: str) -> str:
        supported = [m["model"] for m in TextEmbedding.list_supported_models()]
        supported_set = set(supported)

        if model_name != "AUTO":
            if model_name not in supported_set:
                raise ValueError(
                    f"fastembed 不支持模型: {model_name}\n"
                    f"请运行：python -c \"from fastembed import TextEmbedding; "
                    f"print([m['model'] for m in TextEmbedding.list_supported_models()])\""
                )
            return model_name

        # ✅ AUTO：优先中文/多语言，其次英文
        preferred = [
            "BAAI/bge-small-zh-v1.5",
            "BAAI/bge-base-zh-v1.5",
            "BAAI/bge-large-zh-v1.5",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "nomic-ai/nomic-embed-text-v1",
            "BAAI/bge-small-en-v1.5",
            "BAAI/bge-base-en-v1.5",
        ]

        for m in preferred:
            if m in supported_set:
                return m

        return supported[0]

    # -------------------------
    # Neo4j 连接测试
    # -------------------------
    def test_neo4j(self) -> bool:
        try:
            self.driver = GraphDatabase.driver(
                self.cfg.neo4j_url, auth=(self.cfg.neo4j_user, self.cfg.neo4j_pass)
            )
            with self.driver.session(database=self.cfg.neo4j_db) as session:
                v = session.run("RETURN 1 AS test").single()
                if v and v["test"] == 1:
                    print("✅ Neo4j 基本连接成功")

                rec = session.run(
                    "CALL dbms.components() YIELD name, versions "
                    "RETURN name, versions[0] AS version LIMIT 1"
                ).single()
                if rec:
                    print(f"📊 组件: {rec['name']}, 版本: {rec['version']}")

                total = session.run("MATCH (n) RETURN count(n) AS c").single()
                print(f"📊 总节点数: {total['c'] if total else '未知'}")

                for label in ["Disease", "SymptomNode", "SyndromeNode"]:
                    c = session.run(f"MATCH (n:{label}) RETURN count(n) AS c").single()
                    c = c["c"] if c else 0
                    print(f"📌 {label}: {c}")

            return True
        except Exception as e:
            print(f"❌ Neo4j 连接失败: {e}")
            traceback.print_exc()
            return False

    def close(self):
        if self.driver:
            self.driver.close()

    # -------------------------
    # embeddings（fastembed）
    # -------------------------
    def embed_text(self, text: str) -> List[float]:
        vec = next(self.embedder.embed([text]))
        return vec.tolist()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for v in self.embedder.embed(texts):
            out.append(v.tolist())
        return out

    def test_embeddings(self) -> bool:
        try:
            v = self.embed_text("test")
            self.embed_dim = len(v)
            print(f"✅ embeddings 可用（fastembed），dim={self.embed_dim}")
            return True
        except Exception as e:
            print("❌ embeddings 不可用（fastembed）")
            print(f"错误: {e}")
            traceback.print_exc()
            return False

    # -------------------------
    # Neo4j：执行工具
    # -------------------------
    def _run(self, cypher: str, **params):
        with self.driver.session(database=self.cfg.neo4j_db) as session:
            return session.run(cypher, **params)

    # -------------------------
    # Neo4j：索引与写入
    # -------------------------
    def create_vector_indexes(self):
        if not self.embed_dim:
            raise RuntimeError("embed_dim 为空，请先 test_embeddings()")

        stmts = [
            f"DROP INDEX {self.cfg.idx_disease} IF EXISTS",
            f"""CREATE VECTOR INDEX {self.cfg.idx_disease}
                FOR (n:Disease) ON (n.embedding)
                OPTIONS {{indexConfig: {{
                  `vector.dimensions`: {self.embed_dim},
                  `vector.similarity_function`: 'cosine'
                }}}}""",

            f"DROP INDEX {self.cfg.idx_syndrome} IF EXISTS",
            f"""CREATE VECTOR INDEX {self.cfg.idx_syndrome}
                FOR (n:SyndromeNode) ON (n.embedding)
                OPTIONS {{indexConfig: {{
                  `vector.dimensions`: {self.embed_dim},
                  `vector.similarity_function`: 'cosine'
                }}}}""",

            f"DROP INDEX {self.cfg.idx_symptom} IF EXISTS",
            f"""CREATE VECTOR INDEX {self.cfg.idx_symptom}
                FOR (n:SymptomNode) ON (n.embedding)
                OPTIONS {{indexConfig: {{
                  `vector.dimensions`: {self.embed_dim},
                  `vector.similarity_function`: 'cosine'
                }}}}""",
        ]

        print("🧱 创建向量索引中...")
        for s in stmts:
            self._run(s)
        print("✅ 向量索引创建完成")

    def _write_embeddings(self, label: str, rows: List[Dict[str, Any]], vectors: List[List[float]]) -> int:
        payload = [{"nid": r["nid"], "embedding": v} for r, v in zip(rows, vectors)]
        cypher = f"""
        UNWIND $rows AS row
        MATCH (n:{label}) WHERE id(n) = row.nid
        SET n.embedding = row.embedding
        RETURN count(n) AS updated
        """
        rec = self._run(cypher, rows=payload).single()
        return int(rec["updated"]) if rec else 0

    def backfill_embeddings(self, label: str, text_builder):
        total = 0
        while True:
            cypher_fetch = f"""
            MATCH (n:{label})
            WHERE n.embedding IS NULL
            RETURN id(n) AS nid, properties(n) AS props
            LIMIT $limit
            """
            batch = self._run(cypher_fetch, limit=self.cfg.batch_size).data()
            if not batch:
                break

            texts = []
            rows = []
            for r in batch:
                props = r.get("props") or {}
                rows.append({"nid": r["nid"], "props": props})
                texts.append(text_builder(props))

            vectors = self.embed_texts(texts)
            updated = self._write_embeddings(label, [{"nid": x["nid"]} for x in rows], vectors)
            total += updated
            print(f"[{label}] updated {updated}, total {total}")
            time.sleep(self.cfg.sleep)

        print(f"✅ [{label}] backfill done, total={total}")

    def setup_embeddings_all(self):
        self.create_vector_indexes()

        def disease_text(p: Dict[str, Any]) -> str:
            name = (p.get("tcm_disease") or p.get("name") or "").strip()
            code = (p.get("tcm_code") or "").strip()
            return f"疾病 {name} 代码 {code}".strip()

        def syndrome_text(p: Dict[str, Any]) -> str:
            name = (p.get("name") or "").strip()
            code = (p.get("syndrome_code") or "").strip()
            return f"证型 {name} 代码 {code}".strip()

        def symptom_text(p: Dict[str, Any]) -> str:
            name = (p.get("symptom") or p.get("name") or "").strip()
            code = (p.get("symptom_code") or "").strip()
            return f"症状 {name} 代码 {code}".strip()

        print("🧠 开始写入 Disease.embedding ...")
        self.backfill_embeddings("Disease", disease_text)

        print("🧠 开始写入 SyndromeNode.embedding ...")
        self.backfill_embeddings("SyndromeNode", syndrome_text)

        print("🧠 开始写入 SymptomNode.embedding ...")
        self.backfill_embeddings("SymptomNode", symptom_text)

    # -------------------------
    # 查询（向量只做模糊匹配，返回用 Cypher）
    # -------------------------
    def embed_query(self, q: str) -> List[float]:
        return self.embed_text(q)

    def vector_candidates(self, index_name: str, top_k: int, embedding: List[float]) -> List[Tuple[Dict[str, Any], float]]:
        cypher = """
        CALL db.index.vector.queryNodes($index, $k, $embedding)
        YIELD node, score
        RETURN properties(node) AS props, score AS score
        ORDER BY score DESC
        """
        rows = self._run(cypher, index=index_name, k=top_k, embedding=embedding).data()
        return [(r["props"], float(r["score"])) for r in rows]

    # ✅ 你问的这段：放在“查询”区（vector_candidates 下面）
    def fuzzy_match_all(self, q: str, k_each: int = 5) -> List[Dict[str, Any]]:
        """
        用户随便输入（疾病/证型/症状/口语），用向量做模糊召回：
        - Disease / SyndromeNode / SymptomNode 各取 k_each
        - 返回统一格式：type, name, code, score, nid
        """
        emb = self.embed_query(q)

        cypher = """
        CALL {
          WITH $emb AS emb, $k AS k
          CALL db.index.vector.queryNodes($idx_disease, k, emb)
          YIELD node, score
          RETURN 'Disease' AS type,
                 id(node) AS nid,
                 coalesce(node.tcm_disease, node.name, '') AS name,
                 coalesce(node.tcm_code, '') AS code,
                 score AS score
          UNION ALL
          WITH $emb AS emb, $k AS k
          CALL db.index.vector.queryNodes($idx_syndrome, k, emb)
          YIELD node, score
          RETURN 'SyndromeNode' AS type,
                 id(node) AS nid,
                 coalesce(node.name, '') AS name,
                 coalesce(node.syndrome_code, '') AS code,
                 score AS score
          UNION ALL
          WITH $emb AS emb, $k AS k
          CALL db.index.vector.queryNodes($idx_symptom, k, emb)
          YIELD node, score
          RETURN 'SymptomNode' AS type,
                 id(node) AS nid,
                 coalesce(node.symptom, node.name, '') AS name,
                 coalesce(node.symptom_code, '') AS code,
                 score AS score
        }
        RETURN type, nid, name, code, score
        ORDER BY score DESC
        LIMIT $limit
        """

        rows = self._run(
            cypher,
            emb=emb,
            k=k_each,
            limit=k_each * 3,
            idx_disease=self.cfg.idx_disease,
            idx_syndrome=self.cfg.idx_syndrome,
            idx_symptom=self.cfg.idx_symptom,
        ).data()
        return rows

    def disease_to_symptoms(self, embedding: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
        rel = self.cfg.rel_has_symptom
        cypher = f"""
        CALL db.index.vector.queryNodes('{self.cfg.idx_disease}', $k, $embedding)
        YIELD node AS d, score
        MATCH (d)-[:{rel}]->(s:SymptomNode)
        RETURN
          d.tcm_disease AS disease,
          d.tcm_code AS disease_code,
          score AS match_score,
          collect(DISTINCT {{name: coalesce(s.symptom, s.name), code: s.symptom_code}}) AS symptoms
        ORDER BY match_score DESC
        LIMIT 10
        """
        return self._run(cypher, k=top_k, embedding=embedding).data()

    def syndrome_to_symptoms(self, embedding: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
        rel = self.cfg.rel_has_symptom
        cypher = f"""
        CALL db.index.vector.queryNodes('{self.cfg.idx_syndrome}', $k, $embedding)
        YIELD node AS sy, score
        MATCH (sy)-[:{rel}]->(s:SymptomNode)
        RETURN
          sy.name AS syndrome,
          sy.syndrome_code AS syndrome_code,
          score AS match_score,
          collect(DISTINCT {{name: coalesce(s.symptom, s.name), code: s.symptom_code}}) AS symptoms
        ORDER BY match_score DESC
        LIMIT 10
        """
        return self._run(cypher, k=top_k, embedding=embedding).data()

    def smart_answer(self, q: str, k_each: int = 5, drill_k: int = 3) -> Dict[str, Any]:
        """
        1) 用 fuzzy_match_all 得到三类候选（Cypher）
        2) 取最高分作为意图
        3) Disease/SyndromeNode 自动下钻症状；SymptomNode 返回相似症状候选
        """
        cands = self.fuzzy_match_all(q, k_each=k_each)
        if not cands:
            return {"query": q, "candidates": [], "best": None, "result": None}

        best = cands[0]
        emb = self.embed_query(q)

        result = None
        if best["type"] == "Disease":
            result = self.disease_to_symptoms(emb, top_k=drill_k)
        elif best["type"] == "SyndromeNode":
            result = self.syndrome_to_symptoms(emb, top_k=drill_k)
        else:
            result = {
                "symptom_candidates": [
                    {"name": r["name"], "code": r["code"], "score": r["score"]}
                    for r in cands if r["type"] == "SymptomNode"
                ]
            }

        return {"query": q, "candidates": cands, "best": best, "result": result}

    # -------------------------
    # CLI
    # -------------------------
    def run_cli(self):
        print("==================================================")
        print("TCM Vector System (Neo4j Vector Search + fastembed)")
        print("==================================================")

        print("\n[1] 测试 Neo4j 连接...")
        if not self.test_neo4j():
            return

        print("\n[2] 测试 embeddings（fastembed）...")
        if not self.test_embeddings():
            return

        while True:
            print("\n==================================================")
            print("选择模式：")
            print("  1) setup：创建向量索引 + 批量写 embedding")
            print("  2) ask：输入任意（疾病/证型/症状/口语）-> 返回候选 + 自动下钻")
            print("  3) cand：只返回三类候选（不下钻）")
            print("  4) models：打印 fastembed 支持的模型列表")
            print("  q) 退出")
            cmd = input("输入: ").strip().lower()

            if cmd in ("q", "quit", "exit", "退出"):
                break

            try:
                if cmd == "1":
                    self.setup_embeddings_all()

                elif cmd == "2":
                    q = input("输入疾病/证型/症状（支持口语模糊）：").strip()
                    out = self.smart_answer(q, k_each=5, drill_k=3)

                    print("\n=== Candidates (模糊匹配候选：Cypher 返回) ===")
                    for r in out["candidates"]:
                        print(f"{float(r['score']):.4f}  {r['type']:<12}  {r['name']}  {r['code']}")

                    print("\n=== Best Guess (推测意图) ===")
                    b = out.get("best")
                    if b:
                        print(f"{b['type']} | {b['name']} | {b['code']} | score={float(b['score']):.4f}")

                    print("\n=== Result (下钻结果 / 或症状候选) ===")
                    print(out["result"])

                elif cmd == "3":
                    q = input("输入关键词（口语也可）: ").strip()
                    rows = self.fuzzy_match_all(q, k_each=5)
                    print("\n=== Candidates (Cypher 返回) ===")
                    for r in rows:
                        print(f"{float(r['score']):.4f}  {r['type']:<12}  {r['name']}  {r['code']}")

                elif cmd == "4":
                    models = [m["model"] for m in TextEmbedding.list_supported_models()]
                    print("\n".join(models))

                else:
                    print("未知命令")
            except Exception as e:
                print(f"❌ 执行失败: {e}")
                traceback.print_exc()

        self.close()
        print("bye")


def main():
    sys_ = TCMVectorSystem(CFG)
    sys_.run_cli()


if __name__ == "__main__":
    main()
