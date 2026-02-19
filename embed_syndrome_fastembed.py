#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from typing import List, Dict, Any

from neo4j import GraphDatabase
from fastembed import TextEmbedding

# ====== Neo4j 配置 ======
NEO4J_URL = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "12345678"
NEO4J_DB = "neo4j"

# ====== 你想要的候选模型（按顺序尝试）======
CANDIDATE_MODELS = [
    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-small-zh-v1.5",
    "BAAI/bge-small-zh",
    "BAAI/bge-small-en",
]

# 你 Neo4j 向量索引期望维度（如果选到的模型维度不同，下面会提示你改索引维度）
VECTOR_DIM_EXPECTED = 384

BATCH_SIZE = 200
SLEEP = 0.05


def pick_model() -> str:
    supported = TextEmbedding.list_supported_models()
    supported_names = {m["model"] for m in supported if "model" in m}

    for m in CANDIDATE_MODELS:
        if m in supported_names:
            return m

    # 如果都不在，就直接选第一个支持的模型（兜底）
    if supported:
        return supported[0]["model"]

    raise RuntimeError("fastembed 没有返回任何支持模型列表，请检查 fastembed 安装是否正常。")


def fetch_batch(driver) -> List[Dict[str, Any]]:
    cypher = """
    MATCH (n:SyndromeNode)
    WHERE n.embedding IS NULL OR size(n.embedding) <> $dim
    RETURN id(n) AS nid, n.name AS name, n.syndrome_code AS code
    LIMIT $limit
    """
    with driver.session(database=NEO4J_DB) as s:
        return s.run(cypher, dim=VECTOR_DIM_EXPECTED, limit=BATCH_SIZE).data()


def build_text(row: Dict[str, Any]) -> str:
    name = (row.get("name") or "").strip()
    code = (row.get("code") or "").strip()
    return f"证型 {name} 代码 {code}"


def embed_many(embedder: TextEmbedding, texts: List[str]) -> List[List[float]]:
    out = []
    for v in embedder.embed(texts):
        vec = v.tolist() if hasattr(v, "tolist") else list(v)
        out.append(vec)
    return out


def write_batch(driver, rows: List[Dict[str, Any]], vecs: List[List[float]]) -> int:
    payload = [{"nid": r["nid"], "embedding": v} for r, v in zip(rows, vecs)]
    cypher = """
    UNWIND $rows AS row
    MATCH (n:SyndromeNode) WHERE id(n) = row.nid
    SET n.embedding = row.embedding
    RETURN count(n) AS updated
    """
    with driver.session(database=NEO4J_DB) as s:
        return s.run(cypher, rows=payload).single()["updated"]


def main():
    driver = GraphDatabase.driver(NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASS))

    model_name = pick_model()
    print(f"✅ Using fastembed model: {model_name}")

    embedder = TextEmbedding(model_name=model_name)

    # 维度自检
    test_vec = next(embedder.embed(["test"]))
    dim = len(test_vec)
    print(f"📏 Embedding dimension = {dim}")

    if dim != VECTOR_DIM_EXPECTED:
        raise RuntimeError(
            f"当前模型维度={dim}，但你的索引/期望维度={VECTOR_DIM_EXPECTED}。\n"
            f"解决：要么把 VECTOR_DIM_EXPECTED 改成 {dim} 并重建 Neo4j 向量索引 dimensions，"
            f"要么换一个 384 维的模型。"
        )

    total = 0
    while True:
        batch = fetch_batch(driver)
        if not batch:
            break

        texts = [build_text(r) for r in batch]
        vecs = embed_many(embedder, texts)
        updated = write_batch(driver, batch, vecs)

        total += updated
        print(f"[SyndromeNode] updated {updated}, total {total}")
        time.sleep(SLEEP)

    driver.close()
    print("✅ done, total updated:", total)


if __name__ == "__main__":
    main()
