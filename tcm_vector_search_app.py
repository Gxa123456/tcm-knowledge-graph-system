#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from typing import Any, Dict, List, Optional

from neo4j import GraphDatabase
from openai import OpenAI


# =========================
# 配置区（按你环境改）
# =========================
NEO4J_URL = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "12345678"
NEO4J_DB = "neo4j"

OPENAI_BASE_URL = "http://192.168.100.82:9080/multi_llm/v1"
OPENAI_API_KEY = "NOT_NEED"

# embedding 模型
EMBEDDING_MODEL = "text-embedding-3-small"  # 1536维示例
K_DEFAULT = 10
MIN_SCORE_DEFAULT = 0.70  # 你可以调高/调低

# （可选）让 LLM 生成 Cypher 的 chat 模型
CHAT_MODEL = "mtm_qwen_llm"

# 你的提示词文件路径（如果你想用 LLM 生成 Cypher）
PROMPT_FILE = "cypher_generation.md"


# =========================
# Cypher 模板：向量召回 + 下钻
# =========================
CYPHER_VECTOR_DISEASE_TO_SYMPTOMS = """
CALL db.index.vector.queryNodes('disease_embedding_idx', $k, $embedding)
YIELD node AS d, score
WITH d, score
WHERE $min_score IS NULL OR score >= $min_score
MATCH (d)-[:HAS_SYMPTOM]->(s:SymptomNode)
RETURN
  d.name AS disease,
  d.tcm_code AS disease_code,
  score AS match_score,
  collect(DISTINCT {name: s.name, code: s.symptom_code}) AS symptoms
ORDER BY match_score DESC
LIMIT 50
"""

CYPHER_VECTOR_SYNDROME_TO_SYMPTOMS = """
CALL db.index.vector.queryNodes('syndrome_embedding_idx', $k, $embedding)
YIELD node AS sy, score
WITH sy, score
WHERE $min_score IS NULL OR score >= $min_score
MATCH (sy)-[:HAS_SYMPTOM]->(s:SymptomNode)
RETURN
  sy.name AS syndrome,
  sy.syndrome_code AS syndrome_code,
  score AS match_score,
  collect(DISTINCT {name: s.name, code: s.symptom_code}) AS symptoms
ORDER BY match_score DESC
LIMIT 50
"""

CYPHER_VECTOR_AMBIG_CANDIDATES = """
CALL db.index.vector.queryNodes('disease_embedding_idx', $k, $embedding)
YIELD node AS d, score
RETURN "Disease" AS type, d.name AS name, d.tcm_code AS code, score AS score
UNION ALL
CALL db.index.vector.queryNodes('syndrome_embedding_idx', $k, $embedding)
YIELD node AS sy, score
RETURN "SyndromeNode" AS type, sy.name AS name, sy.syndrome_code AS code, score AS score
UNION ALL
CALL db.index.vector.queryNodes('symptom_embedding_idx', $k, $embedding)
YIELD node AS s, score
RETURN "SymptomNode" AS type, s.name AS name, s.symptom_code AS code, score AS score
ORDER BY score DESC
LIMIT 50
"""


# =========================
# 小工具
# =========================
def looks_like_code(text: str) -> bool:
    # 简单判定：含数字/点/字母混合，且长度>3
    t = text.strip()
    if len(t) <= 3:
        return False
    return bool(re.search(r"[0-9]", t) and re.search(r"[A-Za-z\.]", t))


def classify_intent(q: str) -> str:
    """
    粗分类：disease / syndrome / ambiguous
    你可以按你的业务词表继续增强
    """
    low = q.lower()
    if any(k in q for k in ["证型", "证候", "辨证", "证码"]):
        return "syndrome"
    if any(k in q for k in ["疾病", "病名", "病码", "疾患", "病症"]):
        return "disease"
    # 如果用户明确说“根据证型返回症状/证候”
    if "根据证" in q or "证型" in q or "证候" in q:
        return "syndrome"
    if "根据病" in q or "疾病" in q:
        return "disease"
    return "ambiguous"


def embed_text(client: OpenAI, text: str) -> List[float]:
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=[text])
    return resp.data[0].embedding


def run_query(driver, cypher: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
    with driver.session(database=NEO4J_DB) as s:
        return s.run(cypher, **params).data()


# （可选）把 Neo4j schema 注入到提示词（简化版）
def get_schema_string(driver) -> str:
    """
    用 Neo4j 自带 procedure 获取 schema 概览（只读）
    """
    with driver.session(database=NEO4J_DB) as s:
        labels = [r["label"] for r in s.run("CALL db.labels() YIELD label RETURN label ORDER BY label").data()]
        rels = [r["relationshipType"] for r in s.run("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType ORDER BY relationshipType").data()]

    return (
        "Labels:\n- " + "\n- ".join(labels) + "\n\n"
        "RelationshipTypes:\n- " + "\n- ".join(rels)
    )


def load_prompt(schema: str) -> str:
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        tpl = f.read()
    return tpl.replace("{schema}", schema)


def llm_generate_cypher(client: OpenAI, system_prompt: str, user_q: str) -> str:
    """
    让 LLM 按你的规范只输出 Cypher（可选路径）
    """
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_q},
        ],
        temperature=0,
        max_tokens=800,
    )
    return (resp.choices[0].message.content or "").strip()


def main():
    driver = GraphDatabase.driver(NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASS))
    client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)

    print("========================================")
    print("TCM Neo4j Vector Search App")
    print("输入：")
    print("  - 普通问题：自动用向量检索（推荐）")
    print("  - /cand <文本> ：只做歧义候选（向量 UNION）")
    print("  - /llm <问题> ：让 LLM 按 cypher_generation.md 生成 Cypher（可选）")
    print("  - 退出 / quit / exit")
    print("========================================")

    # 只在你用 /llm 时才需要 schema prompt
    schema_cache: Optional[str] = None
    system_prompt_cache: Optional[str] = None

    while True:
        q = input("\n> ").strip()
        if not q:
            continue
        if q.lower() in ["退出", "quit", "exit"]:
            break

        try:
            if q.startswith("/cand "):
                text = q[len("/cand "):].strip()
                emb = embed_text(client, text)
                rows = run_query(driver, CYPHER_VECTOR_AMBIG_CANDIDATES, {
                    "embedding": emb,
                    "k": K_DEFAULT,
                })
                print("\n--- candidates ---")
                for r in rows[:20]:
                    print(r)
                continue

            if q.startswith("/llm "):
                user_q = q[len("/llm "):].strip()
                if schema_cache is None:
                    schema_cache = get_schema_string(driver)
                    system_prompt_cache = load_prompt(schema_cache)

                cypher = llm_generate_cypher(client, system_prompt_cache or "", user_q)
                print("\n--- LLM generated Cypher ---")
                print(cypher)

                # 你如果让 LLM 生成向量查询，它会用到 $embedding/$k 等参数
                # 我们这里默认用“用户问题本身”的 embedding
                emb = embed_text(client, user_q)

                params = {
                    "embedding": emb,
                    "k": K_DEFAULT,
                    "min_score": MIN_SCORE_DEFAULT,
                    "q": user_q,
                    # 下面这些 name/code 参数你也可以做更精细的抽取
                    "disease_name": user_q,
                    "syndrome_name": user_q,
                    "symptom_name": user_q,
                }

                # 支持多条语句：按分号切（注意不要把 CALL...UNION 误切，一般不会有多分号）
                stmts = [s.strip() for s in cypher.split(";") if s.strip()]
                for i, stmt in enumerate(stmts, 1):
                    rows = run_query(driver, stmt, params)
                    print(f"\n--- result #{i} rows={len(rows)} ---")
                    for r in rows[:10]:
                        print(r)
                continue

            # 默认：自动向量检索 + 下钻
            intent = classify_intent(q)
            emb = embed_text(client, q)

            if intent == "disease":
                rows = run_query(driver, CYPHER_VECTOR_DISEASE_TO_SYMPTOMS, {
                    "embedding": emb,
                    "k": K_DEFAULT,
                    "min_score": MIN_SCORE_DEFAULT,
                })
                print(f"\n--- disease->symptoms rows={len(rows)} ---")
                for r in rows[:5]:
                    print(r)

            elif intent == "syndrome":
                rows = run_query(driver, CYPHER_VECTOR_SYNDROME_TO_SYMPTOMS, {
                    "embedding": emb,
                    "k": K_DEFAULT,
                    "min_score": MIN_SCORE_DEFAULT,
                })
                print(f"\n--- syndrome->symptoms rows={len(rows)} ---")
                for r in rows[:5]:
                    print(r)

            else:
                # 歧义：先候选
                rows = run_query(driver, CYPHER_VECTOR_AMBIG_CANDIDATES, {
                    "embedding": emb,
                    "k": K_DEFAULT,
                })
                print(f"\n--- candidates rows={len(rows)} ---")
                for r in rows[:20]:
                    print(r)

        except Exception as e:
            print("\n❌ 发生错误：")
            print(str(e))
            print("\n提示：")
            print("- 确认你已运行 vector_index_setup.py 写入 embedding")
            print("- 确认 Neo4j 版本支持 VECTOR INDEX 和 db.index.vector.queryNodes")
            print("- 确认你的 base_url 支持 embeddings 接口（/embeddings）")

    driver.close()
    print("\nbye")


if __name__ == "__main__":
    main()
