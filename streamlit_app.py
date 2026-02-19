# -*- coding: utf-8 -*-
from typing import Any, Dict, List, Optional, Tuple


import pandas as pd
import streamlit as st
from neo4j import GraphDatabase

# OpenAI-compatible client
from openai import OpenAI

# Optional local embedding fallback
from sentence_transformers import SentenceTransformer  # 不要吞异常，让它直接报真实原因



# =========================
# 默认配置
# =========================
DEFAULT_NEO4J_URL = "bolt://localhost:7687"
DEFAULT_NEO4J_USER = "neo4j"
DEFAULT_NEO4J_PASS = "12345678"
DEFAULT_NEO4J_DB = "neo4j"

DEFAULT_OPENAI_BASE_URL = "http://192.168.100.82:9080/multi_llm/v1"
DEFAULT_OPENAI_API_KEY = "NOT_NEED"
DEFAULT_CHAT_MODEL = "mtm_qwen_llm"

# embeddings: 如果你的 base_url 不支持 /embeddings，会自动 fallback 到本地模型
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"  # 1536维（示例）
DEFAULT_LOCAL_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# 上面的本地模型维度通常是 384（非常常见）
# 如果你已经在 Neo4j 用 1536 维创建了 vector index，那本地模型维度不匹配，会报错（见下方提示）

DEFAULT_K = 10
DEFAULT_MIN_SCORE = 0.70

# Neo4j vector index names (要与你创建的索引一致)
IDX_DISEASE = "disease_embedding_idx"
IDX_SYNDROME = "syndrome_embedding_idx"
IDX_SYMPTOM = "symptom_embedding_idx"

# Cypher templates: vector recall + drill down
CYPHER_VECTOR_DISEASE_TO_SYMPTOMS = f"""
CALL db.index.vector.queryNodes('{IDX_DISEASE}', $k, $embedding)
YIELD node AS d, score
WITH d, score
WHERE $min_score IS NULL OR score >= $min_score
MATCH (d)-[:HAS_SYMPTOM]->(s:SymptomNode)
RETURN
  d.tcm_disease AS disease,
  d.tcm_code AS disease_code,
  score AS match_score,
  collect(DISTINCT {{name: s.name, code: s.symptom_code}}) AS symptoms
ORDER BY match_score DESC
LIMIT 50
"""

CYPHER_VECTOR_SYNDROME_TO_SYMPTOMS = f"""
CALL db.index.vector.queryNodes('{IDX_SYNDROME}', $k, $embedding)
YIELD node AS sy, score
WITH sy, score
WHERE $min_score IS NULL OR score >= $min_score
MATCH (sy)-[:HAS_SYMPTOM]->(s:SymptomNode)
RETURN
  sy.name AS syndrome,
  sy.syndrome_code AS syndrome_code,
  score AS match_score,
  collect(DISTINCT {{name: s.name, code: s.symptom_code}}) AS symptoms
ORDER BY match_score DESC
LIMIT 50
"""

CYPHER_VECTOR_CANDIDATES = f"""
CALL db.index.vector.queryNodes('{IDX_DISEASE}', $k, $embedding)
YIELD node AS d, score
RETURN "Disease" AS type, d.tcm_disease AS name, d.tcm_code AS code, score AS score
UNION ALL
CALL db.index.vector.queryNodes('{IDX_SYNDROME}', $k, $embedding)
YIELD node AS sy, score
RETURN "SyndromeNode" AS type, sy.name AS name, sy.syndrome_code AS code, score AS score
UNION ALL
CALL db.index.vector.queryNodes('{IDX_SYMPTOM}', $k, $embedding)
YIELD node AS s, score
RETURN "SymptomNode" AS type, s.name AS name, s.symptom_code AS code, score AS score
ORDER BY score DESC
LIMIT 50
"""

# quick intent classifier
def classify_intent(q: str) -> str:
    if any(k in q for k in ["证型", "证候", "辨证", "证码"]):
        return "syndrome"
    if any(k in q for k in ["疾病", "病名", "病码", "疾患", "病症"]):
        return "disease"
    if "根据证" in q or "证型" in q or "证候" in q:
        return "syndrome"
    if "根据病" in q or "疾病" in q:
        return "disease"
    return "ambiguous"


@st.cache_resource
def get_neo4j_driver(url: str, user: str, password: str):
    return GraphDatabase.driver(url, auth=(user, password))


def run_query(driver, db: str, cypher: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
    with driver.session(database=db) as s:
        return s.run(cypher, **params).data()


@st.cache_resource
def get_openai_client(base_url: str, api_key: str):
    return OpenAI(base_url=base_url, api_key=api_key)


@st.cache_resource
def get_local_embedder(model_name: str):
    return SentenceTransformer(model_name)



def try_embed_openai(client: OpenAI, model: str, text: str) -> Optional[List[float]]:
    """
    尝试使用 OpenAI-compatible /embeddings。
    如果你的网关不支持，会抛异常；上层会 fallback 到本地 embedding。
    """
    resp = client.embeddings.create(model=model, input=[text])
    emb = resp.data[0].embedding
    return emb


def embed_text(
    openai_client: OpenAI,
    prefer_openai: bool,
    openai_model: str,
    local_model_name: str,
    text: str
) -> Tuple[List[float], str]:
    """
    返回 (embedding, backend_name)
    """
    if prefer_openai:
        try:
            emb = try_embed_openai(openai_client, openai_model, text)
            return emb, f"openai_embeddings({openai_model})"
        except Exception as e:
            # fallback
            st.warning(f"⚠️ embeddings API 不可用或失败，已改用本地 embedding。错误：{e}")

    embedder = get_local_embedder(local_model_name)
    vec = embedder.encode([text], normalize_embeddings=True)[0]
    return vec.tolist(), f"local_sentence_transformers({local_model_name})"


def show_rows_as_table(rows: List[Dict[str, Any]], title: str):
    st.subheader(title)
    if not rows:
        st.info("（无结果）")
        return
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)


def main():
    st.set_page_config(page_title="TCM Neo4j Vector Search", layout="wide")
    st.title("中医知识图谱 · Neo4j 向量检索（Vector Search）")

    with st.sidebar:
        st.header("连接配置")

        neo4j_url = st.text_input("Neo4j URL", DEFAULT_NEO4J_URL)
        neo4j_user = st.text_input("Neo4j Username", DEFAULT_NEO4J_USER)
        neo4j_pass = st.text_input("Neo4j Password", DEFAULT_NEO4J_PASS, type="password")
        neo4j_db = st.text_input("Neo4j Database", DEFAULT_NEO4J_DB)

        st.divider()
        st.header("LLM / Embedding 配置")

        base_url = st.text_input("OpenAI-compatible base_url", DEFAULT_OPENAI_BASE_URL)
        api_key = st.text_input("api_key", DEFAULT_OPENAI_API_KEY, type="password")
        chat_model = st.text_input("Chat model（可选）", DEFAULT_CHAT_MODEL)

        prefer_openai_embeddings = st.checkbox("优先使用 /embeddings（若网关支持）", value=False)
        embedding_model = st.text_input("Embedding model（OpenAI 兼容）", DEFAULT_EMBEDDING_MODEL)
        local_embed_model = st.text_input("本地 embedding 模型（fallback）", DEFAULT_LOCAL_EMBED_MODEL)

        st.divider()
        st.header("向量检索参数")
        k = st.slider("TopK (k)", 1, 50, DEFAULT_K)
        min_score = st.slider("Min score（相似度阈值）", 0.0, 1.0, float(DEFAULT_MIN_SCORE), 0.01)
        st.caption("分数越高越严格；找不到就调低，比如 0.55~0.65。")

    # connect
    try:
        driver = get_neo4j_driver(neo4j_url, neo4j_user, neo4j_pass)
    except Exception as e:
        st.error(f"Neo4j 连接失败：{e}")
        st.stop()

    try:
        oai = get_openai_client(base_url, api_key)
    except Exception as e:
        st.error(f"OpenAI client 初始化失败：{e}")
        st.stop()

    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        q = st.text_input("输入你的问题（病名/证型/症状/口语均可）", value="")
        mode = st.radio(
            "模式",
            options=["自动（疾病/证型/歧义）", "只候选（歧义向量召回）", "疾病→症状", "证型→症状"],
            horizontal=True
        )
        run_btn = st.button("检索", type="primary", use_container_width=True)

    with col2:
        st.markdown("### 检索说明")
        st.write("- 先把输入转成 embedding")
        st.write("- `db.index.vector.queryNodes` 找最相似节点")
        st.write("- 再沿 `HAS_SYMPTOM` 下钻返回症状")
        st.info("如果你还没给节点写入 embedding，向量检索会失败（需要先跑写入脚本）。")

    if run_btn:
        if not q.strip():
            st.warning("请输入问题/关键词")
            st.stop()

        # embed
        embedding, backend = embed_text(
            openai_client=oai,
            prefer_openai=prefer_openai_embeddings,
            openai_model=embedding_model,
            local_model_name=local_embed_model,
            text=q.strip()
        )
        st.success(f"✅ embedding 已生成：{backend} | dim={len(embedding)}")

        params = {"embedding": embedding, "k": k, "min_score": min_score}

        # run
        try:
            if mode == "只候选（歧义向量召回）":
                st.code(CYPHER_VECTOR_CANDIDATES.strip(), language="cypher")
                rows = run_query(driver, neo4j_db, CYPHER_VECTOR_CANDIDATES, {"embedding": embedding, "k": k})
                show_rows_as_table(rows, "候选结果（type/name/code/score）")

            elif mode == "疾病→症状":
                st.code(CYPHER_VECTOR_DISEASE_TO_SYMPTOMS.strip(), language="cypher")
                rows = run_query(driver, neo4j_db, CYPHER_VECTOR_DISEASE_TO_SYMPTOMS, params)
                show_rows_as_table(rows, "疾病 → 症状（向量召回 + 下钻）")

            elif mode == "证型→症状":
                st.code(CYPHER_VECTOR_SYNDROME_TO_SYMPTOMS.strip(), language="cypher")
                rows = run_query(driver, neo4j_db, CYPHER_VECTOR_SYNDROME_TO_SYMPTOMS, params)
                show_rows_as_table(rows, "证型 → 症状（向量召回 + 下钻）")

            else:
                intent = classify_intent(q)
                if intent == "disease":
                    st.code(CYPHER_VECTOR_DISEASE_TO_SYMPTOMS.strip(), language="cypher")
                    rows = run_query(driver, neo4j_db, CYPHER_VECTOR_DISEASE_TO_SYMPTOMS, params)
                    show_rows_as_table(rows, "自动判定：疾病 → 症状")
                elif intent == "syndrome":
                    st.code(CYPHER_VECTOR_SYNDROME_TO_SYMPTOMS.strip(), language="cypher")
                    rows = run_query(driver, neo4j_db, CYPHER_VECTOR_SYNDROME_TO_SYMPTOMS, params)
                    show_rows_as_table(rows, "自动判定：证型 → 症状")
                else:
                    st.code(CYPHER_VECTOR_CANDIDATES.strip(), language="cypher")
                    rows = run_query(driver, neo4j_db, CYPHER_VECTOR_CANDIDATES, {"embedding": embedding, "k": k})
                    show_rows_as_table(rows, "自动判定：歧义 → 候选列表")

        except Exception as e:
            st.error(f"查询失败：{e}")
            st.markdown("### 常见原因排查")
            st.write("1) 你还没创建 VECTOR INDEX 或索引名不一致（需与代码里的 IDX_* 相同）")
            st.write("2) 节点还没有 embedding 属性（需要先批量写入 embedding）")
            st.write("3) embedding 维度与索引 dimensions 不一致（比如索引用 1536，本地模型输出 384）")
            st.write("4) Neo4j 版本不支持 vector index 或未启用相关功能")
            st.stop()

    st.divider()
    with st.expander("✅ 快速检查（索引/维度/节点 embedding）", expanded=False):
        if st.button("检查索引是否存在"):
            try:
                rows = run_query(driver, neo4j_db, "SHOW INDEXES YIELD name, type, entityType, labelsOrTypes, properties RETURN name, type, entityType, labelsOrTypes, properties", {})
                show_rows_as_table(rows, "Neo4j Indexes")
            except Exception as e:
                st.error(f"SHOW INDEXES 失败：{e}")

        if st.button("抽样检查 embedding 维度"):
            cy = """
            MATCH (d:Disease)
            WHERE d.embedding IS NOT NULL
            RETURN d.tcm_disease AS name, size(d.embedding) AS dim
            LIMIT 10
            """
            try:
                rows = run_query(driver, neo4j_db, cy, {})
                show_rows_as_table(rows, "Disease embedding dim sample")
            except Exception as e:
                st.error(f"抽样失败：{e}")


if __name__ == "__main__":
    main()
