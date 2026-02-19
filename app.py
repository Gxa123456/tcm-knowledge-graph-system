import streamlit as st
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_openai import ChatOpenAI

# ============ 配置 ============
NEO4J_URL = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "12345678"
NEO4J_DATABASE = "neo4j"

LLM_BASE_URL = "http://192.168.100.82:9080/multi_llm/v1"
LLM_MODEL = "mtm_qwen_llm"
LLM_API_KEY = "NOT_NEED"
# ====================================================================

st.set_page_config(page_title="中医知识图谱问答", layout="wide")

@st.cache_resource
def build_chain():
    graph = Neo4jGraph(
        url=NEO4J_URL,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        database=NEO4J_DATABASE,
        enhanced_schema=False,
        refresh_schema=False,
    )

    llm = ChatOpenAI(
        model=LLM_MODEL,
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        temperature=0,
        max_tokens=2000,
    )

    chain = GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph,
        verbose=True,
        allow_dangerous_requests=True,
    )
    return chain

st.title("🧠 中医知识图谱问答（Neo4j + LLM）")

with st.sidebar:
    st.subheader("连接状态")
    st.caption("如果报错，多数是 Neo4j/LLM 不可达或密码不对。")
    if st.button("初始化 / 重连"):
        st.cache_resource.clear()
        st.rerun()

chain = build_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

# 展示历史对话
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# 输入框
question = st.chat_input("输入你的问题，例如：某证候常见症状有哪些？")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("查询中..."):
            try:
                out = chain.invoke({"query": question})
                answer = out.get("result", str(out))
            except Exception as e:
                answer = f"❌ 发生错误：{e}"

        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
