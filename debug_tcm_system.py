#!/usr/bin/env python3
"""
中医知识图谱系统 - 调试版本
包含详细的错误信息和连接测试
"""
import traceback
from neo4j import GraphDatabase

from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
from langchain_openai import ChatOpenAI


class DebugTCMSystem:
    def __init__(self):
        # 硬编码所有配置
        self.config = {
            "neo4j": {
                "url": "bolt://localhost:7687",
                "username": "neo4j",
                "password": "12345678",
                # 可选：如果你有多数据库，指定 database；没有就留 neo4j
                "database": "neo4j",
            },
            "llm": {
                "base_url": "http://192.168.100.82:9080/multi_llm/v1",
                "model_name": "mtm_qwen_llm",
                "api_key": "NOT_NEED"
            }
        }

        self.graph: Neo4jGraph | None = None
        self.chain: GraphCypherQAChain | None = None

    # -------------------------
    # 连接测试
    # -------------------------
    def test_connections(self):
        """测试所有连接"""
        print("🔍 开始连接测试...")
        print("=" * 50)

        print("1) 测试 Neo4j 连接...")
        neo4j_ok = self._test_neo4j()

        print("\n2) 测试 LLM 连接...")
        llm_ok = self._test_llm()

        print("\n" + "=" * 50)
        print("测试结果汇总:")
        print(f"  Neo4j: {'✅ 通过' if neo4j_ok else '❌ 失败'}")
        print(f"  LLM: {'✅ 通过' if llm_ok else '❌ 失败'}")
        print("=" * 50)

        return neo4j_ok and llm_ok

    def _test_neo4j(self):
        """测试 Neo4j 连接"""
        neo = self.config["neo4j"]
        try:
            driver = GraphDatabase.driver(
                neo["url"],
                auth=(neo["username"], neo["password"])
            )
            with driver.session(database=neo.get("database", None)) as session:
                # 1) 基本连接
                v = session.run("RETURN 1 AS test").single()
                if v and v["test"] == 1:
                    print("  ✅ Neo4j 基本连接成功")

                # 2) 数据库版本/当前数据库
                rec = session.run(
                    "CALL dbms.components() YIELD name, versions "
                    "RETURN name, versions[0] AS version LIMIT 1"
                ).single()
                if rec:
                    print(f"  📊 组件: {rec['name']}, 版本: {rec['version']}")

                # 3) 总节点数
                rec = session.run("MATCH (n) RETURN count(n) AS c").single()
                print(f"  📊 总节点数: {rec['c'] if rec else '未知'}")

                # 4) 关键 label 统计
                for label in ["Disease", "SymptomNode", "SyndromeNode"]:
                    rec = session.run(f"MATCH (n:{label}) RETURN count(n) AS c").single()
                    c = rec["c"] if rec else 0
                    if c > 0:
                        print(f"  ✅ 找到 {label}: {c} 个节点")
                    else:
                        print(f"  ⚠️  未找到 {label} 节点")

            driver.close()
            return True

        except Exception as e:
            print(f"  ❌ Neo4j 连接失败: {e}")
            traceback.print_exc()
            return False

    def _test_llm(self):
        """测试 LLM 连接"""
        llm_cfg = self.config["llm"]
        try:
            # 显式传 base_url/api_key（比靠环境变量更稳）
            llm = ChatOpenAI(
                model=llm_cfg["model_name"],
                base_url=llm_cfg["base_url"],
                api_key=llm_cfg["api_key"],
                temperature=0,
                max_tokens=80,
            )

            resp = llm.invoke("你好")
            if getattr(resp, "content", ""):
                print("  ✅ LLM 连接成功")
                print(f"  📊 模型响应: {resp.content[:50]}...")
                return True

            print("  ❌ LLM 返回空响应")
            return False

        except Exception as e:
            print(f"  ❌ LLM 连接失败: {e}")
            traceback.print_exc()
            return False

    # -------------------------
    # 初始化系统
    # -------------------------
    def initialize(self):
        print("\n" + "=" * 50)
        print("初始化中医知识图谱系统...")
        print("=" * 50)

        try:
            neo = self.config["neo4j"]
            llm_cfg = self.config["llm"]

            self.graph = Neo4jGraph(
                url=neo["url"],
                username=neo["username"],
                password=neo["password"],
                database=neo.get("database", "neo4j"),
                enhanced_schema=False,
                refresh_schema=False,
            )

            # 这段你如果不想看到 apoc 报错，也可以直接删掉
            try:
                self.graph.refresh_schema()
                print("  ✅ Neo4j schema 已刷新")
            except Exception:
                # 静默，不打印
                pass

            llm = ChatOpenAI(
                model=llm_cfg["model_name"],
                base_url=llm_cfg["base_url"],
                api_key=llm_cfg["api_key"],
                temperature=0,
                max_tokens=2000,
            )

            self.chain = GraphCypherQAChain.from_llm(
                llm=llm,
                graph=self.graph,
                verbose=True,
                allow_dangerous_requests=True,
            )

            print("\n✅ 系统初始化成功！")
            return True

        except Exception as e:
            print(f"\n❌ 系统初始化失败: {e}")
            traceback.print_exc()
            return False

    # -------------------------
    # 运行交互
    # -------------------------
    def run(self):
        if not self.initialize():
            return

        print("\n" + "=" * 50)
        print("开始查询！输入 '退出' 结束")
        print("=" * 50)

        while True:
            try:
                cmd = input("\n输入命令: ").strip()

                if cmd.lower() in ["退出", "quit", "exit"]:
                    print("谢谢使用！")
                    break

                if not cmd:
                    continue

                if not self.chain:
                    print("❌ chain 未初始化")
                    continue

                print(f"\n🔍 执行查询: {cmd}")
                result = self.chain.invoke({"query": cmd})

                if isinstance(result, dict) and "result" in result:
                    print("\n📋 查询结果:")
                    print("=" * 50)
                    print(result["result"])
                    print("=" * 50)
                else:
                    print("❌ 查询失败（未返回 result 字段）")
                    print(result)

            except KeyboardInterrupt:
                print("\n\n操作已取消")
                break
            except Exception as e:
                print(f"❌ 发生错误: {e}")
                traceback.print_exc()

def main():
    print("中医知识图谱系统")
    system = DebugTCMSystem()
    system.run()


if __name__ == "__main__":
    main()
