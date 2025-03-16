import requests
import json
import pandas as pd
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from datetime import datetime
import uuid


class DifyBackendEvaluator:
    def __init__(self, api_endpoint, api_key):
        """初始化后端服务评估器"""
        self.api_endpoint = api_endpoint.rstrip('/')
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.results = []

    def evaluate_query(self, query, expected_answer=None):
        """使用后端服务API评估单个查询"""
        start_time = time.time()

        # 后端服务API端点
        api_url = f"{self.api_endpoint}/chat-messages"

        # 后端服务API所需的payload格式
        payload = {
            "inputs": {},
            "query": query,
            "user": str(uuid.uuid4()),  # 生成随机用户ID
            "conversation_id": str(uuid.uuid4()),  # 生成随机会话ID
            "response_mode": "blocking",
            "stream": False
        }

        try:
            response = requests.post(
                api_url,
                headers=self.headers,
                json=payload
            )

            response_time = time.time() - start_time

            if response.status_code == 200:
                data = response.json()
                answer = data.get("answer", "")

                # 检索文档可能在不同位置，尝试多种可能的路径
                retrieval_docs = data.get("retrieval_documents", [])
                if not retrieval_docs and "metadata" in data:
                    retrieval_docs = data.get("metadata", {}).get("retrieval_documents", [])

                result = {
                    "query": query,
                    "expected_answer": expected_answer,
                    "actual_answer": answer,
                    "response_time": response_time,
                    "retrieval_count": len(retrieval_docs),
                }

                # 提取检索的文档内容
                if retrieval_docs:
                    doc_contents = []
                    for doc in retrieval_docs:
                        if isinstance(doc, dict):
                            content = None
                            # 尝试不同可能的内容路径
                            if "document" in doc and "content" in doc["document"]:
                                content = doc["document"]["content"]
                            elif "content" in doc:
                                content = doc["content"]

                            if content:
                                doc_contents.append(content[:100] + "...")

                    result["retrieval_docs"] = doc_contents

                # 计算相似度分数(如果有预期答案)
                if expected_answer and answer:
                    from sklearn.feature_extraction.text import TfidfVectorizer
                    from sklearn.metrics.pairwise import cosine_similarity

                    vectorizer = TfidfVectorizer()
                    try:
                        tfidf_matrix = vectorizer.fit_transform([expected_answer, answer])
                        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
                        result["similarity_score"] = round(similarity * 10, 2)
                    except Exception as e:
                        print(f"计算相似度失败: {str(e)}")
                        result["similarity_score"] = 0

                return result
            else:
                error_detail = f"API Error: {response.status_code}"
                try:
                    error_json = response.json()
                    if "message" in error_json:
                        error_detail += f" - {error_json['message']}"
                except:
                    error_detail += f" - {response.text[:200]}"

                return {
                    "query": query,
                    "error": error_detail,
                    "response_time": response_time
                }
        except Exception as e:
            return {
                "query": query,
                "error": f"Exception: {str(e)}",
                "response_time": time.time() - start_time
            }

    def evaluate_dataset(self, dataset_path):
        """评估整个测试集"""
        try:
            # 加载测试集
            if dataset_path.endswith('.csv'):
                test_data = pd.read_csv(dataset_path)
            elif dataset_path.endswith('.json'):
                test_data = pd.read_json(dataset_path)
            else:
                raise ValueError("测试集必须是CSV或JSON格式")

            # 确保测试集包含必要的列
            if "query" not in test_data.columns:
                raise ValueError("测试集必须包含'query'列")

            # 评估每个查询
            for _, row in tqdm(test_data.iterrows(), total=len(test_data), desc="评估进度"):
                query = row["query"]
                expected_answer = row["expected_answer"] if "expected_answer" in row else None

                result = self.evaluate_query(query, expected_answer)
                self.results.append(result)

            # 生成报告
            self.generate_report()

            return self.results
        except Exception as e:
            print(f"评估过程出错: {str(e)}")
            return []

    def generate_report(self):
        """生成评估报告"""
        if not self.results:
            print("没有评估结果，无法生成报告")
            return

        # 确保结果目录存在
        os.makedirs("results", exist_ok=True)

        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存详细结果
        df_results = pd.DataFrame(self.results)
        results_path = f"results/evaluation_{timestamp}.csv"
        df_results.to_csv(results_path, index=False)

        # 计算汇总指标
        summary = {
            "total_queries": len(self.results),
            "successful_queries": sum(1 for r in self.results if "error" not in r),
            "failed_queries": sum(1 for r in self.results if "error" in r),
        }

        # 只计算成功查询的指标
        successful_results = [r for r in self.results if "error" not in r]
        if successful_results:
            df_successful = pd.DataFrame(successful_results)
            summary.update({
                "avg_response_time": df_successful["response_time"].mean(),
                "avg_retrieval_count": df_successful[
                    "retrieval_count"].mean() if "retrieval_count" in df_successful else 0,
                "avg_similarity": df_successful[
                    "similarity_score"].mean() if "similarity_score" in df_successful else None
            })

        # 保存汇总结果
        summary_path = f"results/summary_{timestamp}.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        # 生成简单可视化
        plt.figure(figsize=(12, 8))

        if successful_results:
            # 响应时间图
            plt.subplot(2, 1, 1)
            plt.bar(range(len(successful_results)),
                    [r.get("response_time", 0) for r in successful_results],
                    color="skyblue")
            plt.xlabel("查询ID")
            plt.ylabel("响应时间(秒)")
            plt.title("响应时间分布(仅成功查询)")

            # 相似度分数图(如果有)
            if "similarity_score" in df_successful:
                plt.subplot(2, 1, 2)
                plt.bar(range(len(successful_results)),
                        df_successful["similarity_score"].fillna(0),
                        color="lightgreen")
                plt.xlabel("查询ID")
                plt.ylabel("相似度分数(0-10)")
                plt.title("回答质量相似度分数")
        else:
            plt.text(0.5, 0.5, "没有成功的查询",
                     horizontalalignment='center', verticalalignment='center',
                     transform=plt.gca().transAxes, fontsize=14)

        # 保存图表
        plt.tight_layout()
        chart_path = f"results/chart_{timestamp}.png"
        plt.savefig(chart_path)

        print(f"\n评估完成!")
        print(f"总查询数: {summary['total_queries']}")
        print(f"成功查询数: {summary['successful_queries']}")
        print(f"失败查询数: {summary['failed_queries']}")

        if successful_results:
            print(f"平均响应时间: {summary.get('avg_response_time', 0):.2f}秒")
            print(f"平均检索文档数: {summary.get('avg_retrieval_count', 0):.2f}")
            if summary.get('avg_similarity') is not None:
                print(f"平均相似度分数: {summary['avg_similarity']:.2f}/10")

        print(f"\n详细结果已保存至: {results_path}")
        print(f"汇总报告已保存至: {summary_path}")
        print(f"可视化图表已保存至: {chart_path}")

        return {
            "summary": summary,
            "results_path": results_path,
            "chart_path": chart_path
        }
