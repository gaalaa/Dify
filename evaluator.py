import requests
import json
import pandas as pd
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from datetime import datetime


class DifyEvaluator:
    def __init__(self, api_endpoint, api_key):
        """初始化评估器"""
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.results = []

    def evaluate_query(self, query, expected_answer=None):
        """评估单个查询"""
        start_time = time.time()
        response_time = 0  # 初始化变量

        payload = {
            "inputs": {},
            "query": query,
            "user": "evaluator",  # 添加用户标识符
            "response_mode": "streaming",  # 使用streaming模式
            "conversation_id": "",
            "stream": True  # 设置为流式处理
        }

        try:
            response = requests.post(
                f"{self.api_endpoint}/chat-messages",
                headers=self.headers,
                json=payload,
                stream=True  # 设置为流式处理
            )

            response_time = time.time() - start_time  # 计算响应时间

            if response.status_code == 200:
                answer = ""
                retrieval_docs = []

                # 处理流式响应
                for line in response.iter_lines():
                    if line:
                        line_text = line.decode('utf-8')
                        # 移除SSE前缀"data: "
                        if line_text.startswith('data: '):
                            line_json = line_text[6:]
                            try:
                                data = json.loads(line_json)
                                if "answer" in data:
                                    answer += data.get("answer", "")
                                if "retrieval_documents" in data and not retrieval_docs:
                                    retrieval_docs = data.get("retrieval_documents", [])
                            except json.JSONDecodeError:
                                continue

                # 计算完整响应处理时间
                total_response_time = time.time() - start_time

                result = {
                    "query": query,
                    "expected_answer": expected_answer,
                    "actual_answer": answer,
                    "response_time": total_response_time,  # 使用总响应时间
                    "retrieval_count": len(retrieval_docs),
                    "retrieval_docs": [doc.get("document", {}).get("content", "")[:100] + "..."
                                       for doc in retrieval_docs] if retrieval_docs else []
                }

                if expected_answer and answer:
                    # 简单相似度计算 (可改进)
                    from sklearn.feature_extraction.text import TfidfVectorizer
                    from sklearn.metrics.pairwise import cosine_similarity

                    vectorizer = TfidfVectorizer()
                    try:
                        tfidf_matrix = vectorizer.fit_transform([expected_answer, answer])
                        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
                        result["similarity_score"] = round(similarity * 10, 2)  # 转为10分制
                    except:
                        result["similarity_score"] = 0

                return result
            else:
                error_detail = response.json() if response.content else "No error details"
                return {
                    "query": query,
                    "error": f"API Error: {response.status_code}",
                    "error_detail": error_detail,
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
            "avg_response_time": df_results["response_time"].mean() if "response_time" in df_results else 0,
            "avg_retrieval_count": df_results["retrieval_count"].mean() if "retrieval_count" in df_results else 0,
            "avg_similarity": df_results["similarity_score"].mean() if "similarity_score" in df_results else None
        }

        # 保存汇总结果
        summary_path = f"results/summary_{timestamp}.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        # 生成简单可视化
        plt.figure(figsize=(12, 8))

        # 响应时间图
        plt.subplot(2, 1, 1)
        plt.bar(range(len(self.results)),
                [r.get("response_time", 0) for r in self.results],
                color="skyblue")
        plt.xlabel("查询ID")
        plt.ylabel("响应时间(秒)")
        plt.title("响应时间分布")

        # 相似度分数图(如果有)
        if "similarity_score" in df_results:
            plt.subplot(2, 1, 2)
            plt.bar(range(len(self.results)),
                    df_results["similarity_score"].fillna(0),
                    color="lightgreen")
            plt.xlabel("查询ID")
            plt.ylabel("相似度分数(0-10)")
            plt.title("回答质量相似度分数")

        # 保存图表
        plt.tight_layout()
        chart_path = f"results/chart_{timestamp}.png"
        plt.savefig(chart_path)

        print(f"\n评估完成!")
        print(f"总查询数: {summary['total_queries']}")
        print(f"平均响应时间: {summary['avg_response_time']:.2f}秒")
        print(f"平均检索文档数: {summary['avg_retrieval_count']:.2f}")
        if summary['avg_similarity'] is not None:
            print(f"平均相似度分数: {summary['avg_similarity']:.2f}/10")
        print(f"\n详细结果已保存至: {results_path}")
        print(f"汇总报告已保存至: {summary_path}")
        print(f"可视化图表已保存至: {chart_path}")

        return {
            "summary": summary,
            "results_path": results_path,
            "chart_path": chart_path
        }