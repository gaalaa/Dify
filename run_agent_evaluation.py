from agent_evaluator import DifyAgentEvaluator
import os
import pandas as pd
import uuid


def create_simple_test_set(output_path="data/test_dataset.csv"):
    """创建简单测试集"""
    test_cases = [
        {
            "id": str(uuid.uuid4()),
            "query": "什么是知识库?",
            "expected_answer": "知识库是组织和存储结构化与非结构化信息的系统，便于检索和使用。"
        },
        {
            "id": str(uuid.uuid4()),
            "query": "如何提高知识库的检索效果?",
            "expected_answer": "提高知识库检索效果可以通过优化分块策略、使用混合检索、选择更好的嵌入模型等方法。"
        },
        {
            "id": str(uuid.uuid4()),
            "query": "Dify知识库有哪些特点?",
            "expected_answer": "Dify知识库支持多种文件格式，提供混合检索、自定义提示词、嵌入模型选择等功能，易于使用。"
        },
        {
            "id": str(uuid.uuid4()),
            "query": "知识库评估应该关注哪些指标?",
            "expected_answer": "知识库评估应关注检索准确性、回答质量、响应时间和用户满意度等指标。"
        },
        {
            "id": str(uuid.uuid4()),
            "query": "如何处理原始文档构建知识库?",
            "expected_answer": "处理原始文档构建知识库可以通过智能分块、混合检索策略、上下文压缩和优化提示词来提高质量。"
        }
    ]

    df = pd.DataFrame(test_cases)

    # 确保目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"测试集已创建并保存至: {output_path}")
    return df


def main():
    # 检查是否已有测试集，没有则创建
    test_dataset_path = "data/test_dataset.csv"
    if not os.path.exists(test_dataset_path):
        create_simple_test_set(test_dataset_path)

    # 获取API信息
    print("使用Dify Agent应用API进行评估")
    api_endpoint = input("请输入Dify API端点 (例如: https://api.dify.ai/v1): ")
    api_key = input("请输入后端服务API密钥: ")

    # 创建评估器
    evaluator = DifyAgentEvaluator(api_endpoint, api_key)

    # 运行评估
    print(f"开始评估...")
    evaluator.evaluate_dataset(test_dataset_path)

    print("\n评估已完成，您可以在results目录查看结果文件")


if __name__ == "__main__":
    main()