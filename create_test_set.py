import pandas as pd
import uuid


def create_simple_test_set(output_path="data/test_dataset.csv"):
    """创建简单测试集"""
    test_cases = [
        {
            "id": str(uuid.uuid4()),
            "query": "1+1=?",
            "expected_answer": "2"
        },
        {
            "id": str(uuid.uuid4()),
            "query": "5*5=?",
            "expected_answer": "25"
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
    df.to_csv(output_path, index=False)
    print(f"测试集已创建并保存至: {output_path}")
    return df