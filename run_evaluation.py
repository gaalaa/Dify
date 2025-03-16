from evaluator import DifyEvaluator
from create_test_set import create_simple_test_set
import os


def main():
    # 检查是否已有测试集，没有则创建
    test_dataset_path = "data/test_dataset.csv"
    if not os.path.exists(test_dataset_path):
        create_simple_test_set(test_dataset_path)

    # 获取API信息
    api_endpoint = input("请输入Dify API端点 (例如: https://api.dify.ai/v1 或 您的自托管地址): ")
    api_key = input("请输入Dify API密钥: ")

    # 创建评估器
    evaluator = DifyEvaluator(api_endpoint, api_key)

    # 运行评估
    print(f"开始评估...")
    evaluator.evaluate_dataset(test_dataset_path)

    print("\n评估已完成，您可以在results目录查看结果文件")


if __name__ == "__main__":
    main()