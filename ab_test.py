from evaluator import DifyEvaluator
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime


def run_ab_test(api_endpoint_a, api_key_a, api_endpoint_b, api_key_b, test_dataset_path):
    """运行A/B测试比较两个知识库版本"""
    print("开始A/B测试...")

    # 评估版本A
    print("\n评估版本A...")
    evaluator_a = DifyEvaluator(api_endpoint_a, api_key_a)
    results_a = evaluator_a.evaluate_dataset(test_dataset_path)

    # 评估版本B
    print("\n评估版本B...")
    evaluator_b = DifyEvaluator(api_endpoint_b, api_key_b)
    results_b = evaluator_b.evaluate_dataset(test_dataset_path)

    # 比较结果
    compare_and_visualize(results_a, results_b)


def compare_and_visualize(results_a, results_b):
    """比较两个版本的结果并可视化"""
    # 确保结果目录存在
    os.makedirs("results", exist_ok=True)

    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 转换为DataFrame
    df_a = pd.DataFrame(results_a)
    df_b = pd.DataFrame(results_b)

    # 计算平均指标
    metrics = {
        "平均响应时间(秒)": (df_a["response_time"].mean(), df_b["response_time"].mean()),
        "平均检索文档数": (df_a["retrieval_count"].mean(), df_b["retrieval_count"].mean())
    }

    if "similarity_score" in df_a and "similarity_score" in df_b:
        metrics["平均相似度分数"] = (df_a["similarity_score"].mean(), df_b["similarity_score"].mean())

    # 创建比较可视化
    plt.figure(figsize=(10, 6))

    # 绘制对比条形图
    x = range(len(metrics))
    width = 0.35

    plt.bar([i - width / 2 for i in x], [m[0] for m in metrics.values()], width, label='版本A')
    plt.bar([i + width / 2 for i in x], [m[1] for m in metrics.values()], width, label='版本B')

    plt.xlabel('评估指标')
    plt.ylabel('值')
    plt.title('知识库版本对比')
    plt.xticks(x, metrics.keys())
    plt.legend()

    # 保存图表
    chart_path = f"results/ab_test_chart_{timestamp}.png"
    plt.savefig(chart_path)

    # 保存比较结果
    comparison = {
        "metrics": {k: {"version_a": v[0], "version_b": v[1],
                        "diff_percent": (v[1] - v[0]) / v[0] * 100 if v[0] else 0}
                    for k, v in metrics.items()}
    }

    comparison_path = f"results/ab_test_{timestamp}.json"
    import json
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2)

    # 打印结果
    print("\nA/B测试结果比较:")
    for metric, (val_a, val_b) in metrics.items():
        diff_percent = (val_b - val_a) / val_a * 100 if val_a else 0
        print(f"{metric}: 版本A={val_a:.2f}, 版本B={val_b:.2f}, 差异={diff_percent:.2f}%")

    print(f"\n对比图表已保存至: {chart_path}")
    print(f"对比数据已保存至: {comparison_path}")


if __name__ == "__main__":
    # 测试集路径
    test_dataset_path = "data/test_dataset.csv"

    # 获取API信息
    print("请输入版本A的API信息:")
    api_endpoint_a = input("版本A的API端点: ")
    api_key_a = input("版本A的API密钥: ")

    print("\n请输入版本B的API信息:")
    api_endpoint_b = input("版本B的API端点: ")
    api_key_b = input("版本B的API密钥: ")

    # 运行A/B测试
    run_ab_test(api_endpoint_a, api_key_a, api_endpoint_b, api_key_b, test_dataset_path)