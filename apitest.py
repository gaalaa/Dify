import requests
import json


def test_dify_direct_message(api_endpoint, api_key):
    """测试Dify API直接发送消息"""
    api_endpoint = api_endpoint.rstrip('/')
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # 尝试直接发送消息 - 无需预先创建会话
    print("\n尝试直接发送消息...")

    # 尝试不同的API路径
    endpoints_to_try = [
        f"{api_endpoint}/chat-messages",
        f"{api_endpoint}/completion-messages"
    ]

    for endpoint in endpoints_to_try:
        print(f"\n测试端点: {endpoint}")

        payload = {
            "inputs": {},
            "query": "什么是知识库?",
            "response_mode": "streaming",
            "stream": True
        }

        try:
            response = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                stream=True
            )

            print(f"状态码: {response.status_code}")

            if response.status_code == 200:
                print("成功! 流式响应内容:")
                content = ""
                for line in response.iter_lines():
                    if line:
                        line_text = line.decode('utf-8')
                        print(f"  {line_text[:100]}...")
                        content += line_text
                        if len(content) > 500:
                            print("  (内容太长，截断显示)")
                            break

                print("\n此API端点工作正常!")
                return endpoint
            else:
                print(f"错误响应: {response.text}")
        except Exception as e:
            print(f"请求出错: {str(e)}")

    # 最后尝试Web API
    web_endpoint = api_endpoint.replace("/v1", "") + "/chat-messages"
    print(f"\n尝试Web API端点: {web_endpoint}")

    try:
        web_response = requests.post(
            web_endpoint,
            headers=headers,
            json={"query": "什么是知识库?"}
        )

        print(f"状态码: {web_response.status_code}")
        print(f"响应: {web_response.text[:200]}...")

        if web_response.status_code == 200:
            print("\n此Web API端点工作正常!")
            return web_endpoint
    except Exception as e:
        print(f"Web API请求出错: {str(e)}")

    return None


# 运行测试
if __name__ == "__main__":
    api_endpoint = input("请输入Dify API端点: ")
    api_key = input("请输入API密钥: ")

    working_endpoint = test_dify_direct_message(api_endpoint, api_key)

    if working_endpoint:
        print(f"\n找到有效的API端点: {working_endpoint}")
        print("使用此端点进行知识库评估")
    else:
        print("\n未找到有效的API端点，请检查API密钥和端点")