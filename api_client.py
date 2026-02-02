"""
更简洁的 API 客户端示例 - 使用 requests 直接调用
"""

import json
from gradio_client import Client, handle_file

SERVER_URL = "http://localhost:5011"


def call_healthgpt(image_path, question, task="Analyze Image", model="HealthGPT-M3"):
    """
    调用 HealthGPT（Gradio）API。

    说明：
    - 我们在 app.py 的 click 事件上绑定了 api_name="process_input"
    - 用 gradio_client 可以自动适配实际的 HTTP endpoint（避免 /api/predict 不兼容）
    """
    client = Client(SERVER_URL)
    return client.predict(
        task,
        model,
        question,
        handle_file(image_path),
        api_name="/process_input",
    )


# 使用示例
if __name__ == "__main__":
    # 分析图像
    result = call_healthgpt(
        image_path="/workspace/brain.jpg",
        question="What problems are there with this brain CT?",
        task="Analyze Image",
        model="HealthGPT-M3"
    )
    
    print("API 响应:")
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    
    # 提取文本答案（Analyze Image 返回的第一个元素）
    if isinstance(result, (list, tuple)) and len(result) > 0:
        print(f"\n答案: {result[0]}")

