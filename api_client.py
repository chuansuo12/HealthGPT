"""
更简洁的 API 客户端示例 - 使用 requests 直接调用
"""

import requests
import base64
import json

API_URL = "http://localhost:5011/api/predict"


def call_healthgpt(image_path, question, task="Analyze Image", model="HealthGPT-M3"):
    """
    调用 HealthGPT API
    
    参数:
        image_path: 图片文件路径
        question: 问题文本
        task: "Analyze Image" 或 "Generate Image"
        model: "HealthGPT-M3" 或 "HealthGPT-L14"
    
    返回:
        API 响应结果
    """
    # 读取图片并转换为 base64
    with open(image_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode('utf-8')
    
    # 构建请求
    payload = {
        "data": [
            task,           # option: "Analyze Image" 或 "Generate Image"
            model,          # model_name: "HealthGPT-M3" 或 "HealthGPT-L14"
            question,       # text: 问题文本
            f"data:image/jpeg;base64,{image_base64}"  # image: base64编码的图片
        ]
    }
    
    # 发送请求
    response = requests.post(API_URL, json=payload)
    response.raise_for_status()
    
    return response.json()


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
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    # 提取文本答案（Analyze Image 返回的第一个元素）
    if "data" in result and len(result["data"]) > 0:
        answer = result["data"][0]
        print(f"\n答案: {answer}")

