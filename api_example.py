"""
示例：如何通过 API 调用 HealthGPT 服务

使用前请确保：
1. 服务器已启动：python app.py
2. 服务器运行在 http://0.0.0.0:5011
"""

import requests
import base64
from PIL import Image
import io
import json

# 服务器地址
API_URL = "http://localhost:5011"  # 如果服务器在其他机器，请修改为对应IP


def image_to_base64(image_path):
    """将图片转换为 base64 编码"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def analyze_image_api(image_path, question, model_name="HealthGPT-M3"):
    """
    通过 API 调用 Analyze Image 功能
    
    Args:
        image_path: 图片路径
        question: 问题文本
        model_name: 模型名称 ("HealthGPT-M3" 或 "HealthGPT-L14")
    
    Returns:
        返回的文本答案
    """
    # 将图片转换为 base64
    image_base64 = image_to_base64(image_path)
    
    # 构建请求数据
    data = {
        "data": [
            "Analyze Image",  # option
            model_name,        # model_name
            question,          # text
            f"data:image/jpeg;base64,{image_base64}"  # image (base64格式)
        ]
    }
    
    # 发送 POST 请求
    response = requests.post(
        f"{API_URL}/api/predict",
        json=data,
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        result = response.json()
        # Gradio API 返回格式: {"data": [text_output, image_output, ...]}
        if "data" in result and len(result["data"]) > 0:
            return result["data"][0]  # 返回文本答案
        return result
    else:
        raise Exception(f"API request failed: {response.status_code}, {response.text}")


def generate_image_api(image_path, question, model_name="HealthGPT-M3"):
    """
    通过 API 调用 Generate Image 功能
    
    Args:
        image_path: 图片路径
        question: 问题文本
        model_name: 模型名称 ("HealthGPT-M3")
    
    Returns:
        生成的图片 (base64编码)
    """
    # 将图片转换为 base64
    image_base64 = image_to_base64(image_path)
    
    # 构建请求数据
    data = {
        "data": [
            "Generate Image",  # option
            model_name,        # model_name
            question,          # text
            f"data:image/jpeg;base64,{image_base64}"  # image (base64格式)
        ]
    }
    
    # 发送 POST 请求
    response = requests.post(
        f"{API_URL}/api/predict",
        json=data,
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        result = response.json()
        # Gradio API 返回格式: {"data": [text_output, image_output, ...]}
        if "data" in result and len(result["data"]) > 1:
            # 返回生成的图片 (base64格式)
            return result["data"][1]
        return result
    else:
        raise Exception(f"API request failed: {response.status_code}, {response.text}")


def save_base64_image(base64_str, output_path):
    """将 base64 编码的图片保存到文件"""
    # 移除 data URL 前缀（如果有）
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]
    
    image_data = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(image_data))
    image.save(output_path)
    print(f"Image saved to: {output_path}")


# 示例使用
if __name__ == "__main__":
    # 示例 1: Analyze Image
    print("=" * 50)
    print("示例 1: Analyze Image (图像分析)")
    print("=" * 50)
    
    image_path = "/workspace/brain.jpg"  # 替换为你的图片路径
    question = "What problems are there with this brain CT?"
    
    try:
        answer = analyze_image_api(image_path, question, model_name="HealthGPT-M3")
        print(f"\n问题: {question}")
        print(f"\n答案:\n{answer}\n")
    except Exception as e:
        print(f"错误: {e}")
    
    # 示例 2: Generate Image (如果需要)
    # print("=" * 50)
    # print("示例 2: Generate Image (图像生成)")
    # print("=" * 50)
    # 
    # image_path = "/workspace/input_image.jpg"
    # question = "Reconstruct the image."
    # 
    # try:
    #     generated_image_base64 = generate_image_api(image_path, question, model_name="HealthGPT-M3")
    #     save_base64_image(generated_image_base64, "output_generated.png")
    #     print("图像生成完成！")
    # except Exception as e:
    #     print(f"错误: {e}")

