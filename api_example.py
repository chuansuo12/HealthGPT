"""
示例：如何通过 API 调用 HealthGPT（Gradio 服务）

推荐使用 gradio_client（会自动适配 Gradio 版本和实际 endpoint），避免硬编码 /api/predict 之类路径。
"""

from gradio_client import Client, handle_file

# 服务器地址：如果服务器在其他机器，请改成对应 IP
SERVER_URL = "http://localhost:5011"


def analyze_image_api(image_path: str, question: str, model_name: str = "HealthGPT-M3") -> str:
    client = Client(SERVER_URL)
    # app.py 里给 click 事件绑定了 api_name="process_input"
    result = client.predict(
        "Analyze Image",
        model_name,
        question,
        handle_file(image_path),
        api_name="/process_input",
    )
    # app.py 现在只有两个输出：[text_output, image_output]
    return result[0]


def generate_image_api(image_path: str, question: str, model_name: str = "HealthGPT-M3"):
    client = Client(SERVER_URL)
    result = client.predict(
        "Generate Image",
        model_name,
        question,
        handle_file(image_path),
        api_name="/process_input",
    )
    # 返回第二个输出（图片）
    return result[1]


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
    #     # gradio_client 通常会返回一个文件路径或带 name 的对象（取决于 Gradio 版本）
    #     print(generated_image_base64)
    #     print("图像生成完成！")
    # except Exception as e:
    #     print(f"错误: {e}")

