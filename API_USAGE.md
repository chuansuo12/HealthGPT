# HealthGPT API 使用指南

## 方式一：Web UI 界面（最简单）

### 1. 启动服务器
```bash
python app.py
```

### 2. 访问 Web 界面
打开浏览器访问：`http://localhost:5011` 或 `http://你的服务器IP:5011`

### 3. 使用步骤
1. 选择任务类型：**Analyze Image**（图像分析）
2. 选择模型：**HealthGPT-M3**
3. 上传图片
4. 输入问题（例如："What problems are there with this brain CT?"）
5. 点击 **🚀 Process** 按钮
6. 等待结果返回

---

## 方式二：API 调用（程序化调用）

### 1. 启动服务器
```bash
python app.py
```

服务器启动后，API 端点会自动可用：
- Web UI: `http://localhost:5011`
- API 文档: `http://localhost:5011/docs`
- API 端点: `http://localhost:5011/api/predict`

### 2. 使用 Python 客户端

#### 方法 A：使用提供的示例脚本

```bash
# 修改 api_example.py 或 api_client.py 中的图片路径和问题
python api_example.py
# 或
python api_client.py
```

#### 方法 B：在代码中直接调用

```python
import requests
import base64

# API 端点
API_URL = "http://localhost:5011/api/predict"

# 准备图片（转换为 base64）
with open("/workspace/brain.jpg", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode('utf-8')

# 构建请求
payload = {
    "data": [
        "Analyze Image",           # 任务类型
        "HealthGPT-M3",            # 模型名称
        "What problems are there with this brain CT?",  # 问题
        f"data:image/jpeg;base64,{image_base64}"  # base64编码的图片
    ]
}

# 发送请求
response = requests.post(API_URL, json=payload)
result = response.json()

# 提取答案
answer = result["data"][0]
print(answer)
```

### 3. 使用 curl 命令

```bash
# 将图片转换为 base64（需要先安装 base64 工具）
IMAGE_BASE64=$(base64 -w 0 /workspace/brain.jpg)

# 发送请求
curl -X POST http://localhost:5011/api/predict \
  -H "Content-Type: application/json" \
  -d "{
    \"data\": [
      \"Analyze Image\",
      \"HealthGPT-M3\",
      \"What problems are there with this brain CT?\",
      \"data:image/jpeg;base64,${IMAGE_BASE64}\"
    ]
  }"
```

### 4. API 参数说明

**请求格式：**
```json
{
  "data": [
    "任务类型",      // "Analyze Image" 或 "Generate Image"
    "模型名称",      // "HealthGPT-M3" 或 "HealthGPT-L14"
    "问题文本",      // 你的问题
    "图片base64"     // "data:image/jpeg;base64,<base64编码>"
  ]
}
```

**响应格式：**
```json
{
  "data": [
    "文本答案",      // Analyze Image 返回的文本
    null,            // 或生成的图片（Generate Image）
    ...
  ]
}
```

---

## 方式三：查看 API 文档

启动服务器后，访问 `http://localhost:5011/docs` 可以查看完整的 API 文档和交互式测试界面。

---

## 常见问题

### Q: 如何修改服务器端口？
A: 修改 `app.py` 第 81 行的 `server_port=5011` 为其他端口。

### Q: 如何修改服务器地址？
A: 修改 `app.py` 第 81 行的 `server_name="0.0.0.0"` 为 `"127.0.0.1"`（仅本地）或其他 IP。

### Q: API 调用失败怎么办？
A: 
1. 确保服务器已启动
2. 检查 API_URL 是否正确
3. 检查图片路径和 base64 编码是否正确
4. 查看服务器终端的错误信息

### Q: 如何批量处理多张图片？
A: 可以编写循环脚本，依次调用 API：

```python
import os
from api_client import call_healthgpt

image_dir = "/workspace/images"
questions = ["What problems are there with this brain CT?"]

for image_file in os.listdir(image_dir):
    if image_file.endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(image_dir, image_file)
        result = call_healthgpt(
            image_path=image_path,
            question=questions[0],
            task="Analyze Image",
            model="HealthGPT-M3"
        )
        print(f"{image_file}: {result['data'][0]}")
```

