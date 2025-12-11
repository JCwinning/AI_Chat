# 🤖 AI 聊天应用

一个基于 Streamlit 构建的多语言、多模型 AI 聊天应用，支持流式响应、图片输入和并发模型查询。

## ✨ 功能特色

- **🌐 多语言支持**: 中英文双语界面
- **🤖 多模型聊天**: 同时查询多个 AI 模型
- **🎨 图像生成**: 使用先进的 AI 模型创建和编辑图像
- **📱 响应式设计**: 现代化双面板布局
- **🖼️ 文件支持**: 上传图片和 PDF 文件与 AI 聊天
- **💬 聊天历史**: 持久化对话存储
- **⚡ 流式响应**: 实时显示回复内容
- **🔧 自定义模型**: 添加您自己的 AI 模型配置
- **🌍 多服务商**: 支持 OpenAI、ModelScope、OpenRouter 等

## 🚀 快速开始

### 环境要求

- Python 3.7 或更高版本
- pip 包管理器

### 安装步骤

1. **克隆仓库**（如适用）或下载文件：
   
   ```bash
   # 如果使用 git
   git clone <repository-url>
   cd AI_Chat
   ```

# 否则，确保拥有所有项目文件

```
2. **安装依赖**：
```bash
pip install -r requirements.txt
```

3. **配置模型**：
   创建 `config.py` 文件，配置您的 AI 模型（详见配置章节）。

4. **运行应用**：
   
   ```bash
   streamlit run app.py
   ```

应用将在默认浏览器中打开，地址为 `http://localhost:8501`

## ⚙️ 配置说明

### 设置 `config.py`

在项目根目录创建 `config.py` 文件，配置您的模型信息。使用以下格式：

```python
import keyring

# OpenAI 配置
openai_gpt4 = {
    "api_key": "your-openai-api-key-here",
    "base_url": "https://api.openai.com/v1",
    "models": "gpt-4",
}

# ModelScope 配置
qwen_3_235b = {
    "api_key": keyring.get_password("system", "modelscope"),
    "base_url": "https://api-inference.modelscope.cn/v1",
    "models": "Qwen/Qwen3-235B-A22B-Instruct-2507",
}

# OpenRouter 配置
openrouter_mistral = {
    "api_key": "your-openrouter-api-key-here",
    "base_url": "https://openrouter.ai/api/v1",
    "models": "mistralai/mistral-7b-instruct",
}

# 自定义模型示例
custom_llm = {
    "api_key": "your-api-key",
    "base_url": "https://your-custom-endpoint.com/v1",
    "models": "your-model-name",
}

# 图像生成模型示例
gemini_image = {
    "api_key": "your-openrouter-key",
    "base_url": "https://openrouter.ai/api/v1",
    "models": "google/gemini-2.5-flash-image-preview",
    "type": "image"  # 指定图像生成模型的类型
}
```


## 📖 使用指南

### 基础聊天

1. **选择模型**: 从侧边栏选择一个或多个 AI 模型
2. **配置参数**: 根据需要调整温度和系统提示
3. **开始聊天**: 输入您的消息并按回车键
4. **查看回复**: 并排查看所有选定模型的流式回复

### 高级功能

#### 文件上传

- 点击侧边栏中的"📎 上传文件"
- 选择图片（PNG、JPG、JPEG 格式）或 PDF 文件
- 发送前会显示文件预览
- 图片包含在消息中进行多模态 AI 分析
- PDF 文件将自动转换为 Markdown 格式发送给 AI

#### 图像生成
- 选择一个图像生成模型（例如 Gemini Flash Image, Qwen Image）
- 描述您想要创建的图像
- **多模态生成**: 上传一张图片并要求模型"修改这张图"或"生成变体"

#### 聊天管理

- **新聊天**: 点击"➕ 新聊天"开始新对话
- **保存的聊天**: 从侧边栏访问之前的对话
- **删除聊天**: 使用 🗑️ 按钮删除不需要的对话
- **自动保存**: 聊天记录会自动保存并带时间戳

#### 自定义模型

1. 在模型标签页中点击"➕ 添加新模型"
2. 填写模型详细信息：
   - **模型名称**: 模型的显示名称
   - **API 密钥**: 您的 API 凭证
   - **基础 URL**: API 端点 URL
   - **模型列表**: 逗号分隔的模型名称
3. 点击"添加模型"保存并自动选择它

#### 语言切换

- 使用侧边栏中的语言按钮在中英文之间切换
- 所有界面元素将更新为您选择的语言

## 🏗️ 架构设计

### 项目结构

```
AI_Chat/
├── app.py                 # 主 Streamlit 应用
├── config.py              # 模型配置（用户创建）
├── requirements.txt       # Python 依赖
├── design.md             # 详细技术文档
├── chat_history.json     # 持久化聊天存储（自动生成）
├── CLAUDE.md             # AI 助手开发指南
└── README.md             # 英文说明文档
└── _README_CN.md         # 本中文说明文档
```

### 核心组件

- **多线程**: 使用 Python 多线程进行并行模型请求
- **队列通信**: 流式响应的线程安全消息传递
- **会话管理**: Streamlit 会话状态管理聊天历史和设置
- **文件持久化**: 基于 JSON 的聊天历史存储
- **文件处理**: 图片的 Base64 编码和 PDF 转 Markdown 转换
- **错误处理**: API 故障的优雅降级

## 🛠️ 开发

### 依赖项

应用使用以下主要依赖：

- `streamlit>=1.28.0` - Web 框架
- `openai>=1.0.0` - API 客户端库
- `anthropic>=0.7.0` - Anthropic API 支持
- `pillow>=10.0.0` - 图片处理
- `markitdown>=0.0.1a2` - PDF 转 Markdown 转换
- `python-dotenv>=1.0.0` - 环境变量管理
- `keyring` - 安全凭证存储

### 添加新功能

1. **新模型**: 在 `config.py` 中添加配置
2. **UI 更改**: 修改 `app.py` 时保持多语言结构
3. **新语言**: 扩展 `app.py` 中的 `translations` 字典
4. **存储格式**: 修改保存/加载函数以适应不同的持久化需求

## 🔒 安全性

- **API 密钥**: 使用 keyring 或环境变量安全存储
- **文件上传**: 支持图片（PNG、JPG、JPEG）和 PDF 文件上传
- **会话数据**: 聊天历史以 JSON 格式本地存储
- **无远程存储**: 所有数据保留在您的本地机器上

