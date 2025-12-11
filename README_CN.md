# 🤖 AI 聊天应用

一个基于 Streamlit 构建的多语言、多模型 AI 聊天应用，支持流式响应、图片输入和并发模型查询。

## ✨ 功能特性

- **🌐 多语言支持**: 中文和英文界面
- **🤖 多模型聊天**: 同时查询多个 AI 模型并进行并排比较
- **🎨 图像生成**: 使用高级 AI 模型创建和编辑图像
- **📚 提示词库**: 专业系统提示词库，用于专门的 AI 交互
- **📱 响应式设计**: 现代化的双面板布局，侧边栏配置
- **🖼️ 文件支持**: 上传图片和 PDF 文件进行多模态 AI 分析
- **🔍 网络搜索**: 集成网络搜索功能，获取实时信息
- **💬 聊天历史**: 持久化对话存储，自动保存
- **⚡ 流式响应**: 来自多个模型的实时响应显示
- **🔧 自定义模型**: 动态添加自己的 AI 模型配置
- **🌍 多提供商**: 支持 OpenAI、ModelScope、OpenRouter 等

## 🚀 快速开始

### 系统要求

- Python 3.7 或更高版本
- pip 包管理器

### 安装步骤

1. **克隆仓库**（如适用）或下载文件：
```bash
# 如果使用 git
git clone <repository-url>
cd AI_Chat

# 否则，确保拥有所有项目文件
```

2. **安装依赖**：
```bash
pip install -r requirements.txt
```

3. **配置模型**：
创建一个 `config.py` 文件来配置你的 AI 模型（详见配置部分）。

4. **运行应用**：
```bash
streamlit run app.py
```

应用将在默认浏览器中打开，地址为 `http://localhost:8501`

## ⚙️ 配置说明

### 设置 `config.py`

在项目根目录创建 `config.py` 文件，配置你的模型：

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
    "type": "image"  # 为图像生成模型指定类型
}
```

### 使用 Keyring 安全存储（推荐）

为了更好的安全性，使用 Python 的 keyring 存储 API 密钥：

```bash
# 如果尚未安装，先安装 keyring
pip install keyring

# 安全存储 API 密钥
keyring set system modelscope
keyring set system openrouter
```

然后在 `config.py` 中引用：
```python
api_key = keyring.get_password("system", "modelscope")
```

## 📖 使用指南

### 基础聊天

1. **选择模型**: 从侧边栏选择一个或多个 AI 模型
2. **配置参数**: 根据需要调整温度和系统提示
3. **开始聊天**: 输入消息并按回车键
4. **查看响应**: 并排查看所有选定模型的流式响应

### 高级功能

#### 文件上传
- 在侧边栏点击"📎 上传文件"
- 选择图片（PNG、JPG、JPEG）或 PDF 文件
- 文件在发送前会显示预览
- 在消息中包含图片进行多模态 AI 分析
- PDF 文件会自动转换为 Markdown 并发送给 AI
- 支持 Word 文档（.docx）和 Excel 文件（.xlsx）

#### 图像生成
- 选择图像生成模型（如 Gemini Flash Image、Qwen Image）
- 描述你想要创建的图像
- **多模态生成**: 上传图片并要求模型"修改这个"或"生成变体"

#### 提示词库
1. 导航到"📚 提示词库"选项卡
2. 浏览预配置的系统提示词库
3. 使用搜索栏查找特定提示词
4. 点击"应用"将提示词用作系统提示
5. 提示词包括"Linux 终端"、"旅行指南"、"密码生成器"等角色

#### 网络搜索
1. 在聊天界面中点击开关启用网络搜索
2. 你的消息将使用实时网络信息处理
3. 搜索结果显示来源和日期
4. AI 模型将在响应中包含最新信息

#### 聊天管理
- **新建聊天**: 点击"➕ 新建聊天"开始新的对话
- **保存的聊天**: 从侧边栏访问以前的对话
- **删除聊天**: 使用 🗑️ 按钮删除不需要的对话
- **自动保存**: 聊天会自动保存时间戳

#### 自定义模型
1. 在模型选项卡中点击"➕ 添加新模型"
2. 填写模型详情：
   - **模型名称**: 模型的显示名称
   - **API 密钥**: 你的 API 凭证
   - **基础 URL**: API 端点 URL
   - **模型**: 逗号分隔的模型名称
3. 点击"添加模型"保存并自动选择它

#### 语言切换
- 使用侧边栏中的语言按钮在中英文之间切换
- 所有界面元素将更新为你首选的语言

## 🏗️ 架构设计

### 项目结构

```
AI_Chat/
├── app.py                 # 主 Streamlit 应用
├── config.py              # 模型配置（用户创建）
├── requirements.txt       # Python 依赖
├── prompt_bay.csv        # 系统提示词库
├── design.md             # 详细技术文档
├── chat_history.json     # 持久化聊天存储（自动生成）
├── CLAUDE.md             # AI 助手开发指南
├── README.md             # 英文版本文档
└── README_CN.md          # 中文版本文档
```

### 核心组件

- **多线程**: 使用 Python 线程进行并行模型请求
- **队列通信**: 用于流式响应的线程安全消息传递
- **会话管理**: Streamlit 会话状态用于聊天历史和设置
- **文件持久化**: 基于 JSON 的聊天历史存储
- **文件处理**: 图片的 Base64 编码和 PDF 的 Markdown 转换
- **错误处理**: API 失败的优雅降级

## 🛠️ 开发

### 依赖项

应用使用以下主要依赖：
- `streamlit>=1.28.0` - Web 框架
- `pandas>=2.0.0` - CSV 文件数据处理（提示词库）
- `openai>=1.0.0` - API 客户端库
- `anthropic>=0.7.0` - Anthropic API 支持
- `pillow>=10.0.0` - 图像处理
- `markitdown>=0.0.1a2` - PDF 到 Markdown 转换
- `python-dotenv>=1.0.0` - 环境变量管理
- `keyring` - 安全凭证存储
- `dashscope>=1.14.0` - 阿里云 AI 模型支持
- `openpyxl>=3.0.0` - Excel 文件处理
- `python-docx>=1.2.0` - Word 文档处理
- `tavily-python>=0.3.0` - 网络搜索集成

### 添加新功能

1. **新模型**: 在 `config.py` 中添加配置
2. **UI 更改**: 修改 `app.py` 同时保持多语言结构
3. **新语言**: 在 `app.py` 中扩展 `translations` 字典
4. **存储格式**: 修改保存/加载函数以适应不同的持久化需求

## 🔒 安全性

- **API 密钥**: 使用 keyring 或环境变量安全存储
- **文件上传**: 支持图片（PNG、JPG、JPEG）和 PDF 文件
- **会话数据**: 聊天历史以 JSON 格式本地存储
- **无远程存储**: 所有数据保留在你的本地机器上

## 🌍 支持的提供商

应用支持任何 OpenAI 兼容的 API 提供商：

- **OpenAI**: GPT 模型（GPT-3.5、GPT-4 等）
- **ModelScope**: Qwen 和其他中文模型
- **OpenRouter**: 访问各种开源模型
- **自定义端点**: 任何 OpenAI 兼容的 API

## 🤝 贡献

1. Fork 仓库
2. 创建功能分支
3. 进行更改
4. 充分测试
5. 提交拉取请求

## 📝 许可证

本项目是开源的。请参阅许可证文件了解详情。

## 🆘 故障排除

### 常见问题

**问：模型没有在下拉列表中显示**
- 确保 `config.py` 已正确配置
- 检查 API 密钥是否有效且可访问
- 验证基础 URL 是否正确

**问：文件无法上传**
- 检查文件格式（支持 PNG、JPG、JPEG、PDF）
- 确保文件未损坏
- 验证文件大小限制
- 对于 PDF 文件，确保内容可读

**问：聊天历史未保存**
- 检查应用目录的写入权限
- 确保 `chat_history.json` 未损坏
- 验证磁盘空间可用性

**问：流式响应很慢**
- 检查网络连接
- 验证 API 端点是否可访问
- 考虑减少并发模型数量

### 获取帮助

- 查看 `design.md` 了解详细技术文档
- 查看 `CLAUDE.md` 了解开发指南
- 确保所有依赖都已正确安装

## 🎯 路线图

未来的增强功能可能包括：
- [ ] 语音输入/输出支持
- [ ] 更多语言选项
- [ ] 高级对话分析
- [ ] 自定义功能的插件系统
- [ ] 云同步选项

---

**使用 Streamlit 和 ❤️ 构建**