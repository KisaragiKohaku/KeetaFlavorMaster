# 🦘 Keeta Flavor Master 

基于Streamlit与Mistral-7B的Keeta美食智能推荐助手，集成RAG技术实现精准营养问答

## 版本控制说明
1. gamma版接入DeepSeek API进行辅助意图识别  
2. beta版通过硬编码替代外部模型接入  
3. 当前beta版、gamma版仅作学习用途

## 下载模型 (需提前下载GGUF格式模型)
测试模型为Mistral-7B-Instruct-v0.3-GGUF，Q4-K-S量化版（3.85GB），请于下方网址自行下载，您也可以自行部署其他模型
https://huggingface.co/MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF

## 项目结构
```bash
KeetaFlavorMaster/
├── data/
│   ├── menu.json          # 美食信息
│   └── nutrition.json     # 营养数据
├── models/                # GGUF格式模型
├── app.py                 # Streamlit主程序
├── chatbot.py             # LLM回复模块
├── intent_parser.py       # 意图识别模块（gamma版）
└── retriever.py           # RAG检索模块
```

## 🌟 核心功能
- **菜品检索**：基于ChromaDB + all-MiniLM-L6-v2的RAG检索系统
- **营养分析**：解析JSON数据计算菜品营养指标
- **智能对话**：Mistral-7B-Q4精度量化模型生成专业回复
- **调试面板**：实时展示检索上下文与匹配结果
- **意图识别**：使用DeepSeek API对用户输入进行需求分类从而精确回答

## 🚀 快速上手
### 环境要求
- Python 3.11
- 运行内存 ≥ 16GB
- 测试所用显卡：RTX 4070 Laptop
- 系统要求：Windows

### 安装步骤
```bash
# 1. 克隆项目
git clone https://github.com/KisaragiKohaku/KeetaFlavorMaster.git
cd KeetaFlavorMaster

# 2. 创建虚拟环境
python -m venv venv

# 3. 激活虚拟环境 (Windows)
# 在KeetaFlavorMaster目录下打开cmd，键入执行:
.venv\Scripts\activate

# 4. 安装依赖
pip install -r requirements.txt

# 5. 启动应用
# 在KeetaFlavorMaster目录下打开cmd并激活虚拟环境
# 请注意，务必确保你在虚拟环境中，即cmd命令行有前缀(.venv)标识
# 键入执行：
.venv\Scripts\streamlit run app.py
```
