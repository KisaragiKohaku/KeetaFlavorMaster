# 🦘 Keeta Flavor Master (English Introduction) 

Keeta Food Intelligent Recommendation Assistant based on Streamlit and Mistral-7B, integrated with RAG technology for precise nutritional Q&A.

Demo Video Link: https://drive.google.com/file/d/1d6x_jPrNlg_V51jZ13AnPqwoVh8xpfwH/view?usp=sharing

## Version Control Notes
1. The gamma version integrates DeepSeek API for auxiliary intent identification.  
2. The beta version uses only hard-coding for naive intent identification.  
3. The beta version is no longer maintained and is not recommended for use. The developer has kept this branch purely for archival purposes.  
4. The current beta and gamma versions are for learning purposes only.

## Download Model (GGUF format model needs to be downloaded in advance)
The test model is Mistral-7B-Instruct-v0.3-GGUF, Q4-K-S quantized version (3.85GB). Please download it from the URL below.  
You can also deploy other models yourself.  
https://huggingface.co/MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF

## Project Structure
```bash
KeetaFlavorMaster/
├── data/
│   └── menu.json          # Food information
├── models/                # GGUF format model
├── app.py                 # Streamlit main program
├── chatbot.py             # LLM response module
├── intent_parser.py       # Intent identification module (gamma version)
└── retriever.py           # RAG retrieval module
```

## 🌟 Core Features
- **Dish Retrieval**：RAG retrieval system based on ChromaDB + all-MiniLM-L6-v2
- **Nutrition Analysis**：Parse JSON data to calculate nutritional indicators of dishes
- **Intelligent Dialogue**：Generate professional responses using Mistral-7B-Q4 precision quantized model
- **Debugging Panel**：Real-time display of retrieval context and matching results
- **Intent Identification**：Use DeepSeek API to classify user input for precise answers

## 🚀 Quick Start
### Environment Requirements
- Python 3.11
- RAM ≥ 16GB
- GPU Used for Testing：RTX 4070 Laptop
- System Requirements：Windows

### Installation Steps
```bash
# 1. Clone the project
# For beta version:
git clone -b beta https://github.com/KisaragiKohaku/KeetaFlavorMaster.git

# For gamma version:
git clone -b gamma https://github.com/KisaragiKohaku/KeetaFlavorMaster.git

# Changing to the working directory
cd KeetaFlavorMaster

# 2. Create a virtual environment
python -m venv venv

# 3. Activate the virtual environment (Windows)
# Open cmd in the KeetaFlavorMaster directory and execute:
.venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Start the application
# Ensure you are in the virtual environment, i.e., the cmd command line has the prefix (.venv)
# Execute:
.venv\Scripts\streamlit run app.py
```
---
---
# 🦘 Keeta Flavor Master （中文说明） 

基于Streamlit与Mistral-7B的Keeta美食智能推荐助手，集成RAG技术实现精准营养问答

在线视频演示：https://drive.google.com/file/d/1d6x_jPrNlg_V51jZ13AnPqwoVh8xpfwH/view?usp=sharing

## 版本控制说明
1. gamma版接入DeepSeek API进行辅助意图识别  
2. beta版仅通过硬编码进行朴素意图识别
3. 请注意，beta版目前已停止更新，不建议使用，开发者仅基于纪念意义保留了该分支  
4. 当前beta版、gamma版仅作学习用途

## 下载模型 (需提前下载GGUF格式模型)
测试模型为Mistral-7B-Instruct-v0.3-GGUF，Q4-K-S量化版（3.85GB），请于下方网址自行下载，您也可以自行部署其他模型
https://huggingface.co/MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF

## 项目结构
```bash
KeetaFlavorMaster/
├── data/
│   └── menu.json          # 美食信息
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
# 如克隆beta版，使用以下命令:
git clone -b beta https://github.com/KisaragiKohaku/KeetaFlavorMaster.git

# 如克隆gamma版，使用以下命令:
git clone -b gamma https://github.com/KisaragiKohaku/KeetaFlavorMaster.git

# 进入工作目录
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
