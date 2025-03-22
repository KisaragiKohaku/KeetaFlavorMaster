import streamlit as st
from retriever import NutritionRetriever
from chatbot import FoodChatBot
import os
import sys
import asyncio
import torch


# 在 Streamlit 初始化之前设置 Windows 事件循环策略
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# 环境变量设置
os.environ.update({
    "STREAMLIT_SERVER_FILE_WATCHER_TYPE": "poll",
    "TOKENIZERS_PARALLELISM": "false",
    "PROJECT_ROOT": os.path.abspath(os.path.dirname(__file__))
})


def init_session():
    # 检查初始化设置
    if 'initialized' not in st.session_state:
        st.session_state.messages = []
        try:
            with st.spinner("🚀 Initializing system..."):
                st.session_state.retriever = NutritionRetriever()
                st.session_state.chatbot = FoodChatBot()

                gpu_status = st.sidebar.empty()
                if torch.cuda.is_available():
                    gpu_status.success("✅ GPU acceleration enabled")
                else:
                    gpu_status.warning("⚠️ GPU is not available")

                if len(st.session_state.messages) > 10:
                    st.session_state.messages = st.session_state.messages[-10:]

                st.session_state.initialized = True

        except Exception as sif:
            st.error(f"System initialization failed: {str(sif)}")
            st.stop()


init_session()

# 界面布局
st.title("🦘 Keeta Flavor Master")
st.caption("Powered by Mistral-7B & RAG")

# 调试面板
with st.sidebar:
    with st.expander("⚙️ Debug Dashboard", expanded=False):
        debug_placeholder = st.empty()

# 聊天记录
for msg in st.session_state.messages:
    avatar = "🤖" if msg["role"] == "assistant" else None
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# 用户输入处理
if prompt := st.chat_input("Please enter..."):
    # 添加用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 生成回复
    with st.spinner("🔍 Thinking..."):
        try:
            # RAG检索
            context = st.session_state.retriever.search(prompt)

            # 显示调试信息
            newline = '\n'
            replacement = '  ' + newline
            debug_info = [
                "### Data Retrieval",
                *[f"**Documents {i + 1}**  \n{doc.replace(newline, replacement)}"
                  for i, doc in enumerate(context["documents"])],
                "### Matched Dishes",
                ", ".join([d['name'] for d in context["dishes"]])
            ]
            debug_placeholder.markdown("\n\n".join(debug_info))

            with st.chat_message("assistant", avatar="🤖"):
                response_placeholder = st.empty()  # 在消息块内创建占位符
                full_response = ""

            # 流式生成回复
            for chunk in st.session_state.chatbot.generate_response(prompt, context):
                full_response += chunk
                formatted_response = full_response.replace('\n', '  \n')
                response_placeholder.markdown(f"{formatted_response}▌")

            response_placeholder.markdown(formatted_response)

            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            error_msg = f"⚠️ System busy, please try again later. Error: {str(e)}"
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
