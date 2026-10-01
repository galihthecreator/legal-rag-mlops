import streamlit as st
import requests

st.set_page_config(page_title="Indonesian Legal RAG", page_icon="⚖️", layout="wide")

st.title("⚖️ Indonesian Legal Assistant (RAG)")
st.caption("Powered by Qwen2.5-1.5B & BGE-M3 (RTX 2050 GPU Accelerated)")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Tanyakan sesuatu tentang dokumen hukum..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Mencari referensi hukum & merumuskan jawaban..."):
            try:
                response = requests.post(
                    "http://localhost:8000/api/v1/ask",
                    json={"query": prompt, 
                          "use_hyde": False},
                    timeout=300
                )
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "Tidak ada jawaban.")
                    retrieved_docs = data.get("retrieved_documents", [])

                    st.write(answer)

                    if retrieved_docs:
                        with st.expander("📚 Sumber Referensi Dokumen Hukum"):
                            for idx, doc in enumerate(retrieved_docs, 1):
                                source_file = doc.get("source", "Unknown").split("/")[-1]
                                page_num = doc.get("page", "-")
                                content = doc.get("content", "")
                                st.markdown(f"**[{idx}] {source_file} (Halaman {page_num})**")
                                st.caption(content[:300] + "...")
                                st.divider()

                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    st.error(f"Error dari API: {response.status_code}")
            except Exception as e:
                st.error(f"Gagal terhubung ke server FastAPI: {e}")