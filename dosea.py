import os
import re
from pathlib import Path
from whoosh.index import create_in, open_dir, exists_in
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import QueryParser
from whoosh.analysis import Tokenizer, Token
import jieba
import streamlit as st


# ==============================
# 自定义中文分词器（兼容 Whoosh 新版）
# ==============================

class JiebaTokenizer(Tokenizer):
    def __call__(self, value, positions=False, chars=False, keeporiginal=False,
                 removestops=True, start_pos=0, start_char=0, mode='', **kwargs):
        if not isinstance(value, str):
            value = str(value)
        words = jieba.lcut(value)
        pos = start_pos
        char = start_char
        for word in words:
            if word.strip():
                word_len = len(word)
                yield Token(
                    text=word,
                    pos=pos,
                    startchar=char,
                    endchar=char + word_len,
                    **kwargs
                )
                pos += 1
                char += word_len


def jieba_analyzer():
    return JiebaTokenizer()


# ==============================
# 文档读取函数
# ==============================

def read_pdf(path):
    try:
        from PyPDF2 import PdfReader
        text = ""
        with open(path, "rb") as f:
            reader = PdfReader(f)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        return text
    except Exception as e:
        st.error(f"读取 PDF 失败 {path}: {e}")
        return ""


def read_docx(path):
    try:
        from docx import Document
        doc = Document(path)
        return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    except Exception as e:
        st.error(f"读取 DOCX 失败 {path}: {e}")
        return ""


def read_txt(path):
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        st.error(f"读取 TXT 失败 {path}: {e}")
        return ""


def get_text(file_path: Path):
    ext = file_path.suffix.lower()
    if ext == ".pdf":
        return read_pdf(file_path)
    elif ext == ".docx":
        return read_docx(file_path)
    elif ext == ".txt":
        return read_txt(file_path)
    else:
        return ""


def clean_text(text: str) -> str:
    """仅去除首尾空白，保留内部换行和段落"""
    return text.strip()


def get_mime_type(file_path: str) -> str:
    """根据扩展名返回正确的 MIME 类型"""
    ext = Path(file_path).suffix.lower()
    mime_map = {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".txt": "text/plain",
        ".doc": "application/msword",
    }
    return mime_map.get(ext, "application/octet-stream")


# ==============================
# 初始化索引
# ==============================

INDEX_DIR = "indexdir"
DOCS_DIR = "docs"

os.makedirs(DOCS_DIR, exist_ok=True)

schema = Schema(
    path=ID(stored=True, unique=True),
    content=TEXT(stored=True),  # 原始文本（用于高亮和片段提取）
    tokens=TEXT(stored=False, analyzer=jieba_analyzer())  # 分词后用于搜索
)

if not exists_in(INDEX_DIR):
    os.makedirs(INDEX_DIR, exist_ok=True)
    ix = create_in(INDEX_DIR, schema)
else:
    ix = open_dir(INDEX_DIR)


# ==============================
# 高亮上下文片段函数
# ==============================

def highlight_snippet(text: str, keyword: str, context_len: int = 60) -> str:
    if not keyword or not text:
        return "（无内容）"

    text_lower = text.lower()
    keyword_lower = keyword.lower()
    start = text_lower.find(keyword_lower)
    if start == -1:
        start = text.find(keyword)

    if start == -1:
        return f"（文档被检索到，但未在内容中找到“{keyword}”）"

    left = max(0, start - context_len)
    right = min(len(text), start + len(keyword) + context_len)
    snippet = text[left:right]

    highlighted = re.sub(
        re.escape(keyword),
        r"<mark style='background-color: yellow; color: red; font-weight: bold;'>\g<0></mark>",
        snippet,
        flags=re.IGNORECASE
    )

    prefix = "..." if left > 0 else ""
    suffix = "..." if right < len(text) else ""
    return prefix + highlighted + suffix


# ==============================
# Streamlit 主界面
# ==============================

st.set_page_config(page_title="智能文档检索系统", layout="wide")

# 🔍 搜索框放在最上面
st.title("📄 智能文档检索系统")
query_str = st.text_input("🔍 输入关键词或短语：", placeholder="支持中文短语搜索...")

# 侧边栏：上传 & 管理
with st.sidebar:
    st.header("📂 上传文档")
    uploaded_files = st.file_uploader(
        "支持 PDF / DOCX / TXT",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )

    if uploaded_files:
        for f in uploaded_files:
            save_path = Path(DOCS_DIR) / f.name
            if save_path.exists():
                st.warning(f"⚠️ '{f.name}' 已存在，跳过。")
                continue

            with open(save_path, "wb") as out:
                out.write(f.getbuffer())

            raw_text = get_text(save_path)
            clean_text_content = clean_text(raw_text)
            if not clean_text_content:
                st.warning(f"⚠️ 无法提取文本：{f.name}")
                save_path.unlink(missing_ok=True)
                continue

            try:
                with ix.writer() as writer:
                    tokens_text = " ".join(jieba.lcut(clean_text_content))
                    writer.add_document(
                        path=str(save_path),
                        content=clean_text_content,
                        tokens=tokens_text
                    )
                st.success(f"✅ 已索引：{f.name}")
            except Exception as e:
                st.error(f"索引失败 {f.name}: {e}")
                save_path.unlink(missing_ok=True)

    st.markdown("---")
    st.header("⚙️ 管理")
    if st.button("🗑️ 清空所有文档与索引"):
        import shutil

        shutil.rmtree(INDEX_DIR, ignore_errors=True)
        shutil.rmtree(DOCS_DIR, ignore_errors=True)
        st.success("已清空！请刷新页面。")
        st.experimental_rerun()

    st.markdown("### 📁 已索引文档")
    docs = sorted(Path(DOCS_DIR).glob("*"))
    if docs:
        for doc in docs:
            st.caption(f"- {doc.name}")
    else:
        st.caption("暂无文档")

# ==============================
# 执行搜索逻辑（结构完整，无 EOF 错误）
# ==============================

if query_str.strip():
    try:
        with ix.searcher() as searcher:
            # 先尝试在原始 content 字段中精确匹配
            try:
                query_content = QueryParser("content", ix.schema).parse(query_str)
                results = searcher.search(query_content, limit=20)
            except:
                results = []

            # 如果没找到，再用分词字段 tokens 搜索
            if not results:
                query_tokens = QueryParser("tokens", ix.schema).parse(query_str)
                results = searcher.search(query_tokens, limit=20)

            if results:
                st.subheader(f"共找到 {len(results)} 个结果：")
                for hit in results:
                    filename = Path(hit["path"]).name
                    snippet = highlight_snippet(hit["content"], query_str, context_len=200)
                    with st.expander(f"📄 {filename}"):
                        st.markdown(f"**命中片段**：<br>{snippet}", unsafe_allow_html=True)

                        # 下载按钮（支持所有格式）
                        file_path = hit["path"]
                        with open(file_path, "rb") as f:
                            file_data = f.read()
                        st.download_button(
                            label="📥 下载原文",
                            data=file_data,
                            file_name=filename,
                            mime=get_mime_type(file_path)
                        )
            else:
                st.info("❌ 未找到匹配的文档。")
    except Exception as e:
        st.error(f"搜索出错：{e}")
else:
    st.info("请输入关键词开始搜索")

# 文件末尾留一个空行（避免某些解析器警告）
