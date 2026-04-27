"""
양돈 사양관리 다국어 AI 챗봇 — Streamlit 웹앱
Multilingual Pig Farming Q&A Chatbot for Foreign Workers

Authors: 임채빈 (RDA-NIAS Swine Division)
"""

import os
import re
import streamlit as st
from pathlib import Path

# ===== 페이지 설정 =====
st.set_page_config(
    page_title="양돈 사양관리 다국어 AI 챗봇",
    page_icon="🐷",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🐷 양돈 사양관리 다국어 AI 챗봇")
st.caption("Multilingual Pig Farming Q&A Chatbot for Foreign Workers in Korean Swine Farms")
st.markdown("---")

# ===== 세션 상태 =====
if "messages" not in st.session_state:
    st.session_state.messages = []
if "system_ready" not in st.session_state:
    st.session_state.system_ready = False

# ===== 사이드바 =====
with st.sidebar:
    st.header("⚙️ 시스템 정보")
    st.markdown("""
    **지원 언어**:
    - 🇰🇷 한국어 (Korean)
    - 🇺🇸 English
    - 🇻🇳 Tiếng Việt (Vietnamese)
    - 🇹🇭 ภาษาไทย (Thai)
    - 🇰🇭 ភាសាខ្មែរ (Khmer)
    - 🇳🇵 नेपाली (Nepali)
    """)
    st.markdown("---")
    st.markdown("""
    **사용 방법**:
    1. 모국어 또는 원하는 언어로 질문 입력
    2. AI가 매뉴얼 근거로 답변 생성
    3. 답변 하단에 출처 매뉴얼 표시
    """)
    st.markdown("---")
    st.markdown("""
    **데이터 출처**:
    농촌진흥청 농업과학도서관
    『외국인 근로자를 위한 양돈 사양관리 매뉴얼』
    """)
    st.markdown("---")
    st.caption("개발: 임채빈 (RDA-NIAS Swine Division)")
    st.caption("AI 아이디어톤 출품작 — 2026")


# ===== 임베딩 모델 로드 (가벼운 모델 사용) =====
@st.cache_resource(show_spinner="🔧 임베딩 모델 로딩 중... (최초 1회 약 1~2분)")
def load_embedding_model():
    from sentence_transformers import SentenceTransformer
    # paraphrase-multilingual-MiniLM-L12-v2: 약 470MB, 검증된 안정 모델
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')


# ===== 지식 베이스 구축 (단계별 진행 상황 표시) =====
@st.cache_resource(show_spinner=False)
def build_knowledge_base():
    """PDF 추출 → 청킹 → 임베딩 → ChromaDB 구축"""
    import pdfplumber
    import chromadb
    
    APPENDIX_START = 41
    SKIP_PATTERNS = [
        r"발\s*간\s*사", r"머\s*리\s*말", r"서\s*문",
        r"contents", r"목\s*차", r"국립축산과학원",
        r"외국인 근로자가 증가", r"고령화와 부족한 고용",
    ]
    
    LANGUAGE_MAP = {
        "pig_manual_english.PDF": ("en", "English", "Pig Raising Manual (English)"),
        "pig_manual_vietnamese.PDF": ("vi", "Tiếng Việt", "Hướng dẫn chăn nuôi lợn (Vietnamese)"),
        "pig_manual_thai.PDF": ("th", "ภาษาไทย", "คู่มือการเลี้ยงสุกร (Thai)"),
        "pig_manual_khmer.PDF": ("km", "ភាសាខ្មែរ", "មគ្គុទ្ទេសក៍ចិញ្ចឹមជ្រូក (Khmer)"),
        "pig_manual_nepali.PDF": ("ne", "नेपाली", "बंगुर पालन निर्देशिका (Nepali)"),
    }
    
    def is_boilerplate(chunk):
        matches = sum(1 for p in SKIP_PATTERNS if re.search(p, chunk, re.IGNORECASE))
        return matches >= 2
    
    def chunk_filtered(text, size=1200, overlap=200):
        chunks = []
        start = 0
        while start < len(text):
            chunk = text[start:start+size].strip()
            if len(chunk) > 100 and not is_boilerplate(chunk):
                chunks.append(chunk)
            start += size - overlap
        return chunks
    
    def chunk_simple(text, size=1200, overlap=200):
        chunks = []
        start = 0
        while start < len(text):
            chunk = text[start:start+size].strip()
            if len(chunk) > 100:
                chunks.append(chunk)
            start += size - overlap
        return chunks
    
    def extract_with_sections(pdf_path, appendix_start=APPENDIX_START):
        body, appendix = "", ""
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                t = page.extract_text()
                if not t:
                    continue
                tagged = f"\n[페이지 {page_num}]\n{t}\n"
                if page_num < appendix_start:
                    body += tagged
                else:
                    appendix += tagged
        return body, appendix
    
    # 단계별 상태 표시
    progress_placeholder = st.empty()
    
    # === 1단계: PDF 처리 ===
    progress_placeholder.info("📄 [1/4] PDF 매뉴얼 처리 중...")
    
    manual_dir = Path("manuals")
    if not manual_dir.exists():
        st.error(f"⚠️ manuals 폴더를 찾을 수 없습니다: {manual_dir.absolute()}")
        st.stop()
    
    all_chunks = []
    all_metadatas = []
    found_count = 0
    
    for pdf_filename, (lang_code, lang_name, display_name) in LANGUAGE_MAP.items():
        pdf_path = manual_dir / pdf_filename
        if not pdf_path.exists():
            st.warning(f"   ⚠️ 파일 없음: {pdf_filename}")
            continue
        
        found_count += 1
        body, appendix = extract_with_sections(str(pdf_path))
        body_chunks = chunk_filtered(body)
        appendix_chunks = chunk_simple(appendix)
        
        for i, c in enumerate(body_chunks):
            all_chunks.append(c)
            all_metadatas.append({
                "source": display_name,
                "filename": pdf_filename,
                "language": lang_code,
                "lang_name": lang_name,
                "section": "body",
                "chunk_id": f"body_{i}"
            })
        for i, c in enumerate(appendix_chunks):
            all_chunks.append(c)
            all_metadatas.append({
                "source": display_name,
                "filename": pdf_filename,
                "language": lang_code,
                "lang_name": lang_name,
                "section": "appendix",
                "chunk_id": f"appendix_{i}"
            })
    
    if found_count == 0:
        st.error("⚠️ manuals 폴더에서 PDF 파일을 하나도 찾을 수 없습니다.")
        st.error(f"폴더 내용: {list(manual_dir.iterdir()) if manual_dir.exists() else '폴더 없음'}")
        st.stop()
    
    progress_placeholder.info(f"📄 [1/4] PDF 처리 완료: {found_count}개 매뉴얼, {len(all_chunks)}개 청크")
    
    # === 2단계: 임베딩 모델 로드 ===
    progress_placeholder.info("🧠 [2/4] 임베딩 모델 로드 중... (1~2분)")
    embed_model = load_embedding_model()
    
    # === 3단계: 임베딩 생성 ===
    progress_placeholder.info(f"🔢 [3/4] {len(all_chunks)}개 청크 임베딩 변환 중... (1~3분)")
    embeddings = embed_model.encode(all_chunks, show_progress_bar=False, batch_size=16)
    
    # === 4단계: ChromaDB 구축 ===
    progress_placeholder.info("💾 [4/4] 데이터베이스 구축 중...")
    
    chroma_client = chromadb.Client()
    try:
        chroma_client.delete_collection(name="pig_manuals")
    except:
        pass
    
    collection = chroma_client.create_collection(name="pig_manuals")
    collection.add(
        embeddings=embeddings.tolist(),
        documents=all_chunks,
        metadatas=all_metadatas,
        ids=[f"c_{i}" for i in range(len(all_chunks))]
    )
    
    progress_placeholder.empty()
    
    return collection, embed_model, len(all_chunks)


# ===== 검색 함수 =====
def search_chunks_tiered(question, collection, embed_model, top_k=5):
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 42
    
    try:
        q_lang = detect(question)
    except:
        q_lang = "unknown"
    
    q_emb = embed_model.encode([question])[0].tolist()
    seen, merged_docs, merged_metas = set(), [], []
    
    def add(docs, metas, label):
        for doc, meta in zip(docs, metas):
            key = (meta["source"], meta["chunk_id"])
            if key not in seen and len(merged_docs) < top_k:
                seen.add(key)
                merged_docs.append(doc)
                merged_metas.append({**meta, "tier": label})
    
    # 한국어 질문: 모든 본문 검색
    if q_lang == "ko":
        try:
            t1 = collection.query(
                query_embeddings=[q_emb],
                n_results=int(top_k * 0.8),
                where={"section": "body"}
            )
            add(t1["documents"][0], t1["metadatas"][0], "본문(전체)")
        except:
            pass
        if len(merged_docs) < top_k:
            try:
                t2 = collection.query(
                    query_embeddings=[q_emb],
                    n_results=top_k,
                    where={"section": "appendix"}
                )
                add(t2["documents"][0], t2["metadatas"][0], "부록(전체)")
            except:
                pass
        return {
            "documents": [merged_docs[:top_k]],
            "metadatas": [merged_metas[:top_k]],
            "detected_language": q_lang
        }
    
    # 외국어: 같은 언어 본문 우선
    if q_lang != "unknown":
        try:
            t1 = collection.query(
                query_embeddings=[q_emb],
                n_results=max(2, int(top_k * 0.6)),
                where={"$and": [{"language": q_lang}, {"section": "body"}]}
            )
            add(t1["documents"][0], t1["metadatas"][0], "본문(모국어)")
        except:
            pass
    
    if q_lang != "unknown" and len(merged_docs) < top_k:
        try:
            t2 = collection.query(
                query_embeddings=[q_emb],
                n_results=max(1, int(top_k * 0.2)),
                where={"$and": [{"language": q_lang}, {"section": "appendix"}]}
            )
            add(t2["documents"][0], t2["metadatas"][0], "부록(모국어)")
        except:
            pass
    
    if len(merged_docs) < top_k:
        t3 = collection.query(query_embeddings=[q_emb], n_results=top_k * 2)
        add(t3["documents"][0], t3["metadatas"][0], "다국어보충")
    
    return {
        "documents": [merged_docs[:top_k]],
        "metadatas": [merged_metas[:top_k]],
        "detected_language": q_lang
    }


# ===== 챗봇 함수 =====
LANG_NAMES_FULL = {
    "ko": "Korean (한국어)",
    "en": "English",
    "vi": "Vietnamese (Tiếng Việt)",
    "km": "Khmer (ភាសាខ្មែរ)",
    "th": "Thai (ภาษาไทย)",
    "ne": "Nepali (नेपाली)",
}


def ask_chatbot(question, collection, embed_model, gemini_client, top_k=5, max_retries=3):
    import time
    
    search_results = search_chunks_tiered(question, collection, embed_model, top_k=top_k)
    q_lang = search_results["detected_language"]
    q_lang_full = LANG_NAMES_FULL.get(q_lang, q_lang)
    
    context = ""
    for i, (chunk, meta) in enumerate(zip(
        search_results["documents"][0],
        search_results["metadatas"][0]
    ), 1):
        section_label = "MAIN BODY" if meta["section"] == "body" else "GLOSSARY"
        context += f"\n[Excerpt {i} | Source: {meta['source']} | Section: {section_label}]\n{chunk}\n"
    
    prompt = f"""You are a professional pig farming management assistant designed to help foreign workers in Korean swine farms.

# CRITICAL RULES
1. **Language**: Respond ONLY in {q_lang_full}. Do not mix languages.
2. **Source-grounded**: Answer ONLY using the manual excerpts below. Never invent facts.
3. **Honesty**: If excerpts don't contain enough info, say so in {q_lang_full}: "The manual does not contain specific information. Please consult your farm manager or veterinarian."
4. **Numerical precision**: Include exact numbers (temperature, days, dosage) from manuals.
5. **Practical tone**: Clear, concise, actionable.
6. **Safety**: For health/disease/vaccine, emphasize professional consultation when needed.

# RESPONSE FORMAT
- Direct answer (2-3 sentences)
- Step-by-step details if applicable
- End with: "📚 Source: [manual filenames used]"

# MANUAL EXCERPTS
{context}

# USER QUESTION (in {q_lang_full})
{question}

# YOUR ANSWER (in {q_lang_full} ONLY):
"""
    
    last_error = None
    for attempt in range(max_retries):
        try:
            response = gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            return {
                "answer": response.text,
                "language": q_lang_full,
                "sources": list(set(m["source"] for m in search_results["metadatas"][0])),
            }
        except Exception as e:
            last_error = e
            error_str = str(e)
            if "503" in error_str or "429" in error_str or "UNAVAILABLE" in error_str:
                wait_time = (2 ** attempt) * 3
                time.sleep(wait_time)
                continue
            else:
                raise e
    
    return {
        "answer": f"⚠️ AI 서버가 일시적으로 응답하지 못합니다. 잠시 후 다시 시도해주세요.\n(Error: {last_error})",
        "language": q_lang_full,
        "sources": []
    }


# ===== 메인 로직 =====

# Gemini API 키
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except (KeyError, FileNotFoundError):
    st.error("⚠️ Gemini API 키가 설정되지 않았습니다. Streamlit Secrets에 GEMINI_API_KEY를 등록해주세요.")
    st.stop()

@st.cache_resource
def init_gemini_client(key):
    from google import genai
    return genai.Client(api_key=key)

gemini_client = init_gemini_client(api_key)

# 지식 베이스 구축
collection, embed_model, num_chunks = build_knowledge_base()

if not st.session_state.system_ready:
    st.success(f"✅ 시스템 준비 완료: {num_chunks}개 매뉴얼 청크 인덱싱됨")
    st.session_state.system_ready = True

# ===== 채팅 UI =====
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 참조 매뉴얼 / Sources"):
                for src in msg["sources"]:
                    st.markdown(f"- {src}")

prompt = st.chat_input("질문을 입력하세요 / Type your question in any supported language...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("🔍 매뉴얼 검색 및 답변 생성 중..."):
            result = ask_chatbot(prompt, collection, embed_model, gemini_client)
        
        st.markdown(result["answer"])
        
        if result.get("sources"):
            with st.expander("📚 참조 매뉴얼 / Sources"):
                for src in result["sources"]:
                    st.markdown(f"- {src}")
        
        st.caption(f"🌐 감지된 언어 / Detected: {result['language']}")
    
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result.get("sources", [])
    })
