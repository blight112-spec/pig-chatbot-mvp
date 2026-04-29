"""
양돈 사양관리 다국어 AI 챗봇 — Streamlit 웹앱 (HF Spaces 시연 버전 v4)
2016 농촌진흥청·농협중앙회 발간 외국인근로자용 양돈 매뉴얼 기반
- 언어별 청크 분리 인덱싱
- 본문(10~94p) / 부록(95p~) 자동 분리
- 페이지 필터링 (표지·목차·판권지 자동 제거)
- ChromaDB 텔레메트리 비활성화 + 영구 저장

Authors: 임채빈 (RDA-NIAS Swine Division)
RDA AI Ideathon 2026 Submission
"""

import os

# ChromaDB 텔레메트리 비활성화 (posthog 호환성 이슈 회피)
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY_ENABLED"] = "False"

import re
import tempfile
import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="DonTalk — 양돈 사양관리 다국어 AI 챗봇",
    page_icon="🐷",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🐷 DonTalk")
st.caption("양돈 사양관리 다국어 AI 챗봇 | Multilingual Pig Farming Q&A Chatbot for Foreign Workers")
st.markdown("---")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "system_ready" not in st.session_state:
    st.session_state.system_ready = False

with st.sidebar:
    st.header("⚙️ 시스템 정보")
    st.markdown("""
    **지원 언어 (6개)**:
    - 🇰🇷 한국어 (Korean)
    - 🇺🇸 English
    - 🇻🇳 Tiếng Việt (Vietnamese)
    - 🇹🇭 ภาษาไทย (Thai)
    - 🇰🇭 ភាសាខ្មែរ (Khmer)
    - 🇳🇵 नेपाली (Nepali)
    """)
    st.markdown("---")

    st.markdown("""
    **🎯 핵심 기술**
    - 다국어 RAG (Retrieval-Augmented Generation)
    - 언어별 청크 분리 인덱싱
    - 한국어 병기 자동 추출
    - 본문 / 부록 자동 분리
    - 출처 매뉴얼 자동 표시
    - Gemini 2.5 Flash-Lite 기반 답변 생성
    """)
    st.markdown("---")

    st.markdown("""
    **📖 사용 방법**
    1. 모국어 또는 원하는 언어로 질문 입력
    2. AI가 매뉴얼 근거로 답변 생성
    3. 답변 하단에 출처 매뉴얼 표시
    """)
    st.markdown("---")

    debug_mode = st.checkbox("🔍 검색 결과 상세 보기", value=False,
                             help="AI가 어떤 매뉴얼 청크를 참고했는지 확인할 수 있습니다.")

    st.markdown("---")
    st.markdown("""
    **📚 데이터 출처**
    대한한돈협회
    『외국인근로자용 양돈장관리매뉴얼』 (2016)
    """)
    st.markdown("---")
    st.caption("개발: 임채빈 (RDA-NIAS Swine Division)")
    st.caption("DonTalk · RDA AI 아이디어톤 2026 출품작")


@st.cache_resource(show_spinner="🔧 임베딩 모델 로딩 중... (최초 1회 약 1~2분)")
def load_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')


def split_by_language(text, foreign_lang):
    """텍스트를 한국어와 외국어로 분리 (한글 비율 30% 임계값)"""
    korean_lines = []
    foreign_lines = []

    for line in text.split('\n'):
        line_stripped = line.strip()
        if not line_stripped:
            continue

        if line_stripped.startswith('[페이지'):
            korean_lines.append(line)
            foreign_lines.append(line)
            continue

        korean_chars = len(re.findall(r'[가-힣]', line_stripped))
        total_chars = len(re.sub(r'\s', '', line_stripped))

        if total_chars == 0:
            continue

        korean_ratio = korean_chars / total_chars

        if korean_ratio >= 0.3:
            korean_lines.append(line)
        else:
            foreign_lines.append(line)

    return '\n'.join(korean_lines), '\n'.join(foreign_lines)


@st.cache_resource(show_spinner=False)
def build_knowledge_base():
    """PDF 추출 → 페이지 필터링 → 본문/부록 분리 → 언어별 분리 → 청킹 → 임베딩 → ChromaDB"""
    import pdfplumber
    import chromadb

    # ===== 2016 매뉴얼 구조 기반 페이지 영역 정의 =====
    # 1~9p: 표지·발간사·목차 (제외)
    # 10~94p: 본문 (8개 챕터, 각 챕터 끝에 미니 용어집 포함)
    # 95~104p: 부록 (표준근로계약서, 방역관리지침)
    # 105~106p: 판권지·MEMO (제외)
    BODY_START = 10
    APPENDIX_START = 95
    APPENDIX_END = 104  # 이후는 판권지

    # 표지·목차·판권지·방역지침 헤더 등 의례적 텍스트 패턴
    SKIP_PATTERNS = [
        r"발\s*간\s*사", r"머\s*리\s*말", r"서\s*문",
        r"contents", r"목\s*차", r"국립축산과학원",
        r"농협중앙회", r"축산경영부",
        r"외국인 근로자가 증가", r"고령화와 부족한 고용",
        r"^\s*MEMO\s*$",
        r"인\s*쇄\s*\d{4}년", r"발\s*행\s*인", r"발\s*행\s*처",
    ]

    LANGUAGE_MAP = {
        "pig_manual_english.pdf": ("en", "English",
            "Hog Farm Management Manual (English, 2016)"),
        "pig_manual_vietnamese.pdf": ("vi", "Tiếng Việt",
            "Cẩm nang quản lý trang trại lợn (Vietnamese, 2016)"),
        "pig_manual_thai.pdf": ("th", "ภาษาไทย",
            "คู่มือการจัดการฟาร์มสุกร (Thai, 2016)"),
        "pig_manual_khmer.pdf": ("km", "ភាសាខ្មែរ",
            "មគ្គុទ្ទេសក៍គ្រប់គ្រងកសិដ្ឋានជ្រូក (Khmer, 2016)"),
        "pig_manual_nepali.pdf": ("ne", "नेपाली",
            "सुँगुर फार्म व्यवस्थापन निर्देशिका (Nepali, 2016)"),
    }

    def is_boilerplate(chunk):
        """의례적 텍스트(발간사·목차·판권지 등) 자동 필터링"""
        matches = sum(1 for p in SKIP_PATTERNS
                      if re.search(p, chunk, re.IGNORECASE))
        return matches >= 2

    def chunk_filtered(text, size=1000, overlap=150):
        """본문용 청킹 (의례적 텍스트 필터링 적용)"""
        chunks = []
        start = 0
        while start < len(text):
            chunk = text[start:start + size].strip()
            if len(chunk) > 80 and not is_boilerplate(chunk):
                chunks.append(chunk)
            start += size - overlap
        return chunks

    def chunk_simple(text, size=1000, overlap=150):
        """부록용 단순 청킹"""
        chunks = []
        start = 0
        while start < len(text):
            chunk = text[start:start + size].strip()
            if len(chunk) > 80:
                chunks.append(chunk)
            start += size - overlap
        return chunks

    def extract_with_sections(pdf_path,
                              body_start=BODY_START,
                              appendix_start=APPENDIX_START,
                              appendix_end=APPENDIX_END):
        """페이지 범위별로 본문/부록 분리 추출"""
        body, appendix = "", ""
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                t = page.extract_text()
                if not t:
                    continue
                tagged = f"\n[페이지 {page_num}]\n{t}\n"

                # 페이지 범위에 따라 영역 분류
                if body_start <= page_num < appendix_start:
                    body += tagged
                elif appendix_start <= page_num <= appendix_end:
                    appendix += tagged
                # 그 외 (1~9p 표지/목차, 105p~ 판권지)는 제외
        return body, appendix

    progress_placeholder = st.empty()
    progress_placeholder.info("📄 [1/4] PDF 매뉴얼 처리 및 언어별 분리 중...")

    manual_dir = Path(__file__).parent.parent / "manuals"
    if not manual_dir.exists():
        st.error(f"⚠️ manuals 폴더를 찾을 수 없습니다: {manual_dir.absolute()}")
        st.stop()

    all_chunks = []
    all_metadatas = []
    found_count = 0
    lang_counts = {"ko": 0, "en": 0, "vi": 0, "th": 0, "km": 0, "ne": 0}

    for pdf_filename, (foreign_lang, lang_name, display_name) in LANGUAGE_MAP.items():
        pdf_path = manual_dir / pdf_filename
        if not pdf_path.exists():
            st.warning(f"   ⚠️ 파일 없음: {pdf_filename}")
            continue

        found_count += 1
        body, appendix = extract_with_sections(str(pdf_path))

        # 본문 - 한국어/외국어 분리
        body_ko, body_foreign = split_by_language(body, foreign_lang)

        body_ko_chunks = chunk_filtered(body_ko)
        for i, c in enumerate(body_ko_chunks):
            all_chunks.append(c)
            all_metadatas.append({
                "source": display_name,
                "filename": pdf_filename,
                "language": "ko",
                "lang_name": "한국어",
                "section": "body",
                "chunk_id": f"{pdf_filename}_body_ko_{i}"
            })
            lang_counts["ko"] += 1

        body_foreign_chunks = chunk_filtered(body_foreign)
        for i, c in enumerate(body_foreign_chunks):
            all_chunks.append(c)
            all_metadatas.append({
                "source": display_name,
                "filename": pdf_filename,
                "language": foreign_lang,
                "lang_name": lang_name,
                "section": "body",
                "chunk_id": f"{pdf_filename}_body_{foreign_lang}_{i}"
            })
            lang_counts[foreign_lang] += 1

        # 부록 - 한국어/외국어 분리
        appendix_ko, appendix_foreign = split_by_language(appendix, foreign_lang)

        appendix_ko_chunks = chunk_simple(appendix_ko)
        for i, c in enumerate(appendix_ko_chunks):
            all_chunks.append(c)
            all_metadatas.append({
                "source": display_name,
                "filename": pdf_filename,
                "language": "ko",
                "lang_name": "한국어",
                "section": "appendix",
                "chunk_id": f"{pdf_filename}_appendix_ko_{i}"
            })
            lang_counts["ko"] += 1

        appendix_foreign_chunks = chunk_simple(appendix_foreign)
        for i, c in enumerate(appendix_foreign_chunks):
            all_chunks.append(c)
            all_metadatas.append({
                "source": display_name,
                "filename": pdf_filename,
                "language": foreign_lang,
                "lang_name": lang_name,
                "section": "appendix",
                "chunk_id": f"{pdf_filename}_appendix_{foreign_lang}_{i}"
            })
            lang_counts[foreign_lang] += 1

    if found_count == 0:
        st.error("⚠️ manuals 폴더에서 PDF 파일을 하나도 찾을 수 없습니다.")
        st.stop()

    lang_summary = " | ".join([f"{k}:{v}" for k, v in lang_counts.items() if v > 0])
    progress_placeholder.info(
        f"📄 [1/4] PDF 처리 완료: {found_count}개 매뉴얼, "
        f"총 {len(all_chunks)}개 청크 ({lang_summary})"
    )

    progress_placeholder.info("🧠 [2/4] 임베딩 모델 로드 중...")
    embed_model = load_embedding_model()

    progress_placeholder.info(
        f"🔢 [3/4] {len(all_chunks)}개 청크 임베딩 변환 중..."
    )
    embeddings = embed_model.encode(
        all_chunks, show_progress_bar=False, batch_size=16
    )

    progress_placeholder.info("💾 [4/4] 벡터 데이터베이스 구축 중...")

    chroma_persist_dir = tempfile.mkdtemp(prefix="chroma_pig_")
    chroma_client = chromadb.PersistentClient(
        path=chroma_persist_dir,
        settings=chromadb.Settings(anonymized_telemetry=False)
    )

    try:
        chroma_client.delete_collection(name="pig_manuals")
    except Exception:
        pass

    collection = chroma_client.create_collection(name="pig_manuals")
    collection.add(
        embeddings=embeddings.tolist(),
        documents=all_chunks,
        metadatas=all_metadatas,
        ids=[f"c_{i}" for i in range(len(all_chunks))]
    )

    progress_placeholder.empty()

    return collection, embed_model, len(all_chunks), lang_counts


def search_chunks(question, collection, embed_model, top_k=5):
    """질문 언어를 우선 검색하되, 부족하면 전체에서 보충"""
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 42

    try:
        q_lang = detect(question)
    except Exception:
        q_lang = "unknown"

    q_emb = embed_model.encode([question])[0].tolist()
    seen, merged_docs, merged_metas, merged_dists = set(), [], [], []

    def add(docs, metas, dists):
        for doc, meta, dist in zip(docs, metas, dists):
            key = meta["chunk_id"]
            if key not in seen and len(merged_docs) < top_k:
                seen.add(key)
                merged_docs.append(doc)
                merged_metas.append(meta)
                merged_dists.append(dist)

    # 1단계: 질문 언어 우선 검색
    if q_lang in ["ko", "en", "vi", "th", "km", "ne"]:
        try:
            r1 = collection.query(
                query_embeddings=[q_emb],
                n_results=top_k,
                where={"language": q_lang}
            )
            add(r1["documents"][0], r1["metadatas"][0], r1["distances"][0])
        except Exception:
            pass

    # 2단계: 부족하면 전체에서 보충
    if len(merged_docs) < top_k:
        try:
            r2 = collection.query(
                query_embeddings=[q_emb],
                n_results=top_k * 2
            )
            add(r2["documents"][0], r2["metadatas"][0], r2["distances"][0])
        except Exception:
            pass

    return {
        "documents": [merged_docs[:top_k]],
        "metadatas": [merged_metas[:top_k]],
        "distances": [merged_dists[:top_k]],
        "detected_language": q_lang
    }

def safe_markdown(text):
    """
    마크다운 취소선 오인 방지
    한국어 범위 표기(예: 5~30분)가 마크다운 취소선(~~)으로 인식되어
    텍스트가 가로줄로 그어지는 현상을 방지함.
    물결표 뒤에 zero-width space를 삽입하여 마크다운 파서를 회피.
    """
    if not text:
        return text
    return re.sub(r'~(?=\S)', '~\u200B', text)

LANG_NAMES_FULL = {
    "ko": "Korean (한국어)",
    "en": "English",
    "vi": "Vietnamese (Tiếng Việt)",
    "km": "Khmer (ភាសាខ្មែរ)",
    "th": "Thai (ภาษาไทย)",
    "ne": "Nepali (नेपाली)",
}


def distance_to_similarity(distance, max_dist=20.0):
    if distance is None:
        return 0
    similarity = max(0, (max_dist - distance) / max_dist) * 100
    return round(similarity, 1)


def ask_chatbot(question, collection, embed_model, gemini_client,
                top_k=5, max_retries=3):
    import time

    search_results = search_chunks(question, collection, embed_model, top_k=top_k)
    q_lang = search_results["detected_language"]
    q_lang_full = LANG_NAMES_FULL.get(q_lang, q_lang)

    context = ""
    debug_info = []
    for i, (chunk, meta, dist) in enumerate(zip(
        search_results["documents"][0],
        search_results["metadatas"][0],
        search_results["distances"][0]
    ), 1):
        section_label = "MAIN BODY" if meta["section"] == "body" else "APPENDIX"
        context += (
            f"\n[Excerpt {i} | Source: {meta['source']} | "
            f"Language: {meta['lang_name']} | Section: {section_label}]\n{chunk}\n"
        )

        debug_info.append({
            "rank": i,
            "source": meta["source"],
            "language": meta["lang_name"],
            "section": "본문" if meta["section"] == "body" else "부록",
            "similarity": distance_to_similarity(dist),
            "preview": chunk[:400] + "..." if len(chunk) > 400 else chunk
        })

    prompt = f"""You are a professional pig farming management assistant designed to help foreign workers in Korean swine farms.

# CRITICAL RULES
1. **Language**: Respond ONLY in {q_lang_full}. Do not mix languages.
2. **Source-grounded**: Answer using the manual excerpts below. The excerpts may be in different languages from the question — translate the relevant information into {q_lang_full} for your answer.
3. **Honesty**: If excerpts don't contain enough info, say so in {q_lang_full}: "The manual does not contain specific information. Please consult your farm manager or veterinarian."
4. **Numerical precision**: Include exact numbers (temperature, days, dosage) from manuals.
5. **Practical tone**: Clear, concise, actionable.
6. **Safety**: For health/disease/vaccine, emphasize professional consultation when needed.
7. **Range notation**: When expressing numerical ranges (e.g., "5 to 30 minutes"), use a hyphen "-" or write out "to" instead of the tilde "~". This prevents markdown rendering issues.

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
                model="gemini-2.5-flash-lite",
                contents=prompt
            )
            return {
                "answer": response.text,
                "language": q_lang_full,
                "sources": list(set(m["source"] for m in search_results["metadatas"][0])),
                "debug_info": debug_info,
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
        "answer": (
            f"⚠️ AI 서버가 일시적으로 응답하지 못합니다. "
            f"잠시 후 다시 시도해주세요.\n(Error: {last_error})"
        ),
        "language": q_lang_full,
        "sources": [],
        "debug_info": debug_info,
    }


# ===== 메인 로직 =====
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    st.error("⚠️ Gemini API 키가 설정되지 않았습니다.")
    st.stop()


@st.cache_resource
def init_gemini_client(key):
    from google import genai
    return genai.Client(api_key=key)


gemini_client = init_gemini_client(api_key)

collection, embed_model, num_chunks, lang_counts = build_knowledge_base()

if not st.session_state.system_ready:
    lang_summary = ", ".join(
        [f"{k}: {v}" for k, v in lang_counts.items() if v > 0]
    )
    st.success(
        f"✅ 시스템 준비 완료: 총 {num_chunks}개 청크 인덱싱됨 ({lang_summary})"
    )
    st.session_state.system_ready = True

# 채팅 UI - 이전 메시지 표시
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(safe_markdown(msg["content"]))
        if msg.get("sources"):
            with st.expander("📚 참조 매뉴얼 / Sources"):
                for src in msg["sources"]:
                    st.markdown(f"- {src}")
        if debug_mode and msg.get("debug_info"):
            with st.expander("🔍 검색된 매뉴얼 청크"):
                for info in msg["debug_info"]:
                    st.markdown(
                        f"**[{info['rank']}] {info['source']}** — "
                        f"{info['language']} / {info['section']} | "
                        f"유사도: {info['similarity']}%"
                    )
                    st.text(info['preview'])
                    st.markdown("---")

prompt = st.chat_input(
    "질문을 입력하세요 / Type your question in any supported language..."
)

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🔍 매뉴얼 검색 및 답변 생성 중..."):
            result = ask_chatbot(prompt, collection, embed_model, gemini_client)

        st.markdown(safe_markdown(result["answer"]))

        if result.get("sources"):
            with st.expander("📚 참조 매뉴얼 / Sources"):
                for src in result["sources"]:
                    st.markdown(f"- {src}")

        if debug_mode and result.get("debug_info"):
            with st.expander("🔍 검색된 매뉴얼 청크"):
                for info in result["debug_info"]:
                    st.markdown(
                        f"**[{info['rank']}] {info['source']}** — "
                        f"{info['language']} / {info['section']} | "
                        f"유사도: {info['similarity']}%"
                    )
                    st.text(info['preview'])
                    st.markdown("---")

        st.caption(f"🌐 감지된 언어 / Detected: {result['language']}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result.get("sources", []),
        "debug_info": result.get("debug_info", [])
    })
