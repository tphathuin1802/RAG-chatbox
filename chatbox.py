import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

HERE = Path(__file__).resolve().parent
PAPERS_DIR = HERE / "papers"
INDEX_DIR = HERE / ".rag_index_faiss"

TEMPLATE = (
    "You are a strict, citation-focused assistant for a private knowledge base.\n"
    "RULES:\n"
    "1) Use ONLY the provided context to answer.\n"
    '2) If the answer is not clearly contained in the context, say: "I don\'t know based on the provided documents."\n'
    "3) Do NOT use outside knowledge, guessing, or web information.\n"
    "4) Cite sources as [1], [2], ... matching the Sources list.\n"
    "\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n"
)


def load_env() -> None:
    load_dotenv(HERE / ".env")
    if not os.getenv("GOOGLE_API_KEY"):
        for alt in ("GEMINI_API_KEY", "GOOGLE_GENAI_API_KEY"):
            if os.getenv(alt):
                os.environ["GOOGLE_API_KEY"] = os.environ[alt]
                break

    # LM Studio convenience defaults
    os.environ.setdefault("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
    os.environ.setdefault("LMSTUDIO_API_KEY", "lm-studio")
    os.environ.setdefault(
        "LMSTUDIO_EMBED_MODEL", "text-embedding-nomic-embed-text-v1.5"
    )


def get_embeddings():
    """
    Priority:
    - If LMSTUDIO_BASE_URL is set => use LM Studio embeddings (e.g. nomic)
    - Else => use Gemini embeddings
    """
    lm_base = os.getenv("LMSTUDIO_BASE_URL", "").strip()
    lm_model = os.getenv("LMSTUDIO_EMBED_MODEL", "").strip()
    if lm_base and lm_model:
        return OpenAIEmbeddings(
            model=lm_model,
            api_key=os.getenv("LMSTUDIO_API_KEY", "lm-studio"),
            base_url=lm_base,
        )

    return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")


def get_llm():
    """
    Priority:
    - If LMSTUDIO_CHAT_MODEL is set => use LM Studio ChatOpenAI
    - Else => use Gemini chat model
    """
    lm_base = os.getenv("LMSTUDIO_BASE_URL", "").strip()
    lm_chat_model = os.getenv("LMSTUDIO_CHAT_MODEL", "").strip()
    if lm_base and lm_chat_model:
        return ChatOpenAI(
            model=lm_chat_model,
            api_key=os.getenv("LMSTUDIO_API_KEY", "lm-studio"),
            base_url=lm_base,
            temperature=0.2,
        )

    return ChatGoogleGenerativeAI(
        model="models/gemini-2.0-flash",
        temperature=0.2,
    )


def build_or_load_vectorstore() -> FAISS:
    embeddings = get_embeddings()

    if INDEX_DIR.exists():
        # FAISS.load_local uses pickle for docstore; enable explicit opt-in.
        return FAISS.load_local(
            str(INDEX_DIR),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    if not PAPERS_DIR.exists():
        raise FileNotFoundError(
            f"Missing folder: {PAPERS_DIR}. Create it and put PDFs inside."
        )

    loader = DirectoryLoader(
        path=str(PAPERS_DIR),
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
        use_multithreading=True,
    )
    docs = loader.load()
    if not docs:
        raise RuntimeError(f"No PDFs found under {PAPERS_DIR}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        add_start_index=True,
        strip_whitespace=True,
    )
    chunks = text_splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(chunks, embeddings, distance_strategy="cosine")
    vectorstore.save_local(str(INDEX_DIR))
    return vectorstore


def format_sources(docs) -> str:
    lines: list[str] = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source") or d.metadata.get("file_path") or "unknown"
        page = d.metadata.get("page")
        if page is not None:
            lines.append(f"[{i}] {src} (page {page})")
        else:
            lines.append(f"[{i}] {src}")
    return "\n".join(lines)


def main() -> int:
    load_env()
    # Validate that at least one provider is configured
    has_lmstudio = bool(os.getenv("LMSTUDIO_BASE_URL", "").strip())
    has_gemini = bool(os.getenv("GOOGLE_API_KEY", "").strip())
    if not has_lmstudio and not has_gemini:
        print(
            "No provider configured.\n"
            "- For Gemini: set GOOGLE_API_KEY in .env\n"
            "- For LM Studio: set LMSTUDIO_BASE_URL in .env (default is http://localhost:1234/v1)\n",
            file=sys.stderr,
        )
        return 2

    vectorstore = build_or_load_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 5, "score_threshold": 0.2},
    )

    llm = get_llm()

    system = SystemMessage(content="Follow the template and rules strictly.")

    history: list = [system]
    last_sources = ""

    print("RAG chatbox ready.")
    print("- Type your question and press Enter.")
    print("- Commands: /sources (show last sources), /reload (rebuild index), /exit")

    while True:
        try:
            q = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return 0

        if not q:
            continue
        if q.lower() in ("/exit", "exit", "quit"):
            print("Bye.")
            return 0
        if q.lower() == "/sources":
            print(last_sources or "(no sources yet)")
            continue
        if q.lower() == "/reload":
            if INDEX_DIR.exists():
                # Best-effort cleanup; rebuild will overwrite anyway if this fails.
                for p in INDEX_DIR.glob("*"):
                    try:
                        p.unlink()
                    except Exception:
                        pass
                try:
                    INDEX_DIR.rmdir()
                except Exception:
                    pass
            vectorstore = build_or_load_vectorstore()
            retriever = vectorstore.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": 5, "score_threshold": 0.2},
            )
            print("Index reloaded.")
            continue

        docs = retriever.invoke(q)
        context = "\n\n".join(
            f"Source {i+1}:\n{d.page_content}" for i, d in enumerate(docs)
        )
        last_sources = format_sources(docs) if docs else "(no relevant chunks found)"

        templated_prompt = TEMPLATE.format(
            context=(context if context else "(empty)"),
            question=q,
        )

        user_msg = HumanMessage(
            content=(
                f"{templated_prompt}\n"
                f"\nSources:\n{last_sources}\n"
            )
        )

        # Keep a small rolling window of chat history (plus system)
        history.append(user_msg)
        history = [history[0]] + history[-8:]

        try:
            ai = llm.invoke(history)
        except (ChatGoogleGenerativeAIError, Exception) as e:
            msg = str(e)
            if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
                # Gemini quota/rate-limit. It often includes "retryDelay": "42s"
                retry_s = None
                for token in ("retryDelay': '", 'retryDelay": "'):
                    if token in msg:
                        tail = msg.split(token, 1)[1]
                        # tail starts with e.g. 42s
                        n = ""
                        for ch in tail:
                            if ch.isdigit():
                                n += ch
                            else:
                                break
                        if n:
                            retry_s = int(n)
                        break

                if retry_s is not None:
                    print(
                        f"\nAssistant> (Gemini quota hit: 429 RESOURCE_EXHAUSTED. Retry after ~{retry_s}s)"
                    )
                    print("Assistant> Tip: wait and ask again, or upgrade/enable billing on your Gemini API key.")
                else:
                    print("\nAssistant> (Gemini quota hit: 429 RESOURCE_EXHAUSTED. Please wait and try again.)")
                    print("Assistant> Tip: upgrade/enable billing on your Gemini API key.")
                continue

            print(f"\nAssistant> (Model error) {msg}")
            continue

        history.append(AIMessage(content=ai.content))
        print(f"\nAssistant> {ai.content}")


if __name__ == "__main__":
    raise SystemExit(main())