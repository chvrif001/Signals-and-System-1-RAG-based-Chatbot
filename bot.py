import os
import asyncio
import nest_asyncio
import subprocess
import shutil
import re as re_module

# ── Matplotlib non-interactive backend (must be set before pyplot import) ──────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_together import ChatTogether
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import SentenceTransformer, util

from telegram import Update
from telegram.ext import (
    Application, CommandHandler, MessageHandler, filters, ContextTypes
)

nest_asyncio.apply()

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — all secrets come from environment variables
# ══════════════════════════════════════════════════════════════════════════════
BOT_TOKEN       = os.environ["BOT_TOKEN"]
TOGETHER_API_KEY = os.environ["TOGETHER_API_KEY"]
ADMIN_USER_ID   = int(os.environ.get("ADMIN_USER_ID", "0"))

# ── Paths ─────────────────────────────────────────────────────────────────────
# PDFs committed to the GitHub repo
PDF_FOLDER  = "./knowledge_base"

# ChromaDB stored on Railway persistent volume (mount at /data in Railway dashboard)
# Falls back to local ./chroma_db if no volume is attached (useful for local dev)
CHROMA_DIR  = os.environ.get("CHROMA_DIR", "/data/chroma_db")

# Temp folders for runtime images and plots
IMG_FOLDER  = "/tmp/student_images"
PLOT_FOLDER = "/tmp/plots"

os.makedirs(PDF_FOLDER,  exist_ok=True)
os.makedirs(CHROMA_DIR,  exist_ok=True)
os.makedirs(IMG_FOLDER,  exist_ok=True)
os.makedirs(PLOT_FOLDER, exist_ok=True)

os.environ["TOGETHER_API_KEY"] = TOGETHER_API_KEY

# ══════════════════════════════════════════════════════════════════════════════
# TrOCR — Handwriting Recognition (lazy-loaded to save startup RAM)
# ══════════════════════════════════════════════════════════════════════════════
# Model is NOT loaded at startup — only when the first photo arrives.
# This saves ~1.5GB RAM and prevents OOM crashes on Railway's free tier.
trocr_processor = None
trocr_model     = None
device          = "cuda" if torch.cuda.is_available() else "cpu"


def _load_trocr_if_needed():
    """Load TrOCR model on first use, then keep it in memory."""
    global trocr_processor, trocr_model
    if trocr_model is None:
        print("Loading TrOCR model on first photo request...")
        trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
        trocr_model     = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")
        trocr_model     = trocr_model.to(device)
        print(f"TrOCR ready on {device}")


def extract_handwritten_text(image_path: str) -> str:
    """
    Process full image first; fall back to horizontal strips for tall images.
    This avoids breaking equations across strip boundaries.
    """
    _load_trocr_if_needed()

    img           = Image.open(image_path).convert("RGB")
    width, height = img.size

    # For reasonably sized images, process in one shot
    if height <= 400:
        pixel_values = trocr_processor(
            images=img, return_tensors="pt"
        ).pixel_values.to(device)
        with torch.no_grad():
            ids = trocr_model.generate(pixel_values, max_new_tokens=512)
        return trocr_processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

    # For tall images (full-page photos), use strips
    strip_height = 120
    lines        = []
    for y in range(0, height, strip_height):
        strip        = img.crop((0, y, width, min(y + strip_height, height)))
        pixel_values = trocr_processor(
            images=strip, return_tensors="pt"
        ).pixel_values.to(device)
        with torch.no_grad():
            ids = trocr_model.generate(pixel_values, max_new_tokens=256)
        text = trocr_processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        if text:
            lines.append(text)
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL PLOTTER
# ══════════════════════════════════════════════════════════════════════════════
t_sym = sp.Symbol("t", real=True)

PLOT_KEYWORDS = [
    "plot", "draw", "graph", "sketch", "show me", "visualise", "visualize", "diagram"
]


def parse_signal_expr(text: str):
    s = text.strip()
    s = s.replace("^", "**")
    s = s.replace("{", "(").replace("}", ")")
    s = re_module.sub(r'\be\b', 'E', s)
    s = re_module.sub(r'\bu\s*\(', 'Heaviside(', s)
    s = re_module.sub(r'\b(?:delta|δ)\s*\(', 'DiracDelta(', s)
    local_ns = {
        "t": t_sym, "pi": sp.pi, "E": sp.E,
        "Heaviside": sp.Heaviside, "DiracDelta": sp.DiracDelta,
        "exp": sp.exp, "sin": sp.sin, "cos": sp.cos,
        "sqrt": sp.sqrt, "Abs": sp.Abs,
    }
    return sp.sympify(s, locals=local_ns)


def _lambdify_signal(sympy_expr):
    return sp.lambdify(
        t_sym, sympy_expr,
        modules=["numpy", {
            "Heaviside":  lambda x: np.where(np.asarray(x, dtype=float) >= 0, 1.0, 0.0),
            "DiracDelta": lambda x: np.zeros_like(np.asarray(x, dtype=float)),
        }]
    )


def _extract_expr_from_question(question: str):
    q = question.strip()
    q = re_module.sub(
        r'^(?:plot|draw|graph|sketch|show\s+me|visualise|visualize|diagram)'
        r'\s+(?:the\s+)?(?:signal\s+)?(?:function\s+)?',
        '', q, flags=re_module.IGNORECASE
    ).strip().rstrip("?.")
    if any(c in q for c in ['u(', 't', 'e', 'sin', 'cos', 'delta', 'δ', '^', '*', '+']):
        return q
    return None


def plot_from_expression(expr_str: str, message_id: int):
    try:
        sympy_expr = parse_signal_expr(expr_str)
    except Exception as exc:
        print(f"[parse] {exc}")
        return None
    try:
        t_vals = np.linspace(-4, 8, 4000)
        f_lam  = _lambdify_signal(sympy_expr)
        y_vals = np.real(f_lam(t_vals)).astype(float)
        y_vals = np.clip(y_vals, -10, 10)
    except Exception as exc:
        print(f"[eval] {exc}")
        return None

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(t_vals, y_vals, color="steelblue", linewidth=2)
    ax.axhline(0, color="k", linewidth=0.6)
    ax.axvline(0, color="k", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.set_xlabel("t"); ax.set_ylabel("x(t)")
    ax.set_title(f"x(t) = {expr_str}")
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    path = os.path.join(PLOT_FOLDER, f"plot_{message_id}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    return path


def plot_dirac_arrow(t0: float, message_id: int) -> str:
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.axhline(0, color="k", linewidth=0.8)
    ax.annotate("", xy=(t0, 1), xytext=(t0, 0),
                arrowprops=dict(arrowstyle="-|>", color="steelblue", lw=2.5))
    ax.set_xlim(t0 - 3, t0 + 3)
    ax.set_ylim(-0.2, 1.5)
    ax.set_title(f"δ(t − {t0})" if t0 != 0 else "δ(t)")
    ax.set_xlabel("t"); ax.set_ylabel("δ(t)")
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    path = os.path.join(PLOT_FOLDER, f"plot_{message_id}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    return path


def generate_plot(question: str, message_id: int):
    q = question.lower()
    if "dirac" in q or "delta" in q or "δ" in q:
        m  = re_module.search(
            r'(?:delta|δ)\s*\(\s*t\s*([+-]\s*\d*\.?\d+)?\s*\)', question
        )
        t0 = 0.0
        if m and m.group(1):
            t0 = float(m.group(1).replace(" ", ""))
        return plot_dirac_arrow(t0, message_id)

    expr_str = _extract_expr_from_question(question)
    if expr_str:
        result = plot_from_expression(expr_str, message_id)
        if result:
            return result
    return None


# ══════════════════════════════════════════════════════════════════════════════
# VECTOR STORE
# ══════════════════════════════════════════════════════════════════════════════
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def build_vector_store_if_needed(pdf_folder: str, chroma_dir: str):
    """
    Load existing ChromaDB from the persistent volume if it exists.
    Only rebuild if the index is missing or empty.
    This means after the first deploy, restarts are instant.
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    if os.path.exists(chroma_dir) and os.listdir(chroma_dir):
        print("✅ Loading existing ChromaDB from volume...")
        vs = Chroma(persist_directory=chroma_dir, embedding_function=embeddings)
        count = vs._collection.count()
        print(f"   {count} vectors loaded.")
        if count > 0:
            return vs
        print("   Index was empty — rebuilding...")

    print("📄 Building ChromaDB from PDFs in knowledge_base/...")
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
    if not pdf_files:
        print("⚠️  No PDFs found in knowledge_base/ — bot will answer without context.")
        return None

    loader   = PyPDFDirectoryLoader(pdf_folder)
    docs     = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, chunk_overlap=400,
        separators=["\n\nQuestion", "\n\nQ", "\n\n", "\n"]
    )
    chunks = splitter.split_documents(docs)
    print(f"   {len(docs)} pages → {len(chunks)} chunks")

    vs = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=chroma_dir)
    vs.persist()
    print(f"   ✅ {vs._collection.count()} vectors saved to {chroma_dir}")
    return vs


# ══════════════════════════════════════════════════════════════════════════════
# RAG CHAINS
# ══════════════════════════════════════════════════════════════════════════════
TUTOR_PROMPT = PromptTemplate.from_template(
    "You are a Signals and Systems tutor assistant.\n\n"
    "First, silently classify the student's question into one of three types:\n"
    "  A) FACTUAL — asking for course info, dates, definitions, or simple facts\n"
    "  B) CONCEPTUAL — asking to understand an idea, theorem, or technique\n"
    "  C) CALCULATION — asking to solve or explain a problem or work through math\n\n"
    "Then respond according to the type:\n"
    "  A) FACTUAL → Answer in 2-4 sentences. No steps, no examples.\n"
    "  B) CONCEPTUAL → Explain clearly in a short paragraph of 4-6 sentences. "
    "Give exactly one simple example. No numbered steps.\n"
    "  C) CALCULATION → Work through or explain it step by step with numbered steps. "
    "Explain every formula and symbol used.\n\n"
    "Do not mention the classification in your response. Just answer.\n"
    "If the answer is not in the context, say so honestly.\n\n"
    "Context from course documents:\n{context}\n\n"
    "Student question:\n{question}\n\n"
    "Answer:"
)

CHECK_PROMPT = PromptTemplate.from_template(
    "You are a Signals and Systems tutor marking a student's handwritten solution.\n\n"
    "The correct memo solution is provided below as context.\n"
    "The student's handwritten work has been extracted via OCR and may contain small "
    "recognition errors — use your judgement to interpret garbled symbols.\n\n"
    "Your task:\n"
    "1. State whether the student's answer is CORRECT, PARTIALLY CORRECT, or INCORRECT.\n"
    "2. Identify every error or missing step clearly.\n"
    "3. Explain what the correct approach should be for each error.\n"
    "4. Be encouraging — point out what they did right too.\n\n"
    "Memo context (correct solution):\n{context}\n\n"
    "Student's handwritten work (OCR extracted):\n{question}\n\n"
    "Marking feedback:"
)

GENERAL_PHOTO_PROMPT = PromptTemplate.from_template(
    "You are a patient Signals and Systems tutor. A student has sent you a photo of "
    "their handwritten work and has given you a specific instruction.\n\n"
    "The handwriting has been extracted via OCR and may contain small recognition errors.\n\n"
    "Student's instruction: {task}\n\n"
    "Student's handwritten work (OCR extracted):\n{question}\n\n"
    "Rules:\n"
    "- Respond directly to what the student asked.\n"
    "- Use clear numbered steps where appropriate.\n"
    "- Explain your reasoning — don't just state conclusions.\n"
    "- Be encouraging: acknowledge what they got right, then address errors.\n"
    "- If the OCR looks garbled in a critical spot, say so and give your best interpretation.\n\n"
    "Your response:"
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_chains(vs):
    retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    llm = ChatTogether(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        temperature=0.7,
        max_tokens=1024,
    )

    def make_rag(prompt):
        return (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt | llm | StrOutputParser()
        )

    general_photo_chain = (
        {"task": lambda x: x["task"], "question": lambda x: x["question"]}
        | GENERAL_PHOTO_PROMPT | llm | StrOutputParser()
    )

    return make_rag(TUTOR_PROMPT), make_rag(CHECK_PROMPT), general_photo_chain


# ── Boot up vector store and chains ───────────────────────────────────────────
vector_store = build_vector_store_if_needed(PDF_FOLDER, CHROMA_DIR)
qa_chain = check_chain = photo_chain = None

if vector_store:
    qa_chain, check_chain, photo_chain = build_chains(vector_store)
    print("✅ RAG chains ready")
else:
    print("⚠️  Running without knowledge base — add PDFs to knowledge_base/ and redeploy.")

# ── Semantic scorer (for eval, not used in bot responses) ─────────────────────
scorer = SentenceTransformer(EMBEDDING_MODEL)


# ══════════════════════════════════════════════════════════════════════════════
# TELEGRAM HANDLERS
# ══════════════════════════════════════════════════════════════════════════════
async def send_long(update: Update, text: str):
    """Split messages that exceed Telegram's 4096 character limit."""
    for i in range(0, len(text), 4096):
        await update.message.reply_text(text[i:i + 4096])


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Hi! I'm your Signals & Systems tutor bot.\n\n"
        "Here's what I can do:\n"
        "📚 Answer questions about course notes and memos\n"
        "📊 Plot any signal expression:\n"
        "   e.g. _plot 2u(t-2)_\n"
        "   e.g. _draw u(t) - u(t-3)_\n"
        "   e.g. _sketch e^{-t}u(t)_\n"
        "📷 Work with your handwritten solutions — just send a photo\n\n"
        "💡 *Pro tip for photos:* Add a caption to tell me what you want!\n"
        "   No caption? I'll compare it to the memo automatically.\n\n"
        "Use /help for the full guide.",
        parse_mode="Markdown"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🆘 *How to use this bot:*\n\n"
        "1️⃣ *Ask a question* — just type it\n"
        "   e.g. 'When is the class test?'\n"
        "   e.g. 'Explain convolution'\n\n"
        "2️⃣ *Plot a signal* — I parse the expression exactly\n"
        "   e.g. _plot 2*u(t-2)_\n"
        "   e.g. _draw u(t) + u(t+1)_\n"
        "   e.g. _sketch u(t) - u(t-3)_\n"
        "   e.g. _graph e^{-2t}*u(t)_\n"
        "   e.g. _plot sin(2*pi*t)_\n"
        "   e.g. _draw delta(t-1)_\n\n"
        "3️⃣ *Work with your handwritten solutions* — send a photo\n"
        "   📌 *With a caption* — I'll do exactly what you ask\n"
        "   📌 *No caption* — I'll compare to the memo\n\n"
        "📌 For best OCR results: good lighting, flat paper, clear writing",
        parse_mode="Markdown"
    )


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    question = update.message.text.strip()
    q_lower  = question.lower()
    msg_id   = update.message.message_id

    # ── Plot request ──────────────────────────────────────────────────────────
    if any(kw in q_lower for kw in PLOT_KEYWORDS):
        await update.message.reply_text("📊 Generating your plot…")
        fig_path = generate_plot(question, msg_id)
        if fig_path and os.path.exists(fig_path):
            await update.message.reply_photo(
                photo=open(fig_path, "rb"),
                caption=f"📈 {question}"
            )
        else:
            await update.message.reply_text(
                "⚠️ I couldn't parse that expression.\n"
                "Try: _plot 2*u(t-2)_ or _draw u(t) + u(t-1)_",
                parse_mode="Markdown"
            )
        return

    # ── Tutor question ────────────────────────────────────────────────────────
    if not qa_chain:
        await update.message.reply_text(
            "⚠️ No knowledge base loaded yet. The admin needs to add PDFs."
        )
        return

    await update.message.reply_text("🤔 Thinking…")
    try:
        answer = qa_chain.invoke(question)
        await send_long(update, answer)
    except Exception as e:
        await update.message.reply_text(f"❌ Something went wrong: {str(e)}")


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    caption = (update.message.caption or "").strip()
    await update.message.reply_text(
        "📷 Got your photo! Running handwriting recognition… "
        "(this takes ~15–30s on CPU)"
    )
    try:
        photo_file = await update.message.photo[-1].get_file()
        img_path   = os.path.join(IMG_FOLDER, f"{update.message.message_id}.jpg")
        await photo_file.download_to_drive(img_path)

        extracted = extract_handwritten_text(img_path)
        await update.message.reply_text(
            f"📝 Extracted handwriting:\n\n{extracted}\n\n⏳ Processing…"
        )

        if caption:
            if not photo_chain:
                await update.message.reply_text(
                    "⚠️ Bot not fully initialised yet."
                )
                return
            response = photo_chain.invoke({"task": caption, "question": extracted})
        else:
            if not check_chain:
                await update.message.reply_text(
                    "⚠️ No memo loaded. Add a caption to tell me what you'd like me to do."
                )
                return
            response = check_chain.invoke(extracted)

        await send_long(update, response)
    except Exception as e:
        await update.message.reply_text(f"❌ OCR failed: {str(e)}")


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Admin-only: upload a new PDF to the knowledge base.
    Only the ADMIN_USER_ID can trigger this — all other users are blocked.
    """
    global vector_store, qa_chain, check_chain, photo_chain

    # ── Admin gate ────────────────────────────────────────────────────────────
    if update.effective_user.id != ADMIN_USER_ID:
        await update.message.reply_text(
            "⚠️ Only the administrator can upload documents to the knowledge base."
        )
        return

    doc = update.message.document
    if not doc.file_name.endswith(".pdf"):
        await update.message.reply_text("⚠️ Please send a PDF file.")
        return

    await update.message.reply_text(
        f"📄 Received *{doc.file_name}* — adding to knowledge base…",
        parse_mode="Markdown"
    )
    try:
        pdf_file = await doc.get_file()
        pdf_path = os.path.join(PDF_FOLDER, doc.file_name)
        await pdf_file.download_to_drive(pdf_path)

        # Force rebuild of the index with the new PDF included
        import shutil
        if os.path.exists(CHROMA_DIR):
            shutil.rmtree(CHROMA_DIR)
        os.makedirs(CHROMA_DIR, exist_ok=True)

        vector_store                       = build_vector_store_if_needed(PDF_FOLDER, CHROMA_DIR)
        qa_chain, check_chain, photo_chain = build_chains(vector_store)

        await update.message.reply_text(
            f"✅ *{doc.file_name}* added! Knowledge base now has "
            f"{vector_store._collection.count()} vectors.\n\n"
            "Students can now ask questions about this document.",
            parse_mode="Markdown"
        )
    except Exception as e:
        await update.message.reply_text(f"❌ Failed to process PDF: {str(e)}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
async def main():
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help",  help_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.PHOTO,                   handle_photo))
    app.add_handler(MessageHandler(filters.Document.PDF,            handle_document))
    print("✅ Bot is running!")
    await app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    asyncio.run(main())
