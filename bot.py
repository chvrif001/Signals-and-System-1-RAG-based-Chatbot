import os
import asyncio
import nest_asyncio
import re
import base64
import requests
import tempfile
import textwrap
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_together import ChatTogether
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

from telegram import Update, request as tg_request
from telegram.ext import (
    ApplicationBuilder, CommandHandler,
    MessageHandler, filters, ContextTypes
)

nest_asyncio.apply()

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
BOT_TOKEN         = os.environ["BOT_TOKEN"]
TOGETHER_API_KEY  = os.environ["TOGETHER_API_KEY"]
GEMINI_API_KEY    = os.environ.get("GEMINI_API_KEY", "")
ADMIN_USER_ID     = int(os.environ.get("ADMIN_USER_ID", "0"))

LLM_MODEL         = "meta-llama/Meta-Llama-3-8B-Instruct-Lite"
TOGETHER_ENDPOINT = "https://api.together.xyz/v1/chat/completions"

PDF_FOLDER  = "./knowledge_base"
CHROMA_DIR  = os.environ.get("CHROMA_DIR", "/data/chroma_db")
PLOT_FOLDER = "/tmp/plots"

os.makedirs(PDF_FOLDER,  exist_ok=True)
os.makedirs(CHROMA_DIR,  exist_ok=True)
os.makedirs(PLOT_FOLDER, exist_ok=True)

os.environ["TOGETHER_API_KEY"] = TOGETHER_API_KEY

# ══════════════════════════════════════════════════════════════════════════════
# SAFE REPLY WRAPPER (retry on ConnectTimeout / TimedOut)
# ══════════════════════════════════════════════════════════════════════════════
async def _safe_reply(update: Update, text: str,
                      parse_mode: str | None = None,
                      retries: int = 3,
                      delay: float = 3.0) -> None:
    """Send a text reply with automatic retry on network timeout."""
    for attempt in range(1, retries + 1):
        try:
            await update.message.reply_text(text, parse_mode=parse_mode)
            return
        except Exception as exc:
            print(f"[safe_reply] attempt {attempt}/{retries} failed: {exc}")
            if attempt < retries:
                await asyncio.sleep(delay * attempt)
    print("[safe_reply] all retries exhausted — message not delivered")


async def _safe_reply_photo(update: Update, path: str,
                             caption: str = "",
                             retries: int = 3,
                             delay: float = 5.0) -> bool:
    """Send a photo reply with automatic retry."""
    for attempt in range(1, retries + 1):
        try:
            with open(path, "rb") as f:
                await update.message.reply_photo(
                    photo=f, caption=caption[:1024])
            return True
        except Exception as exc:
            print(f"[safe_reply_photo] attempt {attempt}/{retries} failed: {exc}")
            if attempt < retries:
                await asyncio.sleep(delay * attempt)
    return False

# keep old name as alias so nothing breaks
_send_photo_with_retry = _safe_reply_photo

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STORE
# ══════════════════════════════════════════════════════════════════════════════
_session_store: dict[int, dict] = {}

def session_store(chat_id: int, text: str, source: str) -> None:
    _session_store[chat_id] = {"text": text, "source": source}

def session_get(chat_id: int) -> dict | None:
    return _session_store.get(chat_id)

def session_clear(chat_id: int) -> None:
    _session_store.pop(chat_id, None)

def session_has(chat_id: int) -> bool:
    return chat_id in _session_store

# ══════════════════════════════════════════════════════════════════════════════
# SYMPY SYMBOLS
# ══════════════════════════════════════════════════════════════════════════════
t_sym = sp.Symbol("t",     real=True)
s_sym = sp.Symbol("s",     complex=True)
w_sym = sp.Symbol("omega", real=True)
n_sym = sp.Symbol("n",     integer=True)
tau   = sp.Symbol("tau",   real=True)
a_sym = sp.Symbol("a",     positive=True)

# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL PARSER
# ══════════════════════════════════════════════════════════════════════════════
def _rect(x):
    return sp.Piecewise(
        (sp.Integer(1),      sp.Abs(x) < sp.Rational(1, 2)),
        (sp.Rational(1, 2),  sp.Eq(sp.Abs(x), sp.Rational(1, 2))),
        (sp.Integer(0),      True)
    )

def _normalise(expr: str) -> str:
    s = expr.strip()
    s = s.replace("^", "**").replace("{", "(").replace("}", ")")
    s = re.sub(r'\brect\s*\(',                      'rect(',          s)
    s = re.sub(r'\bu\s*\(\s*t\s*\)',                'Heaviside(t)',   s)
    s = re.sub(r'\bu\s*\(\s*t\s*([+-][^)]+)\)',     r'Heaviside(t\1)', s)
    s = re.sub(r'\bE\*\*(-[^\s\*\+\-\(\),]+)',      r'E**(\1)',       s)
    s = re.sub(r'\be\*\*(-[^\s\*\+\-\(\),]+)',      r'E**(\1)',       s)
    s = re.sub(r'(\d)(t\b)',                         r'\1*\2',         s)
    s = re.sub(r'(\d)(sin|cos|exp|sqrt|rect)',       r'\1*\2',         s)
    s = re.sub(r'(\d)\s*[eE]\*\*',                  r'\1*E**',        s)
    s = re.sub(r'\be\b',                             'E',              s)
    s = re.sub(r'\bu\s*[\(\[]',                      'Heaviside(',     s)
    s = re.sub(r'\]',                                ')',               s)
    s = re.sub(r'\b(?:delta|δ)\s*[\(\[]',           'DiracDelta(',    s)
    s = re.sub(
        r'\bsinc\(([^)]+)\)',
        lambda m: f'(sin(pi*({m.group(1)})))/(pi*({m.group(1)}))',
        s
    )
    return s

_COMMON_NS = {
    "t": t_sym, "s": s_sym, "omega": w_sym, "n": n_sym,
    "pi": sp.pi, "E": sp.E, "j": sp.I,
    "Heaviside":  sp.Heaviside,
    "DiracDelta": sp.DiracDelta,
    "exp":  sp.exp,  "sin": sp.sin, "cos": sp.cos,
    "sqrt": sp.sqrt, "Abs": sp.Abs, "log": sp.log,
    "rect": _rect,
}

def parse_ct_expr(text: str) -> sp.Expr:
    return sp.sympify(_normalise(text), locals=_COMMON_NS)

def _normalise_dt(expr: str) -> str:
    s = expr.strip()
    s = s.replace("^", "**").replace("{", "(").replace("}", ")")
    s = re.sub(r'\be\b', 'E', s)
    s = re.sub(r'\bu\s*\[([^\]]+)\]',                r'UnitStep(\1)', s)
    s = re.sub(r'\b(?:delta|δ)\s*\[([^\]]+)\]',      r'KronDelta(\1)', s)
    s = re.sub(r'(\d)(n\b)',                          r'\1*\2', s)
    return s

def _unit_step_dt(val):
    arr = np.asarray(val, dtype=float)
    return np.where(arr >= 0, 1.0, 0.0)

def _kron_delta_dt(val):
    arr = np.asarray(val, dtype=float)
    return (arr == 0).astype(float)

def parse_dt_expr(text: str):
    s = _normalise_dt(text)
    ns = {
        "n": None, "pi": np.pi, "E": np.e,
        "exp": np.exp, "sin": np.sin, "cos": np.cos,
        "sqrt": np.sqrt, "abs": np.abs, "Abs": np.abs,
        "UnitStep": _unit_step_dt, "KronDelta": _kron_delta_dt, "log": np.log,
    }
    def evaluator(n_array: np.ndarray) -> np.ndarray:
        local = dict(ns)
        local["n"] = n_array
        return np.real(eval(s, {"__builtins__": {}}, local)).astype(float)  # noqa: S307
    return evaluator

_SIGNAL_CHARS = ['(', '[', 't', 'n', 's', 'sin', 'cos', 'exp',
                 'delta', 'δ', '**', '*', '+', 'sqrt', 'log',
                 'Heaviside', 'DiracDelta', 'rect']

_QUESTION_PREFIX = re.compile(
    r'^(?:what\s+is\s+(?:the\s+)?|what\'s\s+(?:the\s+)?|find\s+(?:the\s+)?|'
    r'compute\s+(?:the\s+)?|calculate\s+(?:the\s+)?|determine\s+(?:the\s+)?|'
    r'give\s+me\s+(?:the\s+)?|show\s+me\s+(?:the\s+)?)',
    re.IGNORECASE
)

_VERB_PREFIX = re.compile(
    r'^(?:plot|draw|graph|sketch|visualis[ae]|diagram|'
    r'laplace\s+(?:transform\s+)?(?:of\s+)?|'
    r'inverse\s+laplace\s+(?:transform\s+)?(?:of\s+)?|'
    r'(?:i)?ilt\s+(?:of\s+)?|'
    r'fourier\s+(?:transform\s+)?(?:of\s+)?|'
    r'inverse\s+fourier\s+(?:transform\s+)?(?:of\s+)?|'
    r'(?:i)?ift\s+(?:of\s+)?|'
    r'(?:i)?ft\s+(?:of\s+)?|f\.?t\.?\s+(?:of\s+)?|fourier\s+series\s+(?:of\s+)?|'
    r'convolve\s+|convolution\s+(?:of\s+)?|compute\s+|calculate\s+|find\s+)',
    re.IGNORECASE
)

_TRAILING_NOISE = re.compile(
    r'\s+(?:looks?\s+like|for\s+me|please|now|here|to\s+me|as\s+well)\s*$',
    re.IGNORECASE
)

def extract_expr(question: str) -> str | None:
    q = question.strip()
    q = _QUESTION_PREFIX.sub('', q).strip()
    q = _VERB_PREFIX.sub('', q).strip()
    q = _TRAILING_NOISE.sub('', q)
    q = q.rstrip("?.")
    if any(c in q for c in _SIGNAL_CHARS):
        return q
    if re.match(r'^-?[\d][\d\.]*$', q.strip()):
        return q
    if re.search(r'\be\b', q):
        return q
    return None

# ══════════════════════════════════════════════════════════════════════════════
# RESPONSE-MODE GATE
# ══════════════════════════════════════════════════════════════════════════════
_MATH_TITLE_KEYWORDS = (
    "laplace", "fourier", "convolution", "transform", "series",
    "plot", "signal", "derivation", "calculation", "equation",
    "answer —", "marking feedback", "tutor answer",
    "inverse laplace", "inverse fourier",
)
_PLAIN_TEXT_TITLES = ("tutor answer",)

def _is_math_title(title: str) -> bool:
    t = title.lower()
    if any(kw in t for kw in _PLAIN_TEXT_TITLES):
        return False
    return any(kw in t for kw in _MATH_TITLE_KEYWORDS)

# ══════════════════════════════════════════════════════════════════════════════
# LATEX / MATHTEXT SANITISER
# ══════════════════════════════════════════════════════════════════════════════
def _sanitise_mathtext(s: str) -> str:
    s = s.replace('∞', r'\infty')
    s = s.replace('∑', r'\sum')
    s = s.replace('∫', r'\int')
    s = s.replace('∏', r'\prod')
    s = s.replace('δ', r'\delta')
    s = s.replace('Δ', r'\Delta')
    s = s.replace('ω', r'\omega')
    s = s.replace('Ω', r'\Omega')
    s = s.replace('π', r'\pi')
    s = s.replace('α', r'\alpha')
    s = s.replace('β', r'\beta')
    s = s.replace('τ', r'\tau')
    s = s.replace('σ', r'\sigma')
    s = s.replace('μ', r'\mu')
    s = s.replace('λ', r'\lambda')
    s = s.replace('θ', r'\theta')
    s = s.replace('φ', r'\phi')
    s = s.replace('★', r'\star')
    s = s.replace('·', r'\cdot')
    s = s.replace('×', r'\times')
    s = s.replace('≈', r'\approx')
    s = s.replace('≥', r'\geq')
    s = s.replace('≤', r'\leq')
    s = s.replace('≠', r'\neq')
    s = re.sub(r'\\theta\\left\(([^)]+)\\right\)', r'u(\1)', s)
    s = s.replace(r'\theta\left(t\right)', r'u(t)')
    s = re.sub(r'(?<![a-zA-Z\\])i(?![a-zA-Z0-9{\\])', 'j', s)
    s = re.sub(r'\\mathcal\{([^}]+)\}',     r'\\mathbf{\1}', s)
    s = re.sub(r'\\mathscr\{([^}]+)\}',     r'\\mathbf{\1}', s)
    s = re.sub(r'\\mathrm\{([^}]+)\}',      r'\\rm \1',      s)
    s = re.sub(r'\\operatorname\{([^}]+)\}', r'\\rm \1',      s)
    s = re.sub(r'\\text\{([^}]*)\}',         r'\\rm \1',      s)
    s = s.replace(r'\qquad', r'\ \ \ \ ')
    s = s.replace(r'\quad',  r'\ \ ')
    s = s.replace(r'\,',     r'\ ')
    s = s.replace(r'\;',     r'\ ')
    s = re.sub(r'\\left\s*',  '', s)
    s = re.sub(r'\\right\s*', '', s)
    s = re.sub(
        r'(\\(?:pi|alpha|beta|gamma|delta|epsilon|zeta|eta|theta|iota|kappa|'
        r'lambda|mu|nu|xi|rho|sigma|tau|upsilon|phi|chi|psi|omega|'
        r'Gamma|Delta|Theta|Lambda|Xi|Pi|Sigma|Upsilon|Phi|Psi|Omega|'
        r'infty|cdot|times|star|approx|geq|leq|neq|sum|int|prod|rm|mathbf))'
        r'(?=[A-Za-z])',
        r'\1 ', s
    )
    return s

# ══════════════════════════════════════════════════════════════════════════════
# LATEX MATH RENDERER
# ══════════════════════════════════════════════════════════════════════════════
def _sympy_to_latex(expr: sp.Expr) -> str:
    return sp.latex(expr)

def _try_render_row(ax, x_label, x_expr, y, label, latex_str, fontsize=22):
    ax.text(x_label, y, f"{label}:",
            transform=ax.transAxes,
            fontsize=fontsize - 1, fontweight="bold",
            color="#0055aa", va="top", ha="left", usetex=False)
    if not latex_str:
        return
    safe = _sanitise_mathtext(latex_str)
    try:
        ax.text(x_expr, y, f"${safe}$",
                transform=ax.transAxes,
                fontsize=fontsize, color="#111111",
                va="top", ha="left", usetex=False)
    except Exception as e:
        print(f"[render row fallback] {label}: {e}")
        ax.text(x_expr, y, latex_str,
                transform=ax.transAxes,
                fontsize=fontsize - 3, color="#333333",
                va="top", ha="left", fontfamily="monospace", usetex=False)

def _render_math_png(title: str, steps: list[tuple[str, str]], msg_id: int) -> str | None:
    try:
        n      = len(steps)
        FONT   = 22
        ROW_IN = 0.95
        PAD_IN = 1.6
        fig_h  = max(4.0, n * ROW_IN + PAD_IN)
        fig_w  = 16.0

        fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="white")
        ax.set_facecolor("white")
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        title_clean = re.sub(r'[^\x00-\x7F]+', '', title).strip()
        ax.text(0.5, 0.97, title_clean,
                transform=ax.transAxes,
                fontsize=FONT + 4, fontweight="bold",
                ha="center", va="top", color="#1a1a2e", usetex=False)

        divider_y = 1.0 - (PAD_IN * 0.45 / fig_h)
        ax.plot([0.01, 0.99], [divider_y, divider_y],
                color="#aaaaaa", linewidth=1.2,
                transform=ax.transAxes)

        row_frac = ROW_IN / fig_h
        y = divider_y - (0.15 / fig_h) - row_frac * 0.15

        for label, latex_str in steps:
            _try_render_row(ax, 0.01, 0.28, y, label, latex_str, fontsize=FONT)
            y -= row_frac
            if y < 0.01:
                break

        try:
            fig.tight_layout(pad=0.5)
        except Exception as e:
            print(f"[_render_math_png] tight_layout non-fatal: {e}")

        path = os.path.join(PLOT_FOLDER, f"math_{msg_id}.png")
        fig.savefig(path, dpi=180, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        plt.close("all")
        return path
    except Exception as e:
        print(f"[_render_math_png] FAILED: {e}")
        plt.close("all")
        return None

# ══════════════════════════════════════════════════════════════════════════════
# LLM RESPONSE RENDERER
# ══════════════════════════════════════════════════════════════════════════════
_PLAIN_TO_MATHTEXT = [
    (r'\bω\b',  r'\\omega'),
    (r'\bΩ\b',  r'\\Omega'),
    (r'\bπ\b',  r'\\pi'),
    (r'\bδ\b',  r'\\delta'),
    (r'\bτ\b',  r'\\tau'),
    (r'\bα\b',  r'\\alpha'),
    (r'\bβ\b',  r'\\beta'),
    (r'\bσ\b',  r'\\sigma'),
    (r'\^(-?[\w/]+)',    r'^{\1}'),
    (r'e\^\{([^}]+)\}', r'e^{\1}'),
    (r'\bsinc\b',        r'\\mathrm{sinc}'),
]

_EQ_LINE_RE = re.compile(
    r'^.*(?:'
    r'[A-Za-zΩωπδτ]\([^)]*\)\s*='
    r'|=\s*\\frac'
    r'|\\int'
    r'|[∫∑∏∞]'
    r'|(?:\^|_)\{[^}]+\}'
    r').*$',
    re.IGNORECASE
)

def _plain_to_mt(expr: str) -> str:
    for pat, rep in _PLAIN_TO_MATHTEXT:
        expr = re.sub(pat, rep, expr)
    return expr

def _extract_math_blocks(text: str) -> list[tuple[str, str]]:
    segments = []
    parts = re.split(r'(\$\$[\s\S]+?\$\$|\$[^$\n]+?\$)', text)
    for part in parts:
        if part.startswith('$$') and part.endswith('$$'):
            segments.append(('math', part[2:-2].strip()))
        elif part.startswith('$') and part.endswith('$'):
            segments.append(('math', part[1:-1].strip()))
        else:
            segments.append(('prose', part))
    return segments

def _split_prose_lines(prose: str) -> list[tuple[str, str]]:
    out = []
    for line in prose.split('\n'):
        stripped = line.strip()
        if _EQ_LINE_RE.match(stripped) and len(stripped) > 5:
            out.append(('math', _plain_to_mt(stripped)))
        else:
            out.append(('prose', line))
    return out

def render_response_png(llm_text: str, title: str, msg_id: int) -> str | None:
    rows: list[tuple[str, str]] = []
    for kind, content in _extract_math_blocks(llm_text):
        if kind == 'math':
            rows.append(('math', content))
        else:
            rows.extend(_split_prose_lines(content))

    while rows and rows[0][1].strip() == '':
        rows.pop(0)
    while rows and rows[-1][1].strip() == '':
        rows.pop()

    if not rows:
        return None

    FIG_W      = 22.0
    PROSE_FS   = 22
    MATH_FS    = 28
    LINE_H     = 0.75
    MATH_H     = 1.10
    TITLE_H    = 1.10
    PAD        = 0.8
    WRAP_WIDTH = 100

    total_h = TITLE_H + PAD
    for kind, txt in rows:
        if kind == 'math':
            total_h += MATH_H
        else:
            n_lines = max(1, len(textwrap.wrap(txt, width=WRAP_WIDTH)) if txt.strip() else 1)
            total_h += LINE_H * n_lines
    total_h = max(8.0, total_h)

    MAX_PAGE_H = 40.0
    render_response_png._extra_pages = []

    if total_h > MAX_PAGE_H:
        pages = []
        page_rows = []
        page_h = TITLE_H + PAD
        for row in rows:
            kind, txt = row
            row_h = MATH_H if kind == 'math' else \
                    LINE_H * max(1, len(textwrap.wrap(txt, width=WRAP_WIDTH)) if txt.strip() else 1)
            if page_h + row_h > MAX_PAGE_H and page_rows:
                pages.append(page_rows)
                page_rows = [row]
                page_h = TITLE_H + PAD + row_h
            else:
                page_rows.append(row)
                page_h += row_h
        if page_rows:
            pages.append(page_rows)

        for pi, p_rows in enumerate(pages[1:], start=2):
            p_h = max(8.0, sum(
                MATH_H if k == 'math'
                else LINE_H * max(1, len(textwrap.wrap(t, width=WRAP_WIDTH)) if t.strip() else 1)
                for k, t in p_rows
            ) + TITLE_H + PAD)
            p_fig, p_ax = plt.subplots(figsize=(FIG_W, p_h), facecolor='white')
            p_ax.set_facecolor('white'); p_ax.axis('off')
            p_ax.set_xlim(0, 1); p_ax.set_ylim(0, p_h)
            p_y = p_h - 0.15
            p_ax.text(0.5, p_y, f"{title}  (page {pi})",
                      fontsize=18, fontweight='bold', color='#1a1a2e',
                      ha='center', va='top', usetex=False)
            p_y -= TITLE_H
            p_ax.plot([0.02, 0.98], [p_y + 0.10, p_y + 0.10], color='#cccccc', linewidth=1.0)
            for kind, txt in p_rows:
                if not txt.strip():
                    p_y -= LINE_H * 0.45; continue
                if kind == 'math':
                    safe = _sanitise_mathtext(txt)
                    try:
                        p_ax.text(0.06, p_y, f'${safe}$', fontsize=MATH_FS, color='#003388',
                                  va='top', ha='left', usetex=False)
                    except Exception:
                        p_ax.text(0.06, p_y, txt, fontsize=MATH_FS - 2, color='#444444',
                                  va='top', ha='left', fontfamily='monospace', usetex=False)
                    p_y -= MATH_H
                else:
                    display = re.sub(r'\*\*?([^*]+)\*\*?', r'\1', txt)
                    display = re.sub(r'__?([^_]+)__?', r'\1', display)
                    is_heading = bool(re.match(r'Step\s+\d+', txt.strip()) or
                                      re.match(r'Problem\s+\d+', txt.strip()))
                    wrapped = textwrap.wrap(display, width=WRAP_WIDTH) or [display]
                    p_ax.text(0.03, p_y, '\n'.join(wrapped), fontsize=PROSE_FS,
                              color='#1a1a2e' if is_heading else '#222222',
                              fontweight='bold' if is_heading else 'normal',
                              va='top', ha='left', usetex=False)
                    p_y -= LINE_H * len(wrapped)
            try:
                p_fig.tight_layout(pad=0.3)
            except Exception:
                pass
            p_path = os.path.join(PLOT_FOLDER, f'response_{msg_id}_p{pi}.png')
            p_fig.savefig(p_path, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
            plt.close('all')
            render_response_png._extra_pages.append(p_path)

        rows = pages[0]
        total_h = max(8.0, sum(
            MATH_H if k == 'math'
            else LINE_H * max(1, len(textwrap.wrap(t, width=WRAP_WIDTH)) if t.strip() else 1)
            for k, t in rows
        ) + TITLE_H + PAD)

    fig, ax = plt.subplots(figsize=(FIG_W, total_h), facecolor='white')
    ax.set_facecolor('white'); ax.axis('off')
    ax.set_xlim(0, 1); ax.set_ylim(0, total_h)

    y = total_h - 0.15
    ax.text(0.5, y, title, fontsize=18, fontweight='bold', color='#1a1a2e',
            ha='center', va='top', usetex=False)
    y -= TITLE_H
    ax.plot([0.02, 0.98], [y + 0.10, y + 0.10], color='#cccccc', linewidth=1.0)

    INDENT = 0.03; MATH_INDENT = 0.06
    for kind, txt in rows:
        if not txt.strip():
            y -= LINE_H * 0.45; continue
        if kind == 'math':
            safe = _sanitise_mathtext(txt)
            try:
                ax.text(MATH_INDENT, y, f'${safe}$', fontsize=MATH_FS, color='#003388',
                        va='top', ha='left', usetex=False)
            except Exception:
                ax.text(MATH_INDENT, y, txt, fontsize=MATH_FS - 2, color='#444444',
                        va='top', ha='left', fontfamily='monospace', usetex=False)
            y -= MATH_H
        else:
            display = re.sub(r'\*\*?([^*]+)\*\*?', r'\1', txt)
            display = re.sub(r'__?([^_]+)__?', r'\1', display)
            is_heading = bool(re.match(r'\*\*', txt) or
                              re.match(r'Step\s+\d+', txt.strip()) or
                              re.match(r'Problem\s+\d+', txt.strip()) or
                              re.match(r'Why it', txt.strip()) or
                              re.match(r'Study Tip', txt.strip()))
            wrapped = textwrap.wrap(display, width=WRAP_WIDTH) if display.strip() else [display]
            if not wrapped:
                wrapped = [display]
            ax.text(INDENT, y, '\n'.join(wrapped), fontsize=PROSE_FS,
                    color='#1a1a2e' if is_heading else '#222222',
                    fontweight='bold' if is_heading else 'normal',
                    va='top', ha='left', usetex=False)
            y -= LINE_H * len(wrapped)

    try:
        fig.tight_layout(pad=0.3)
    except Exception:
        pass

    path = os.path.join(PLOT_FOLDER, f'response_{msg_id}.png')
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close('all')
    return path

# ══════════════════════════════════════════════════════════════════════════════
# SEND LLM RESPONSE
# ══════════════════════════════════════════════════════════════════════════════
async def send_llm_response(update: Update, response_text: str,
                             title: str, msg_id: int,
                             force_image: bool = False) -> None:
    if force_image or _is_math_title(title):
        png = render_response_png(response_text, title, msg_id)
        if png and os.path.exists(png):
            success = await _safe_reply_photo(update, png, title)
            if success:
                for extra in getattr(render_response_png, '_extra_pages', []):
                    if os.path.exists(extra):
                        await _safe_reply_photo(update, extra, f"{title} (cont.)")
                return
            print(f"[send_llm_response] photo failed, falling back to text")

    for i in range(0, len(response_text), 4096):
        await _safe_reply(update, response_text[i:i + 4096])

# ══════════════════════════════════════════════════════════════════════════════
# LAPLACE STEP BUILDER
# ══════════════════════════════════════════════════════════════════════════════
def _identify_laplace_rule_latex(expr: sp.Expr) -> str:
    s = str(expr)
    if "DiracDelta" in s:
        return r"\mathcal{L}\{\delta(t)\} = 1"
    if "Heaviside" in s and "exp" not in s and "sin" not in s and "cos" not in s:
        return r"\mathcal{L}\{u(t)\} = \frac{1}{s}"
    if "exp" in s and "sin" not in s and "cos" not in s:
        return r"\mathcal{L}\{e^{-at}f(t)\} = F(s+a)"
    if "sin" in s:
        return r"\mathcal{L}\{\sin(\omega t)\,u(t)\} = \frac{\omega}{s^2+\omega^2}"
    if "cos" in s:
        return r"\mathcal{L}\{\cos(\omega t)\,u(t)\} = \frac{s}{s^2+\omega^2}"
    if str(expr) == str(t_sym):
        return r"\mathcal{L}\{t\,u(t)\} = \frac{1}{s^2}"
    if expr == sp.Integer(1):
        return r"\mathcal{L}\{\delta(t)\} = 1"
    return r"\text{General transform pair}"

def _build_laplace_steps(expr_str: str) -> tuple[list[tuple[str, str]], str]:
    steps: list[tuple[str, str]] = []
    try:
        f      = parse_ct_expr(expr_str)
        f_tex  = _sympy_to_latex(f)
        steps.append(("Input",      rf"f(t) = {f_tex}"))
        steps.append(("Definition", r"F(s) = \int_{0}^{\infty} f(t)\,e^{-st}\,dt"))
        steps.append(("Rule / Form", _identify_laplace_rule_latex(f)))

        args = sp.Add.make_args(f)
        if len(args) > 1:
            partial = []
            for term in args:
                try:
                    r = sp.laplace_transform(term, t_sym, s_sym, noconds=True)
                    partial.append(
                        rf"\mathcal{{L}}\{{{_sympy_to_latex(term)}\}} = {_sympy_to_latex(r)}"
                    )
                except Exception:
                    pass
            if partial:
                steps.append(("Linearity", r"\quad+\quad".join(partial)))

        result = sp.laplace_transform(f, t_sym, s_sym, noconds=True)
        result = sp.simplify(result)
        steps.append(("Result", rf"F(s) = {_sympy_to_latex(result)}"))
        return steps, ""
    except Exception as e:
        return [], f"❌ Could not compute Laplace transform: {e}"

# ══════════════════════════════════════════════════════════════════════════════
# INVERSE LAPLACE STEP BUILDER
# ══════════════════════════════════════════════════════════════════════════════
def _identify_inv_laplace_rule_latex(expr: sp.Expr) -> str:
    s = str(expr)
    if "s + " in s or "s+" in s or "(s+" in s or "(s +" in s:
        return r"\mathcal{L}^{-1}\left\{\frac{1}{s+a}\right\} = e^{-at}u(t)"
    if re.search(r's\*\*2', s) and "+" in s:
        return (r"\mathcal{L}^{-1}\left\{\frac{\omega}{s^2+\omega^2}\right\} = \sin(\omega t)u(t)"
                r"\quad\text{or}\quad"
                r"\mathcal{L}^{-1}\left\{\frac{s}{s^2+\omega^2}\right\} = \cos(\omega t)u(t)")
    if s.strip() in ("1/s", "s**(-1)"):
        return r"\mathcal{L}^{-1}\left\{\frac{1}{s}\right\} = u(t)"
    if s.strip() == "1":
        return r"\mathcal{L}^{-1}\{1\} = \delta(t)"
    return r"\text{Partial fractions / Bromwich integral}"

def _build_inv_laplace_steps(expr_str: str) -> tuple[list[tuple[str, str]], str]:
    steps: list[tuple[str, str]] = []
    try:
        F = sp.sympify(_normalise(expr_str), locals={
            **_COMMON_NS, "s": s_sym, "t": t_sym
        })
        F_tex = _sympy_to_latex(F)

        steps.append(("Input",      rf"F(s) = {F_tex}"))
        steps.append(("Definition",
                       r"f(t) = \mathcal{L}^{-1}\{F(s)\} = "
                       r"\frac{1}{2\pi j}\int_{c-j\infty}^{c+j\infty} F(s)\,e^{st}\,ds"))
        steps.append(("Rule / Form", _identify_inv_laplace_rule_latex(F)))

        try:
            pf = sp.apart(F, s_sym)
            if pf != F:
                steps.append(("Partial Fractions",
                               rf"F(s) = {_sympy_to_latex(pf)}"))
        except Exception:
            pass

        result = sp.inverse_laplace_transform(F, s_sym, t_sym)
        result = sp.simplify(result)

        result_tex = _sympy_to_latex(result)
        result_tex = result_tex.replace(r"\theta\left(t\right)", r"u(t)")
        result_tex = re.sub(r'\\theta\(t\)', r'u(t)', result_tex)

        steps.append(("Result", rf"f(t) = {result_tex}"))
        steps.append(("Note",
                       r"\text{Valid for } t \geq 0 \text{ (causal / right-sided signal)}"))
        return steps, ""

    except Exception as e:
        return [], f"❌ Could not compute Inverse Laplace transform: {e}"

def compute_inv_laplace(expr_str: str) -> str:
    lines = ["━━━ 📐 INVERSE LAPLACE TRANSFORM ━━━\n"]
    lines.append(f"Input:  F(s) = {expr_str}\n")
    lines.append(r"Definition:  f(t) = (1/2πj) ∫ F(s)·e^(st) ds" + "\n")
    try:
        F = sp.sympify(_normalise(expr_str), locals={**_COMMON_NS, "s": s_sym})
    except Exception as e:
        return f"❌ Could not parse expression: {e}"
    try:
        pf = sp.apart(F, s_sym)
        if pf != F:
            lines.append(f"Partial fractions:\n   F(s) = {sp.pretty(pf)}\n")
    except Exception:
        pass
    try:
        result = sp.inverse_laplace_transform(F, s_sym, t_sym)
        result = sp.simplify(result)
        lines.append(f"Result:\n   f(t) = {sp.pretty(result)}\n")
        lines.append(f"\n✅  f(t) = {sp.pretty(result)}")
    except Exception as e:
        lines.append(f"❌ SymPy could not find a closed form: {e}")
    return "\n".join(lines)

# ══════════════════════════════════════════════════════════════════════════════
# FOURIER STEP BUILDER
# ══════════════════════════════════════════════════════════════════════════════
def _identify_fourier_rule_latex(expr: sp.Expr) -> str:
    s = str(expr)
    if "DiracDelta" in s:
        return r"\mathcal{F}\{\delta(t)\} = 1"
    if "Heaviside" in s and "exp" not in s:
        return r"\mathcal{F}\{u(t)\} = \pi \delta(\omega) + \frac{1}{j\omega}"
    if "exp" in s and "sin" not in s and "cos" not in s:
        return r"\mathcal{F}\{e^{-at}u(t)\} = \frac{1}{a+j\omega},\quad a>0"
    if "sin" in s:
        return r"\mathcal{F}\{\sin(\omega_0 t)\} = j\pi [\delta(\omega+\omega_0)-\delta(\omega-\omega_0)]"
    if "cos" in s:
        return r"\mathcal{F}\{\cos(\omega_0 t)\} = \pi [\delta(\omega+\omega_0)+\delta(\omega-\omega_0)]"
    if expr == sp.Integer(1):
        return r"\mathcal{F}\{1\} = 2\pi \delta(\omega)"
    if "Piecewise" in str(expr):
        return r"\mathcal{F}\{\mathrm{rect}(t/\tau)\} = \tau \,\mathrm{sinc}(\omega \tau /2)"
    return r"F(\omega) = \int_{-\infty}^{\infty} f(t)\,e^{-j\omega t}\,dt"

def _fourier_direct(f: sp.Expr) -> sp.Expr:
    try:
        result = sp.integrate(f * sp.exp(-sp.I * w_sym * t_sym), (t_sym, -sp.oo, sp.oo))
        if not result.has(sp.Integral):
            return sp.simplify(result)
    except Exception:
        pass

    s = str(f)
    if "cos" in s:
        m = re.search(r'cos\(\s*([^)]+?)\s*\*?\s*t\b', str(f))
        if m:
            w0 = sp.sympify(m.group(1).strip(), locals=_COMMON_NS)
            return sp.pi * (sp.DiracDelta(w_sym - w0) + sp.DiracDelta(w_sym + w0))
    if "sin" in s:
        m = re.search(r'sin\(\s*([^)]+?)\s*\*?\s*t\b', str(f))
        if m:
            w0 = sp.sympify(m.group(1).strip(), locals=_COMMON_NS)
            return sp.I * sp.pi * (sp.DiracDelta(w_sym + w0) - sp.DiracDelta(w_sym - w0))
    if "Heaviside" in s and "exp" not in s:
        coeff = sp.Integer(1)
        if f.is_Mul:
            nums = [a for a in f.args if a.is_number]
            if nums:
                coeff = sp.Mul(*nums)
        return sp.simplify(coeff * (sp.pi * sp.DiracDelta(w_sym) + 1 / (sp.I * w_sym)))
    if f == sp.Integer(1):
        return 2 * sp.pi * sp.DiracDelta(w_sym)
    raise ValueError("No closed-form pair found")

def _build_fourier_steps(expr_str: str) -> tuple[list[tuple[str, str]], str]:
    steps: list[tuple[str, str]] = []
    try:
        f      = parse_ct_expr(expr_str)
        f_tex  = _sympy_to_latex(f)
        steps.append(("Input",      rf"f(t) = {f_tex}"))
        steps.append(("Definition", r"F(\omega) = \int_{-\infty}^{\infty} f(t)\, e^{-j\omega t}\, dt"))
        steps.append(("Known pair", _identify_fourier_rule_latex(f)))

        result     = _fourier_direct(f)
        result_tex = _sympy_to_latex(result)
        result_tex = re.sub(r'(?<![a-zA-Z])i(?![a-zA-Z])', 'j', result_tex)
        steps.append(("Result", rf"F(\omega) = {result_tex}"))
        return steps, ""
    except Exception as e:
        return [], f"❌ Could not compute Fourier transform: {e}"

# ══════════════════════════════════════════════════════════════════════════════
# INVERSE FOURIER STEP BUILDER
# ══════════════════════════════════════════════════════════════════════════════
def _identify_inv_fourier_rule_latex(expr: sp.Expr) -> str:
    s = str(expr)
    if "DiracDelta" in s:
        return r"\mathcal{F}^{-1}\{\delta(\omega)\} = \frac{1}{2\pi}"
    if "omega" in s and ("a +" in s or "a+" in s or "j*omega" in s or "j\omega" in s):
        return r"\mathcal{F}^{-1}\left\{\frac{1}{a+j\omega}\right\} = e^{-at}u(t),\quad a>0"
    if "pi" in s and "DiracDelta" in s:
        return r"\mathcal{F}^{-1}\{\pi\delta(\omega\pm\omega_0)\} \to \cos/\sin \text{ terms}"
    return r"f(t) = \frac{1}{2\pi}\int_{-\infty}^{\infty} F(\omega)\,e^{j\omega t}\,d\omega"

def _build_inv_fourier_steps(expr_str: str) -> tuple[list[tuple[str, str]], str]:
    steps: list[tuple[str, str]] = []
    try:
        F = sp.sympify(_normalise(expr_str), locals={
            **_COMMON_NS, "omega": w_sym, "w": w_sym
        })
        F_tex = _sympy_to_latex(F)

        steps.append(("Input",      rf"F(\omega) = {F_tex}"))
        steps.append(("Definition",
                       r"f(t) = \frac{1}{2\pi}\int_{-\infty}^{\infty} "
                       r"F(\omega)\,e^{j\omega t}\,d\omega"))
        steps.append(("Rule / Form", _identify_inv_fourier_rule_latex(F)))

        try:
            pf = sp.apart(F, w_sym)
            if pf != F:
                steps.append(("Partial Fractions",
                               rf"F(\omega) = {_sympy_to_latex(pf)}"))
        except Exception:
            pass

        result = sp.integrate(
            F * sp.exp(sp.I * w_sym * t_sym) / (2 * sp.pi),
            (w_sym, -sp.oo, sp.oo)
        )
        if result.has(sp.Integral):
            result = sp.inverse_fourier_transform(
                F.subs(w_sym, 2 * sp.pi * sp.Symbol('f')),
                sp.Symbol('f'), t_sym
            )

        result = sp.simplify(result)
        result_tex = _sympy_to_latex(result)
        result_tex = re.sub(r'(?<![a-zA-Z])i(?![a-zA-Z])', 'j', result_tex)

        steps.append(("Result", rf"f(t) = {result_tex}"))
        return steps, ""

    except Exception as e:
        return [], f"❌ Could not compute Inverse Fourier transform: {e}"

def compute_inv_fourier(expr_str: str) -> str:
    lines = ["━━━ 📡 INVERSE FOURIER TRANSFORM ━━━\n"]
    lines.append(f"Input:  F(ω) = {expr_str}\n")
    lines.append("Definition:  f(t) = (1/2π) ∫ F(ω)·e^(jωt) dω\n")
    try:
        F = sp.sympify(_normalise(expr_str), locals={**_COMMON_NS, "omega": w_sym})
    except Exception as e:
        return f"❌ Could not parse expression: {e}"
    try:
        result = sp.integrate(
            F * sp.exp(sp.I * w_sym * t_sym) / (2 * sp.pi),
            (w_sym, -sp.oo, sp.oo)
        )
        if not result.has(sp.Integral):
            result = sp.simplify(result)
            lines.append(f"Result:\n   f(t) = {sp.pretty(result)}\n")
            lines.append(f"\n✅  f(t) = {sp.pretty(result)}")
        else:
            lines.append("❌ Integral could not be evaluated in closed form.")
    except Exception as e:
        lines.append(f"❌ Could not evaluate: {e}")
    return "\n".join(lines)

# ══════════════════════════════════════════════════════════════════════════════
# PERIODIC SUMMATION FOURIER STEP BUILDER
# ══════════════════════════════════════════════════════════════════════════════
def _build_periodic_fourier_steps(g_expr_str: str, period: float = 2.0
                                   ) -> tuple[list[tuple[str, str]], str]:
    steps: list[tuple[str, str]] = []
    try:
        g      = parse_ct_expr(g_expr_str)
        g_tex  = _sympy_to_latex(g)
        T_sym  = sp.Rational(period).limit_denominator(1000)
        w0     = 2 * sp.pi / T_sym
        w0_tex = _sympy_to_latex(w0)

        steps.append(("Signal",
                       rf"x(t) = \sum_{{k=-\infty}}^{{\infty}} g(t - {_sympy_to_latex(T_sym)}k)"
                       rf",\quad g(t) = {g_tex}"))
        steps.append(("Property",
                       rf"x(t)=\sum_k g(t-kT) \;\xrightarrow{{\mathcal{{F}}}}\;"
                       rf"X(\omega)=\omega_0\sum_{{n=-\infty}}^{{\infty}}"
                       rf"G(n\omega_0)\,\delta(\omega-n\omega_0)"))
        steps.append(("Fund. freq.",
                       rf"\omega_0 = \frac{{2\pi}}{{T}} = \frac{{2\pi}}{{{_sympy_to_latex(T_sym)}}} "
                       rf"= {w0_tex}\ \mathrm{{rad/s}}"))
        steps.append(("Definition",
                       r"G(\omega) = \int_{-\infty}^{\infty} g(t)\,e^{-j\omega t}\,dt"))
        try:
            G_result = _fourier_direct(g)
            G_tex    = _sympy_to_latex(G_result)
            G_tex    = re.sub(r'(?<![a-zA-Z\\])i(?![a-zA-Z0-9{])', 'j', G_tex)
            steps.append(("G(ω)", rf"G(\omega) = {G_tex}"))
        except Exception:
            steps.append(("G(ω)",
                           rf"G(\omega) = \int_{{-\infty}}^{{\infty}} {g_tex}\,e^{{-j\omega t}}\,dt"))
        steps.append(("Substitute",
                       rf"X(\omega) = {w0_tex}\sum_{{n=-\infty}}^{{\infty}}"
                       rf"G(n\cdot {w0_tex})\,\delta(\omega - n\cdot {w0_tex})"))
        steps.append(("Result",
                       rf"X(\omega) = \omega_0\sum_{{n=-\infty}}^{{\infty}}"
                       rf"G(n\omega_0)\,\delta(\omega-n\omega_0)"))
        return steps, ""
    except Exception as e:
        return [], f"❌ Could not compute periodic Fourier transform: {e}"

# ══════════════════════════════════════════════════════════════════════════════
# FOURIER SERIES STEP BUILDER
# ══════════════════════════════════════════════════════════════════════════════
def _build_fourier_series_steps(expr_str: str, period: float,
                                 n_terms: int = 5) -> tuple[list[tuple[str, str]], str]:
    steps: list[tuple[str, str]] = []
    try:
        f     = parse_ct_expr(expr_str)
        f_tex = _sympy_to_latex(f)
        T     = sp.Rational(period).limit_denominator(1000)
        w0    = 2 * sp.pi / T
        w0_tex = _sympy_to_latex(w0)

        steps.append(("Input",  rf"f(t) = {f_tex},\quad T = {_sympy_to_latex(T)}"))
        steps.append(("Fund. freq.", rf"\omega_0 = \frac{{2\pi}}{{T}} = {w0_tex}\ \mathrm{{rad/s}}"))
        steps.append(("Coefficients",
                       r"a_0=\frac{1}{T}\int_0^T f(t)\,dt,\quad"
                       r"a_n=\frac{2}{T}\int_0^T f(t)\cos(n\omega_0 t)\,dt,\quad"
                       r"b_n=\frac{2}{T}\int_0^T f(t)\sin(n\omega_0 t)\,dt"))

        a0 = sp.simplify(sp.integrate(f, (t_sym, 0, T)) / T)
        steps.append(("a0 (DC)", rf"a_0 = {_sympy_to_latex(a0)}"))

        for k in range(1, n_terms + 1):
            try:
                an = sp.simplify(2 * sp.integrate(f * sp.cos(k * w0 * t_sym), (t_sym, 0, T)) / T)
                bn = sp.simplify(2 * sp.integrate(f * sp.sin(k * w0 * t_sym), (t_sym, 0, T)) / T)
                steps.append((f"n = {k}",
                               rf"a_{{{k}}}={_sympy_to_latex(an)},\quad b_{{{k}}}={_sympy_to_latex(bn)}"))
            except Exception:
                steps.append((f"n = {k}", r"\text{could not evaluate}"))

        try:
            series = sp.fourier_series(f, (t_sym, 0, T))
            trunc  = series.truncate(n_terms)
            steps.append(("Truncated series",
                           rf"f(t)\approx {_sympy_to_latex(trunc)}"))
        except Exception:
            pass

        return steps, ""
    except Exception as e:
        return [], f"❌ Could not compute Fourier series: {e}"

def compute_fourier_series(expr_str: str, period: float, n_terms: int = 5) -> str:
    lines = ["━━━ 🎵 FOURIER SERIES ━━━\n"]
    lines.append(f"Input: f(t) = {expr_str},  T = {period:.4g}\n")
    try:
        f = parse_ct_expr(expr_str)
    except Exception as e:
        return f"❌ Could not parse expression: {e}"
    T  = sp.Rational(period).limit_denominator(1000)
    w0 = 2 * sp.pi / T
    try:
        a0 = sp.simplify(sp.integrate(f, (t_sym, 0, T)) / T)
        lines.append(f"a₀ = {sp.pretty(a0)}")
    except Exception:
        pass
    try:
        series = sp.fourier_series(f, (t_sym, 0, T))
        trunc  = series.truncate(n_terms)
        lines.append(f"\nf(t) ≈ {sp.pretty(trunc)}")
        lines.append("\n✅  Series computed successfully.")
    except Exception as e:
        lines.append(f"\n⚠️  {e}")
    return "\n".join(lines)

def compute_fourier(expr_str: str) -> str:
    lines = ["━━━ 📡 FOURIER TRANSFORM ━━━\n"]
    lines.append(f"Input:  f(t) = {expr_str}\n")
    try:
        f = parse_ct_expr(expr_str)
        result = _fourier_direct(f)
        lines.append(f"✅  F(ω) = {sp.pretty(result)}")
    except Exception as e:
        lines.append(f"❌ {e}")
    return "\n".join(lines)

def compute_laplace(expr_str: str) -> str:
    lines = ["━━━ 📐 LAPLACE TRANSFORM ━━━\n"]
    lines.append(f"Input:  f(t) = {expr_str}\n")
    try:
        f      = parse_ct_expr(expr_str)
        result = sp.laplace_transform(f, t_sym, s_sym, noconds=True)
        result = sp.simplify(result)
        lines.append(f"✅  F(s) = {sp.pretty(result)}")
    except Exception as e:
        lines.append(f"❌ {e}")
    return "\n".join(lines)

# ══════════════════════════════════════════════════════════════════════════════
# CONVOLUTION
# ══════════════════════════════════════════════════════════════════════════════
def _simplify_heaviside_powers(expr: sp.Expr) -> sp.Expr:
    return expr.replace(
        lambda e: e.is_Pow and isinstance(e.base, sp.Heaviside) and e.exp.is_positive,
        lambda e: e.base
    )

def _extract_heaviside_onset(expr: sp.Expr) -> sp.Expr | None:
    for arg in sp.preorder_traversal(expr):
        if isinstance(arg, sp.Heaviside):
            inner = arg.args[0]
            sol = sp.solve(inner, t_sym)
            if sol:
                return sol[0]
            if inner == t_sym:
                return sp.Integer(0)
    return None

def _compute_causal_limits(f: sp.Expr, g: sp.Expr):
    d_f = _extract_heaviside_onset(f)
    d_g = _extract_heaviside_onset(g)
    if d_f is None or d_g is None:
        return None
    return d_f, t_sym - d_g

def _lambdify_ct(expr: sp.Expr):
    return sp.lambdify(
        t_sym, expr,
        modules=["numpy", {
            "Heaviside":  lambda x: np.where(np.asarray(x, float) >= 0, 1., 0.),
            "DiracDelta": lambda x: np.zeros_like(np.asarray(x, float)),
            "rect":       lambda x: np.where(np.abs(x) < 0.5, 1.0,
                                    np.where(np.abs(x) == 0.5, 0.5, 0.0)),
        }]
    )

def _numerical_convolution_plot(f_expr: sp.Expr, g_expr: sp.Expr, msg_id: int) -> str | None:
    try:
        t_vals = np.linspace(-2, 20, 5000)
        dt     = t_vals[1] - t_vals[0]
        f_vals = np.real(_lambdify_ct(f_expr)(t_vals)).astype(float)
        g_vals = np.real(_lambdify_ct(g_expr)(t_vals)).astype(float)
        conv   = np.convolve(f_vals, g_vals, mode="full") * dt
        t_full = t_vals[0] + np.arange(len(conv)) * dt
        mask   = (t_full >= -2) & (t_full <= 20)
        fig, axes = plt.subplots(3, 1, figsize=(9, 7))
        axes[0].plot(t_vals, f_vals, color="steelblue")
        axes[0].set(title="f(t)", xlabel="t"); axes[0].grid(True, alpha=.3)
        axes[1].plot(t_vals, g_vals, color="darkorange")
        axes[1].set(title="g(t)", xlabel="t"); axes[1].grid(True, alpha=.3)
        axes[2].plot(t_full[mask], conv[mask], color="green")
        axes[2].set(title="(f ★ g)(t)  [numerical]", xlabel="t")
        axes[2].grid(True, alpha=.3)
        fig.tight_layout()
        path = os.path.join(PLOT_FOLDER, f"conv_{msg_id}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close("all")
        return path
    except Exception as e:
        print(f"[numerical_conv] {e}")
        return None

def _build_convolution_steps(expr1_str: str, expr2_str: str,
                              msg_id: int) -> tuple[list[tuple[str, str]], str, str | None]:
    steps: list[tuple[str, str]] = []
    try:
        f = parse_ct_expr(expr1_str)
        g = parse_ct_expr(expr2_str)
    except Exception as e:
        return [], f"❌ Could not parse signals: {e}", None

    steps.append(("f(t)",       _sympy_to_latex(f)))
    steps.append(("g(t)",       _sympy_to_latex(g)))
    steps.append(("Definition", r"(f\star g)(t)=\int_{-\infty}^{\infty}f(\tau)\,g(t-\tau)\,d\tau"))

    f_tau     = f.subs(t_sym, tau)
    g_shift   = g.subs(t_sym, t_sym - tau)
    integrand = sp.expand(f_tau * g_shift)
    steps.append(("Substitution",
                   rf"f(\tau)={_sympy_to_latex(f_tau)},\quad g(t-\tau)={_sympy_to_latex(g_shift)}"))
    steps.append(("Integrand",   _sympy_to_latex(integrand)))

    causal_limits = _compute_causal_limits(f, g)
    limits = (tau, causal_limits[0], causal_limits[1]) if causal_limits else (tau, -sp.oo, sp.oo)

    plot_path = None
    try:
        result = sp.integrate(integrand, limits)
        result = sp.simplify(result)
        result = _simplify_heaviside_powers(result)
        if result.has(sp.Integral):
            raise ValueError("unevaluated integral")
        steps.append(("Result", rf"(f\star g)(t) = {_sympy_to_latex(result)}"))
        plot_path = _numerical_convolution_plot(f, g, msg_id)
    except Exception as e:
        steps.append(("Note",     rf"\text{{Symbolic integration failed: {str(e)[:60]}}}"))
        steps.append(("Fallback", r"\text{See numerical plot}"))
        plot_path = _numerical_convolution_plot(f, g, msg_id)

    return steps, "", plot_path

def _parse_two_signals(text: str):
    m = re.split(r'\bwith\b|\band\b|\*|\bstar\b', text, maxsplit=1, flags=re.IGNORECASE)
    if len(m) == 2:
        e1 = re.sub(r'^.*?(?:convolve|convolution\s+of|f\s*=|f\(t\)\s*=)\s*',
                    '', m[0], flags=re.IGNORECASE).strip()
        e2 = re.sub(r'^.*?(?:g\s*=|g\(t\)\s*=)\s*', '', m[1], flags=re.IGNORECASE).strip()
        e1 = extract_expr(e1) or e1
        e2 = extract_expr(e2) or e2
        return e1, e2
    return None, None

def compute_convolution(expr1_str: str, expr2_str: str, msg_id: int = 0):
    lines = ["━━━ 🔁 CONVOLUTION ━━━\n"]
    lines.append(f"f(t) = {expr1_str}\ng(t) = {expr2_str}\n")
    try:
        f = parse_ct_expr(expr1_str)
        g = parse_ct_expr(expr2_str)
    except Exception as e:
        return f"❌ {e}", None
    f_tau     = f.subs(t_sym, tau)
    g_shift   = g.subs(t_sym, t_sym - tau)
    integrand = sp.expand(f_tau * g_shift)
    causal_limits = _compute_causal_limits(f, g)
    limits = (tau, causal_limits[0], causal_limits[1]) if causal_limits else (tau, -sp.oo, sp.oo)
    plot_path = None
    try:
        result = sp.integrate(integrand, limits)
        result = sp.simplify(result)
        result = _simplify_heaviside_powers(result)
        if result.has(sp.Integral):
            raise ValueError("unevaluated")
        lines.append(f"✅  (f ★ g)(t) = {sp.pretty(result)}")
        plot_path = _numerical_convolution_plot(f, g, msg_id)
    except Exception as e:
        lines.append(f"❌ Symbolic failed: {e}")
        plot_path = _numerical_convolution_plot(f, g, msg_id)
    return "\n".join(lines), plot_path

# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL PLOTTER
# ══════════════════════════════════════════════════════════════════════════════
PLOT_KEYWORDS     = ["plot", "draw", "graph", "sketch", "show me",
                     "visualise", "visualize", "diagram"]
DISCRETE_KEYWORDS = ["x[n]", "u[n]", "delta[n]", "δ[n]", "h[n]", "y[n]", "[n]"]

def is_discrete(question: str) -> bool:
    return any(kw in question for kw in DISCRETE_KEYWORDS)

def plot_ct(expr_str: str, msg_id: int) -> str | None:
    try:
        expr   = parse_ct_expr(expr_str)
        t_vals = np.linspace(-4, 8, 4000)
        y_vals = np.real(_lambdify_ct(expr)(t_vals)).astype(float)
        y_vals = np.clip(y_vals, -10, 10)
    except Exception as e:
        print(f"[plot_ct] {e}"); return None
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(t_vals, y_vals, color="steelblue", lw=2)
    ax.axhline(0, color="k", lw=.6)
    ax.axvline(0, color="k", lw=.6, ls="--", alpha=.5)
    ax.set(xlabel="t", ylabel="x(t)", title=f"x(t) = {expr_str}")
    ax.grid(True, alpha=.4)
    fig.tight_layout()
    path = os.path.join(PLOT_FOLDER, f"plot_{msg_id}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    return path

def plot_dt(expr_str: str, msg_id: int) -> str | None:
    try:
        evaluator = parse_dt_expr(expr_str)
        n_vals    = np.arange(-10, 21)
        y_vals    = np.clip(evaluator(n_vals), -10, 10)
    except Exception as e:
        print(f"[plot_dt] {e}"); return None
    fig, ax = plt.subplots(figsize=(10, 3))
    ml, sl, _ = ax.stem(n_vals, y_vals, linefmt="steelblue", markerfmt="o", basefmt="k-")
    ml.set_markersize(5); sl.set_linewidth(1.5)
    ax.axhline(0, color="k", lw=.6)
    ax.set(xlabel="n", ylabel="x[n]", title=f"x[n] = {expr_str}")
    ax.grid(True, alpha=.3); ax.set_xticks(n_vals[::2])
    fig.tight_layout()
    path = os.path.join(PLOT_FOLDER, f"plot_{msg_id}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    return path

def plot_dirac_arrow(t0: float, msg_id: int) -> str:
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.axhline(0, color="k", lw=.8)
    ax.annotate("", xy=(t0, 1), xytext=(t0, 0),
                arrowprops=dict(arrowstyle="-|>", color="steelblue", lw=2.5))
    ax.set_xlim(t0 - 3, t0 + 3); ax.set_ylim(-0.2, 1.5)
    label = f"δ(t − {t0})" if t0 != 0 else "δ(t)"
    ax.set(title=label, xlabel="t", ylabel="δ(t)"); ax.grid(True, alpha=.4)
    fig.tight_layout()
    path = os.path.join(PLOT_FOLDER, f"plot_{msg_id}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    return path

def generate_plot(question: str, msg_id: int) -> str | None:
    q = question.lower()
    if is_discrete(question):
        expr_str = extract_expr(question)
        if expr_str:
            return plot_dt(expr_str, msg_id)
    if "dirac" in q or ("delta" in q and "[n]" not in q):
        m  = re.search(r'(?:delta|δ)\s*\(\s*t\s*([+-]\s*\d*\.?\d+)?\s*\)', question)
        t0 = float(m.group(1).replace(" ", "")) if (m and m.group(1)) else 0.0
        return plot_dirac_arrow(t0, msg_id)
    expr_str = extract_expr(question)
    return plot_ct(expr_str, msg_id) if expr_str else None

# ══════════════════════════════════════════════════════════════════════════════
# KEYWORD ROUTERS
# ══════════════════════════════════════════════════════════════════════════════
LAPLACE_KEYS     = ["laplace", "l transform", "l{", "laplace transform", "laplace of"]
INV_LAPLACE_KEYS = ["inverse laplace", "inv laplace", "ilt", "ilaplace",
                    "inverse l transform", "laplace inverse"]
FOURIER_KEYS     = ["fourier transform", "ft{", "fourier of", "f transform",
                    "ft of", "compute ft", "find ft", "f.t. of", "fourier tf"]
INV_FOURIER_KEYS = ["inverse fourier", "inv fourier", "ift", "ifourier",
                    "inverse ft", "inverse f transform", "fourier inverse"]
FS_KEYS          = ["fourier series", "periodic signal", "series of"]
CONV_KEYS        = ["convolution", "convolve", "f*g", "f★g", "f star g"]
PERIODIC_FOURIER_KEYS = ["sum", "summation", "periodic summation",
                          "x(t) =", "x(t)=", "k=-inf", "g(t-2k)", "g(t -"]

def is_inv_laplace(q: str) -> bool:  return any(k in q for k in INV_LAPLACE_KEYS)
def is_laplace(q: str) -> bool:
    if is_inv_laplace(q): return False
    return any(k in q for k in LAPLACE_KEYS)
def is_inv_fourier(q: str) -> bool:  return any(k in q for k in INV_FOURIER_KEYS)
def is_fourier(q: str) -> bool:
    if is_inv_fourier(q): return False
    return any(k in q for k in FOURIER_KEYS)
def is_fs(q: str)   -> bool:  return any(k in q for k in FS_KEYS)
def is_conv(q: str) -> bool:  return any(k in q for k in CONV_KEYS)
def is_plot(q: str) -> bool:  return any(k in q for k in PLOT_KEYWORDS)
def is_periodic_fourier(q: str) -> bool:
    return any(k in q for k in FOURIER_KEYS) and any(k in q for k in PERIODIC_FOURIER_KEYS)

# ══════════════════════════════════════════════════════════════════════════════
# MARK CALCULATOR
# ══════════════════════════════════════════════════════════════════════════════
WEIGHTS      = {"tutorials": .10, "labs": .10, "tests": .20, "exam": .60}
PASS_MARK    = 50.0
TRIGGER_KEYS = ["pass", "exam mark", "calculate marks", "how much do i need",
                "calculate mark breakdown", "what do i need", "minimum", "final mark"]
STEPS_MC     = ["test1", "test2", "labs", "tutorials"]
STEP_PROMPTS = {
    "test1":     "What did you get for *Test 1*? (0–100)",
    "test2":     "What did you get for *Test 2*? (0–100)",
    "labs":      "What is your *Labs average*? (0–100)",
    "tutorials": "What is your *Tutorials average*? (0–100)",
}
mark_sessions: dict = {}

def is_mark_trigger(text: str) -> bool:
    return any(kw in text.lower() for kw in TRIGGER_KEYS)

def compute_result(data: dict) -> str:
    test_avg = (data["test1"] + data["test2"]) / 2
    weighted = (test_avg * WEIGHTS["tests"] +
                data["labs"] * WEIGHTS["labs"] +
                data["tutorials"] * WEIGHTS["tutorials"])
    req_exam = (PASS_MARK - weighted) / WEIGHTS["exam"]
    lines = [
        "📊 *Your Mark Breakdown*\n",
        f"  Test average:       {test_avg:.1f}%",
        f"  Labs average:       {data['labs']:.1f}%",
        f"  Tutorials average:  {data['tutorials']:.1f}%",
        f"\n  *Marks secured so far: {weighted:.1f}% (out of 40%)*",
        f"  *(Exam carries the remaining 60%)*\n",
    ]
    if req_exam <= 0:
        lines.append("✅ You've already secured enough to pass!")
    elif req_exam > 100:
        lines.append(f"❌ You'd need {req_exam:.1f}% — mathematically impossible. Give it your best!")
    else:
        lines.append(f"🎯 *You need at least {req_exam:.1f}% in the exam to pass.*")
        if req_exam <= 50:   lines.append("💪 Very achievable — keep it up!")
        elif req_exam <= 70: lines.append("📚 Tough but doable with a solid plan!")
        else:                lines.append("⚠️ Hard work required — start now!")
    return "\n".join(lines)

async def handle_mark_session(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    chat_id = update.effective_chat.id
    text    = update.message.text.strip()
    if is_mark_trigger(text) and chat_id not in mark_sessions:
        mark_sessions[chat_id] = {"step": "test1", "data": {}}
        await _safe_reply(update,
            "Let's calculate what you need to pass!\n\n" + STEP_PROMPTS["test1"],
            parse_mode="Markdown")
        return True
    if chat_id in mark_sessions:
        session = mark_sessions[chat_id]
        step    = session["step"]
        if text.lower() in ["cancel", "stop", "quit", "exit"]:
            del mark_sessions[chat_id]
            await _safe_reply(update, "❌ Cancelled.")
            return True
        try:
            value = float(text)
            if not 0 <= value <= 100:
                raise ValueError
        except ValueError:
            await _safe_reply(update, "⚠️ Enter a number 0–100, or type *cancel*.",
                              parse_mode="Markdown")
            return True
        session["data"][step] = value
        idx = STEPS_MC.index(step)
        if idx + 1 < len(STEPS_MC):
            nxt = STEPS_MC[idx + 1]
            session["step"] = nxt
            await _safe_reply(update, STEP_PROMPTS[nxt], parse_mode="Markdown")
        else:
            result = compute_result(session["data"])
            del mark_sessions[chat_id]
            await _safe_reply(update, result, parse_mode="Markdown")
        return True
    return False

# ══════════════════════════════════════════════════════════════════════════════
# OCR — Gemini Flash
# ══════════════════════════════════════════════════════════════════════════════
def _ocr_image_bytes(image_bytes: bytes, mime: str) -> str:
    if not image_bytes or len(image_bytes) < 100:
        return f"❌ OCR failed: image too small ({len(image_bytes)} bytes)"
    if not GEMINI_API_KEY:
        return "❌ OCR failed: GEMINI_API_KEY not set."
    mime = mime.lower().lstrip(".")
    if mime == "jpg": mime = "jpeg"
    if mime not in ("jpeg", "png", "webp", "gif"): mime = "jpeg"
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    prompt_text = (
        "You are an expert at reading handwritten academic work and engineering diagrams.\n\n"
        "First determine what the image contains:\n"
        "  A) HANDWRITTEN TEXT / EQUATIONS\n"
        "  B) DRAWN DIAGRAM\n"
        "  C) BOTH\n\n"
        "If A: Transcribe ALL text and equations exactly.\n"
        "If B: Describe the diagram structurally (type, nodes, connections, labels, I/O).\n"
        "If C: Do both — text first, then diagram.\n\n"
        "Output ONLY the transcribed/described content. No commentary."
    )
    payload = {"contents": [{"parts": [
        {"text": prompt_text},
        {"inline_data": {"mime_type": f"image/{mime}", "data": b64}}
    ]}]}
    url = (f"https://generativelanguage.googleapis.com/v1/models"
           f"/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}")
    try:
        resp    = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        content = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        return content
    except Exception as e:
        return f"❌ OCR request failed: {e}"

def _extract_pdf_bytes(pdf_bytes: bytes) -> str:
    import io
    try:
        from pypdf import PdfReader
    except ImportError:
        from PyPDF2 import PdfReader
    parts  = []
    reader = PdfReader(io.BytesIO(pdf_bytes))
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        parts.append(f"--- Page {i} ---\n{text}" if text else
                     f"--- Page {i} --- [image-only page]")
    return "\n\n".join(parts) if parts else "[No text extracted]"

# ══════════════════════════════════════════════════════════════════════════════
# DOCUMENT PARSER
# ══════════════════════════════════════════════════════════════════════════════
def _build_question_index(doc_text: str) -> list[dict]:
    blocks = []
    lines  = doc_text.split('\n')
    spans  = []
    char_pos = 0
    for line in lines:
        stripped = line.strip()
        m_main = re.match(r'^(?:Question|Q\.?)\s*(\d+)\b', stripped, re.IGNORECASE)
        if m_main:
            spans.append((char_pos, m_main.group(1), None))
        m_sub = re.match(r'^\(?([a-zA-Z]{1,2}|[ivxlIVXL]+)\)?[\.\)]\s', stripped)
        if m_sub and spans:
            spans.append((char_pos, spans[-1][1] if spans else None, m_sub.group(1).lower()))
        char_pos += len(line) + 1
    for i, (start, qid, sub) in enumerate(spans):
        end  = spans[i + 1][0] if i + 1 < len(spans) else len(doc_text)
        text = doc_text[start:end].strip()
        blocks.append({'id': str(qid) if qid else None, 'sub': sub,
                       'start': start, 'end': end, 'text': text})
    return blocks

def extract_question_with_context(doc_text: str, instruction: str) -> str:
    m = re.search(r'[Qq](?:uestion)?\s*(\d+)\s*[\.\(]?\s*([a-zA-Z])?', instruction)
    if not m:
        return instruction
    target_q   = m.group(1)
    target_sub = m.group(2).lower() if m.group(2) else None
    blocks     = _build_question_index(doc_text)
    q_blocks   = [b for b in blocks if b['id'] == target_q]
    if not q_blocks:
        return instruction
    preamble_text = "\n\n".join(
        b['text'] for b in q_blocks if b['sub'] is None).strip()
    if target_sub is None:
        full_text = "\n\n".join(b['text'] for b in q_blocks).strip()
        return (f"The student is asking about Question {target_q}.\n\n"
                f"--- Extracted question text ---\n{full_text}\n\n"
                f"--- Student instruction ---\n{instruction}")
    sub_blocks = [b for b in q_blocks if b['sub'] == target_sub]
    if not sub_blocks:
        return (f"The student is asking about Question {target_q}({target_sub}).\n\n"
                f"--- Question {target_q} preamble ---\n{preamble_text or '(none)'}\n\n"
                f"--- Student instruction ---\n{instruction}")
    sub_text = "\n\n".join(b['text'] for b in sub_blocks).strip()
    parts = [f"The student is asking about Question {target_q}({target_sub})."]
    if preamble_text:
        parts.append(f"--- Question {target_q} preamble ---\n{preamble_text}")
    parts.append(f"--- Question {target_q}({target_sub}) text ---\n{sub_text}")
    parts.append(f"--- Student instruction ---\n{instruction}")
    return "\n\n".join(parts)

# ══════════════════════════════════════════════════════════════════════════════
# AUTO-ROUTE OCR OUTPUT
# ══════════════════════════════════════════════════════════════════════════════
def auto_route_extracted_text(extracted: str) -> str | None:
    lower = extracted.lower()
    if re.search(r'sum.*g\s*\(t', lower) and "fourier" in lower:
        g_def = re.search(r'g\s*\(\s*t\s*\)\s*=\s*([^\n,]+)', lower)
        T_def = re.search(r'[Tt]\s*=\s*([\d\.]+)', lower)
        g_part = g_def.group(1).strip() if g_def else "g(t)"
        T_part = T_def.group(1) if T_def else "2"
        return f"fourier transform of x(t) = sum g(t-{T_part}k), g(t) = {g_part}, T={T_part}"
    for pattern, prefix in [
        (r'inverse\s+laplace\s+(?:transform\s+)?(?:of\s+)?(.+)', "inverse laplace of"),
        (r'inverse\s+fourier\s+(?:transform\s+)?(?:of\s+)?(.+)', "inverse fourier of"),
        (r'(?:find|compute)?\s*(?:the\s+)?laplace\s+(?:transform\s+)?(?:of\s+)?(.+)', "laplace of"),
        (r'(?:find|compute)?\s*(?:the\s+)?fourier\s+transform\s+(?:of\s+)?(.+)', "fourier transform of"),
        (r'fourier\s+series\s+(?:of\s+)?(.+)', "fourier series of"),
        (r'(?:find|compute)?\s*(?:the\s+)?convolution\s+(?:of\s+)?(.+)', "convolve"),
        (r'(?:sketch|plot|draw|graph)\s+(?:the\s+signal\s+)?(.+)', "plot"),
    ]:
        m = re.search(pattern, lower, re.IGNORECASE | re.DOTALL)
        if m:
            expr = m.group(1).strip().split('\n')[0].strip(' .')
            return f"{prefix} {expr}"
    return None

# ══════════════════════════════════════════════════════════════════════════════
# SESSION-AWARE LLM PROMPTS
# ══════════════════════════════════════════════════════════════════════════════
_LATEX_INSTRUCTION = (
    "FORMATTING RULE: Wrap ALL math in $...$ or $$...$$ LaTeX. "
    "Use \\frac{}{}, \\int, \\sum, \\omega, \\delta, \\pi, \\mathcal{L}, \\mathcal{F}. "
    "Always space LaTeX commands from following letters: $\\pi G$ not $\\piG$."
)

_SESSION_RULES = (
    "IMPORTANT:\n"
    "1. The uploaded document is the PRIMARY source of truth.\n"
    "2. Use general knowledge only to explain/clarify — never to override the doc.\n"
    "3. If OCR looks garbled, say so and give your best interpretation.\n"
    f"4. {_LATEX_INSTRUCTION}"
)

def _prompt_explain_memo(doc_text: str, instruction: str) -> str:
    return (f"{_SESSION_RULES}\n\nYou are a patient Signals & Systems tutor.\n\n"
            f"Uploaded memo:\n\"\"\"\n{doc_text}\n\"\"\"\n\n"
            f"Student instruction: {instruction}\n\n"
            f"1. Identify the question/section.\n"
            f"2. Re-explain step-by-step for a first-year student.\n"
            f"3. For every formula, add a plain-English explanation.\n"
            f"4. Generate TWO similar practice problems with solutions.\n"
            f"5. End with one study tip.\n"
            f"Wrap all math in $...$ or $$...$$.")

def _prompt_mark(memo_text: str, student_work: str) -> str:
    return (f"{_SESSION_RULES}\n\nMemo:\n\"\"\"\n{memo_text}\n\"\"\"\n\n"
            f"Student's work:\n\"\"\"\n{student_work}\n\"\"\"\n\n"
            f"1. OVERALL VERDICT: CORRECT / PARTIALLY CORRECT / INCORRECT\n"
            f"2. MARKS: estimate earned vs total\n"
            f"3. WHAT IS CORRECT\n"
            f"4. ERRORS: what student wrote → what it should be → concept missed\n"
            f"5. IMPROVEMENT: how to fix each error\n"
            f"6. ENCOURAGEMENT\nWrap all math in $...$ or $$...$$.")

def _prompt_solve(doc_text: str, instruction: str) -> str:
    return (f"{_SESSION_RULES}\n\nDocument:\n\"\"\"\n{doc_text}\n\"\"\"\n\n"
            f"Student instruction: {instruction}\n\n"
            f"Respond directly. Numbered steps for math. Explain every formula. "
            f"Wrap all math in $...$ or $$...$$.")

def _prompt_explain(doc_text: str, question: str) -> str:
    return (f"{_SESSION_RULES}\n\nDocument:\n\"\"\"\n{doc_text}\n\"\"\"\n\n"
            f"Student question: {question}\n\n"
            f"FACTUAL → 2-4 sentences. CONCEPTUAL → paragraph + example. "
            f"CALCULATION → numbered steps. Wrap all math in $...$ or $$...$$.")

def _call_llm(prompt: str, max_tokens: int = 1800) -> str:
    payload = {
        "model": LLM_MODEL,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type":  "application/json",
    }
    try:
        resp = requests.post(TOGETHER_ENDPOINT, json=payload, headers=headers, timeout=90)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"❌ LLM call failed: {e}"

def _route_session_prompt(doc_text: str, instruction: str) -> str:
    instr_lower = instruction.lower()
    if any(kw in instr_lower for kw in ["explain how", "step by step", "walk me through",
                                         "practice", "similar", "break down"]):
        return _prompt_explain_memo(doc_text, instruction)
    if any(kw in instr_lower for kw in ["mark", "check", "feedback", "grade",
                                         "is my answer", "did i get"]):
        return _prompt_mark(doc_text, instruction)
    if re.search(r'[Qq](?:uestion)?\s*\d+', instruction):
        ctx = extract_question_with_context(doc_text, instruction)
        if any(kw in instr_lower for kw in ["solve", "answer", "find", "compute",
                                              "calculate", "work out"]):
            return _prompt_solve(doc_text, ctx)
        return _prompt_explain(doc_text, ctx)
    if any(kw in instr_lower for kw in ["solve", "calculate", "find", "compute",
                                          "work out", "answer"]):
        return _prompt_solve(doc_text, instruction)
    return _prompt_explain(doc_text, instruction)

# ══════════════════════════════════════════════════════════════════════════════
# VECTOR STORE
# ══════════════════════════════════════════════════════════════════════════════
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def build_vector_store_if_needed(pdf_folder: str, chroma_dir: str):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    if os.path.exists(chroma_dir) and os.listdir(chroma_dir):
        print("✅ Loading existing ChromaDB…")
        try:
            client = chromadb.PersistentClient(
                path=chroma_dir, settings=Settings(anonymized_telemetry=False))
            vs    = Chroma(client=client, embedding_function=embeddings)
            count = vs._collection.count()
            print(f"   {count} vectors loaded.")
            if count > 0:
                return vs
            print("   Index empty — rebuilding…")
        except Exception as e:
            print(f"   Load failed: {e} — rebuilding…")

    print("📄 Building ChromaDB from PDFs…")
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
    if not pdf_files:
        print("⚠️  No PDFs found.")
        return None
    loader   = PyPDFDirectoryLoader(pdf_folder)
    docs     = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, chunk_overlap=400,
        separators=["\n\nQuestion", "\n\nQ", "\n\n", "\n"]
    )
    chunks = splitter.split_documents(docs)
    print(f"   {len(docs)} pages → {len(chunks)} chunks")
    client = chromadb.PersistentClient(
        path=chroma_dir, settings=Settings(anonymized_telemetry=False))
    vs = Chroma.from_documents(chunks, embedding=embeddings, client=client)
    print(f"   ✅ {vs._collection.count()} vectors saved")
    return vs

# ══════════════════════════════════════════════════════════════════════════════
# RAG CHAINS
# ══════════════════════════════════════════════════════════════════════════════
TUTOR_PROMPT = PromptTemplate.from_template(
    "You are a Signals and Systems tutor assistant.\n\n"
    "FORMATTING RULE: Wrap ALL math in $...$ or $$...$$ LaTeX. "
    "Use \\frac{{}}{{}}, \\int, \\sum, \\omega, \\delta, \\pi, "
    "\\mathcal{{L}}, \\mathcal{{F}}, e^{{-st}}. "
    "Always space commands from letters: $\\pi G$ not $\\piG$.\n\n"
    "Classify the question (silently):\n"
    "  A) FACTUAL → 2-4 sentences.\n"
    "  B) CONCEPTUAL → short paragraph + one example.\n"
    "  C) CALCULATION → numbered steps, explain every symbol.\n\n"
    "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def build_chains(vs):
    retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    llm = ChatTogether(model=LLM_MODEL, temperature=0.7, max_tokens=1024)
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | TUTOR_PROMPT | llm | StrOutputParser()
    )

vector_store = build_vector_store_if_needed(PDF_FOLDER, CHROMA_DIR)
qa_chain     = None
if vector_store:
    qa_chain = build_chains(vector_store)
    print("✅ RAG chain ready")
else:
    print("⚠️  Running without knowledge base.")

# ══════════════════════════════════════════════════════════════════════════════
# TELEGRAM UTILITY
# ══════════════════════════════════════════════════════════════════════════════
async def send_long_code(update: Update, text: str) -> None:
    chunk_size = 4090
    for i in range(0, len(text), chunk_size):
        await _safe_reply(update, f"```\n{text[i:i+chunk_size]}\n```", parse_mode="Markdown")

# ══════════════════════════════════════════════════════════════════════════════
# COMMAND HANDLERS
# ══════════════════════════════════════════════════════════════════════════════
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await _safe_reply(update,
         "👋 Hi there! I'm your *Signals & Systems 1* tutor bot.\n\n"
        "I am here to help you understand any Signals and Systems 1 topic that you may be struggling in, or need clarification in. Feel free to ask me any Signals and Systems 1 related questions. I can help you with the following calculations: \n"
        "📡 *Laplace Transforms*\n" 
        "📡 *Inverse Laplace Transforms*\n"
        "📡 *Fourier Transform*\n"
        "📡 *Inverse Fourier Transforms*\n"
        "📡 *Convolution*\n"
        "📊 *Plot signals*\n\n"
        "I can also help you calculate how much you need in your exam to pass the course. Ask me anything, I am here for you. Do not suffer while I am here.\n"
        "Use /help for the full guide.",
        parse_mode="Markdown")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await _safe_reply(update,
        "👋*Full guide:*\n\n"
        "1. *Laplace*           _laplace of e^(-2*t)*u(t)_\n"
        "2. *Inverse Laplace*   _inverse laplace of 1/(s+2)_\n"
        "   Also: _ilt of s/(s^2+4)_\n"
        "3. *Fourier*           _fourier transform of e^(-t)*u(t)_\n"
        "4. *Inverse Fourier*   _inverse fourier of 1/(1+j*omega)_\n"
        "   Also: _ift of 2*pi*DiracDelta(omega)_\n"
        "5. *Periodic Fourier*  _fourier transform of x(t)=sum g(t-2k), g(t)=e^(-t)*u(t), T=2_\n"
        "6. *Fourier Series*    _fourier series of t**2, T=2*pi_\n"
        "7. *Convolution*       _convolve e^(-t)*u(t) with u(t)_\n"
        "8. *Plot*              _plot 2*u(t-2)_  /  _draw u[n]-u[n-3]_\n"
        "9. *Upload PDF/image* then ask:\n"
        "     _explain how Question 1(b) was solved_\n"
        "     _answer Question 2(a)_\n"
        "     _mark my work against this memo_\n"
        "10. *Mark calculator* — _how much do I need to pass_\n\n"
        "Use * for multiply, ** for power\n"
        "e.g. e**(-2*t)*u(t)  or  e^-2t*u(t)",
        parse_mode="Markdown")

# ══════════════════════════════════════════════════════════════════════════════
# PERIOD EXTRACTOR
# ══════════════════════════════════════════════════════════════════════════════
def _extract_period(text: str) -> float | None:
    m = re.search(r'[Tt]\s*=\s*([0-9.]+\s*\*?\s*pi|[0-9.]+)', text)
    if not m:
        return None
    raw = m.group(1).replace(" ", "")
    if "pi" in raw:
        num = raw.replace("pi", "").replace("*", "") or "1"
        return float(num) * float(np.pi)
    return float(raw)

# ══════════════════════════════════════════════════════════════════════════════
# TEXT HANDLER
# ══════════════════════════════════════════════════════════════════════════════
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    question = update.message.text.strip()
    q_lower  = question.lower()
    msg_id   = update.message.message_id
    chat_id  = update.effective_chat.id

    # ── 1. Mark calculator ────────────────────────────────────────────────────
    if await handle_mark_session(update, context):
        return

    # ── 2. Session-based document Q&A ─────────────────────────────────────────
    if session_has(chat_id):
        sess = session_get(chat_id)
        await _safe_reply(update,
            f"⏳ Working on it using *{sess['source']}* as reference…",
            parse_mode="Markdown")

        pending_work = context.user_data.pop("pending_student_work", None)
        if pending_work:
            prompt   = _prompt_mark(sess["text"], pending_work)
            response = _call_llm(prompt)
            title    = f"Marking Feedback — {sess['source']}"
            await send_llm_response(update, response, title, msg_id)
        else:
            prompt   = _route_session_prompt(sess["text"], question)
            response = _call_llm(prompt)
            title    = f"Answer — {sess['source']}"
            await send_llm_response(update, response, title, msg_id)

        session_clear(chat_id)
        await _safe_reply(update,
            "_(Session cleared — uploaded file no longer in memory.)_",
            parse_mode="Markdown")
        return

    # ── 3. Math tools ─────────────────────────────────────────────────────────

    # ── Inverse Laplace ───────────────────────────────────────────────────────
    if is_inv_laplace(q_lower):
        expr_str = extract_expr(question)
        if not expr_str:
            await _safe_reply(update,
                "⚠️ Please include an expression, e.g.:\n"
                "  _inverse laplace of 1/(s+2)_\n"
                "  _ilt of s/(s^2+4)_", parse_mode="Markdown")
            return
        await _safe_reply(update, "⏳ Computing Inverse Laplace transform…")
        steps, err = _build_inv_laplace_steps(expr_str)
        if err:
            await _safe_reply(update, err)
        else:
            png = _render_math_png("Inverse Laplace Transform", steps, msg_id)
            if png and os.path.exists(png):
                success = await _safe_reply_photo(
                    update, png, f"Inverse Laplace of  F(s) = {expr_str}")
                if not success:
                    await send_long_code(update, compute_inv_laplace(expr_str))
            else:
                await send_long_code(update, compute_inv_laplace(expr_str))
        return

    # ── Inverse Fourier ───────────────────────────────────────────────────────
    if is_inv_fourier(q_lower):
        expr_str = extract_expr(question)
        if not expr_str:
            await _safe_reply(update,
                "⚠️ Please include an expression, e.g.:\n"
                "  _inverse fourier of 1/(1+j*omega)_\n"
                "  _ift of 2*pi*DiracDelta(omega)_", parse_mode="Markdown")
            return
        await _safe_reply(update, "⏳ Computing Inverse Fourier transform…")
        steps, err = _build_inv_fourier_steps(expr_str)
        if err:
            await _safe_reply(update, err)
        else:
            png = _render_math_png("Inverse Fourier Transform", steps, msg_id)
            if png and os.path.exists(png):
                success = await _safe_reply_photo(
                    update, png, f"Inverse Fourier of  F(ω) = {expr_str}")
                if not success:
                    await send_long_code(update, compute_inv_fourier(expr_str))
            else:
                await send_long_code(update, compute_inv_fourier(expr_str))
        return

    # ── Periodic summation Fourier ────────────────────────────────────────────
    if is_periodic_fourier(q_lower):
        period_val = _extract_period(question) or 2.0
        g_str = None
        g_def_q = re.search(r'g\s*\(\s*t\s*\)\s*=\s*([^,\n]+)', question, re.IGNORECASE)
        g_str = g_def_q.group(1).strip() if g_def_q else None
        if not g_str:
            await _safe_reply(update,
                "⚠️ Please tell me what *g(t)* is, e.g.:\n"
                "  _fourier transform of x(t) = sum g(t-2k), g(t) = e^(-t)*u(t), T=2_",
                parse_mode="Markdown")
            return
        await _safe_reply(update,
            f"⏳ Computing periodic summation Fourier (g(t)={g_str}, T={period_val})…")
        steps, err = _build_periodic_fourier_steps(g_str, period_val)
        if err:
            await _safe_reply(update, err)
        else:
            png = _render_math_png("Fourier Transform  (Periodic Summation)", steps, msg_id)
            if png and os.path.exists(png):
                await _safe_reply_photo(update, png,
                    f"Periodic Fourier: x(t)=Σg(t−{period_val}k), g(t)={g_str}, T={period_val}")
        return

    # ── Laplace ───────────────────────────────────────────────────────────────
    if is_laplace(q_lower):
        expr_str = extract_expr(question)
        if not expr_str:
            await _safe_reply(update,
                "⚠️ Please include an expression, e.g.:\n"
                "  _laplace of e^(-2*t)*u(t)_", parse_mode="Markdown")
            return
        await _safe_reply(update, "⏳ Computing Laplace transform…")
        steps, err = _build_laplace_steps(expr_str)
        if err:
            await _safe_reply(update, err)
        else:
            png = _render_math_png("Laplace Transform", steps, msg_id)
            if png and os.path.exists(png):
                success = await _safe_reply_photo(
                    update, png, f"Laplace Transform of  f(t) = {expr_str}")
                if not success:
                    await send_long_code(update, compute_laplace(expr_str))
            else:
                await send_long_code(update, compute_laplace(expr_str))
        return

    # ── Fourier Transform ─────────────────────────────────────────────────────
    if is_fourier(q_lower):
        expr_str = extract_expr(question)
        if not expr_str:
            await _safe_reply(update,
                "⚠️ Please include an expression, e.g.:\n"
                "  _fourier transform of e^(-t)*u(t)_", parse_mode="Markdown")
            return
        await _safe_reply(update, "⏳ Computing Fourier transform…")
        steps, err = _build_fourier_steps(expr_str)
        if err:
            await _safe_reply(update, err)
        else:
            png = _render_math_png("Fourier Transform", steps, msg_id)
            if png and os.path.exists(png):
                success = await _safe_reply_photo(
                    update, png, f"Fourier Transform of  f(t) = {expr_str}")
                if not success:
                    await send_long_code(update, compute_fourier(expr_str))
            else:
                await send_long_code(update, compute_fourier(expr_str))
        return

    # ── Fourier Series ────────────────────────────────────────────────────────
    if is_fs(q_lower):
        period = _extract_period(question)
        if not period:
            await _safe_reply(update,
                "⚠️ Please include the period, e.g.:\n"
                "  _fourier series of t, T=2_", parse_mode="Markdown")
            return
        expr_str = extract_expr(question)
        if not expr_str:
            expr_str = re.sub(r',?\s*[Tt]\s*=\s*[^\s]+', '', question).strip()
            expr_str = extract_expr(expr_str) or expr_str
        await _safe_reply(update,
            f"⏳ Computing Fourier series for f(t)={expr_str}, T={period:.4g}…")
        steps, err = _build_fourier_series_steps(expr_str, period)
        if err:
            await _safe_reply(update, err)
        else:
            png = _render_math_png("Fourier Series", steps, msg_id)
            if png and os.path.exists(png):
                success = await _safe_reply_photo(
                    update, png, f"Fourier Series: f(t)={expr_str}, T={period:.4g}")
                if not success:
                    await send_long_code(update, compute_fourier_series(expr_str, period))
            else:
                await send_long_code(update, compute_fourier_series(expr_str, period))
        return

    # ── Convolution ───────────────────────────────────────────────────────────
    if is_conv(q_lower):
        e1, e2 = _parse_two_signals(question)
        if not (e1 and e2):
            await _safe_reply(update,
                "⚠️ Please specify both signals, e.g.:\n"
                "  _convolve e^(-t)*u(t) with u(t)_", parse_mode="Markdown")
            return
        await _safe_reply(update,
            f"⏳ Computing convolution of f(t)={e1} and g(t)={e2}…")
        steps, err, plot_path = _build_convolution_steps(e1, e2, msg_id)
        if err:
            await _safe_reply(update, err)
        else:
            png = _render_math_png("Convolution  (f ★ g)(t)", steps, msg_id)
            if png and os.path.exists(png):
                success = await _safe_reply_photo(
                    update, png, f"Convolution: f(t)={e1} ★ g(t)={e2}")
                if not success:
                    text_result, _ = compute_convolution(e1, e2, msg_id)
                    await send_long_code(update, text_result)
            else:
                text_result, plot_path2 = compute_convolution(e1, e2, msg_id)
                await send_long_code(update, text_result)
                plot_path = plot_path or plot_path2
            if plot_path and os.path.exists(plot_path):
                await _safe_reply_photo(update, plot_path,
                                        "📊 Numerical convolution (f ★ g)(t)")
        return

    # ── Plot ──────────────────────────────────────────────────────────────────
    if is_plot(q_lower):
        await _safe_reply(update, "📊 Generating plot…")
        fig_path = generate_plot(question, msg_id)
        if fig_path and os.path.exists(fig_path):
            success = await _safe_reply_photo(update, fig_path, f"📈 {question}")
            if not success:
                await _safe_reply(update,
                    "⚠️ Plot generated but could not be sent. Please try again.")
        else:
            await _safe_reply(update,
                "⚠️ Could not parse that expression.\n"
                "Examples: _plot 2*u(t-2)_  /  _draw u[n]-u[n-3]_",
                parse_mode="Markdown")
        return

    # ── 4. General tutor Q&A via RAG ──────────────────────────────────────────
    if not qa_chain:
        await _safe_reply(update, "⚠️ No knowledge base loaded.")
        return
    await _safe_reply(update, "🤔 Thinking…")
    try:
        answer = qa_chain.invoke(question)
        await send_llm_response(update, answer, "Tutor Answer", msg_id)
    except Exception as e:
        await _safe_reply(update, f"❌ Something went wrong: {str(e)}")

# ══════════════════════════════════════════════════════════════════════════════
# PHOTO HANDLER
# ══════════════════════════════════════════════════════════════════════════════
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    caption = (update.message.caption or "").strip()
    chat_id = update.effective_chat.id
    msg_id  = update.message.message_id

    await _safe_reply(update,
        "📷 Got your photo — running OCR… (~15–30s)")

    import io
    photo_file  = await update.message.photo[-1].get_file()
    buf         = io.BytesIO()
    await photo_file.download_to_memory(buf)
    image_bytes = buf.getvalue()

    extracted = _ocr_image_bytes(image_bytes, "jpeg")
    await _safe_reply(update, f"📝 Extracted:\n\n{extracted}")

    sess = session_get(chat_id)

    if caption:
        mark_keywords = ["mark", "check", "compare", "grade", "evaluate", "feedback"]
        if sess and any(kw in caption.lower() for kw in mark_keywords):
            await _safe_reply(update,
                f"⏳ Marking against *{sess['source']}*…", parse_mode="Markdown")
            prompt   = _prompt_mark(sess["text"], extracted)
            response = _call_llm(prompt)
            await send_llm_response(update, response,
                                    f"Marking Feedback — {sess['source']}", msg_id)
            session_clear(chat_id)
            await _safe_reply(update,
                "_(Session cleared.)_", parse_mode="Markdown")
        else:
            doc_text = sess["text"] if sess else extracted
            source   = sess["source"] if sess else "handwritten photo"
            await _safe_reply(update,
                f"⏳ Processing using *{source}* as reference…",
                parse_mode="Markdown")
            prompt   = _route_session_prompt(doc_text, caption)
            response = _call_llm(prompt)
            await send_llm_response(update, response, f"Answer — {source}", msg_id)
            if sess:
                session_clear(chat_id)
                await _safe_reply(update, "_(Session cleared.)_", parse_mode="Markdown")
    else:
        routed_command = auto_route_extracted_text(extracted)
        if routed_command:
            await _safe_reply(update,
                f"🔍 Detected: `{routed_command[:120]}`\nSolving…",
                parse_mode="Markdown")
            q_lower = routed_command.lower()

            if is_inv_laplace(q_lower):
                expr_str = extract_expr(routed_command)
                if expr_str:
                    steps, err = _build_inv_laplace_steps(expr_str)
                    if not err:
                        png = _render_math_png("Inverse Laplace Transform", steps, msg_id)
                        if png and os.path.exists(png):
                            if await _safe_reply_photo(update, png,
                                                        f"Inverse Laplace of {expr_str}"):
                                return
                    await send_long_code(update, compute_inv_laplace(expr_str or routed_command))
                    return

            elif is_inv_fourier(q_lower):
                expr_str = extract_expr(routed_command)
                if expr_str:
                    steps, err = _build_inv_fourier_steps(expr_str)
                    if not err:
                        png = _render_math_png("Inverse Fourier Transform", steps, msg_id)
                        if png and os.path.exists(png):
                            if await _safe_reply_photo(update, png,
                                                        f"Inverse Fourier of {expr_str}"):
                                return
                    await send_long_code(update, compute_inv_fourier(expr_str or routed_command))
                    return

            elif is_laplace(q_lower):
                expr_str = extract_expr(routed_command)
                if expr_str:
                    steps, err = _build_laplace_steps(expr_str)
                    if not err:
                        png = _render_math_png("Laplace Transform", steps, msg_id)
                        if png and os.path.exists(png):
                            if await _safe_reply_photo(update, png,
                                                        f"Laplace of {expr_str}"):
                                return
                    await send_long_code(update, compute_laplace(expr_str))
                    return

            elif is_fourier(q_lower):
                expr_str = extract_expr(routed_command)
                if expr_str:
                    steps, err = _build_fourier_steps(expr_str)
                    if not err:
                        png = _render_math_png("Fourier Transform", steps, msg_id)
                        if png and os.path.exists(png):
                            if await _safe_reply_photo(update, png,
                                                        f"Fourier of {expr_str}"):
                                return
                    await send_long_code(update, compute_fourier(expr_str))
                    return

            elif is_conv(q_lower):
                e1, e2 = _parse_two_signals(routed_command)
                if e1 and e2:
                    steps, err, plot_path = _build_convolution_steps(e1, e2, msg_id)
                    if not err:
                        png = _render_math_png("Convolution", steps, msg_id)
                        if png and os.path.exists(png):
                            await _safe_reply_photo(update, png, f"{e1} ★ {e2}")
                        if plot_path and os.path.exists(plot_path):
                            await _safe_reply_photo(update, plot_path,
                                                     "📊 Numerical convolution")
                        return

            elif is_plot(q_lower):
                fig_path = generate_plot(routed_command, msg_id)
                if fig_path and os.path.exists(fig_path):
                    if await _safe_reply_photo(update, fig_path, f"📈 {routed_command}"):
                        return

            if qa_chain:
                await _safe_reply(update, "🤔 Solving with tutor…")
                try:
                    answer = qa_chain.invoke(extracted)
                    await send_llm_response(update, answer, "Tutor Answer", msg_id)
                except Exception as e:
                    await _safe_reply(update, f"❌ {str(e)}")
            else:
                await _safe_reply(update,
                    "⚠️ No knowledge base loaded. Add a caption with your question.")
        else:
            if sess:
                context.user_data["pending_student_work"] = extracted
                await _safe_reply(update,
                    f"I have *{sess['source']}* loaded.\n\n"
                    "Reply *mark* to compare, or tell me what you'd like.",
                    parse_mode="Markdown")
            else:
                session_store(chat_id, extracted, "Handwritten photo / diagram")
                await _safe_reply(update,
                    "Photo loaded. What would you like me to do?\n"
                    "  - Solve this\n"
                    "  - Explain step by step\n"
                    "  - What is this question asking?\n"
                    "  - Describe this diagram")

# ══════════════════════════════════════════════════════════════════════════════
# DOCUMENT HANDLER
# ══════════════════════════════════════════════════════════════════════════════
async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    doc       = update.message.document
    caption   = (update.message.caption or "").strip()
    chat_id   = update.effective_chat.id
    file_name = doc.file_name or "uploaded_file"
    extension = os.path.splitext(file_name)[-1].lower()

    SUPPORTED_IMAGES = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
    if extension not in {".pdf"} | SUPPORTED_IMAGES:
        await _safe_reply(update,
            "⚠️ Unsupported file type. Please send a PDF or image (PNG, JPG, WEBP).")
        return

    await _safe_reply(update,
        f"📄 Received *{file_name}* — extracting content…",
        parse_mode="Markdown")

    tg_file = await doc.get_file()
    with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as tmp:
        tmp_path = tmp.name
    await tg_file.download_to_drive(tmp_path)
    try:
        with open(tmp_path, "rb") as fh:
            file_bytes = fh.read()
    finally:
        os.unlink(tmp_path)

    if extension == ".pdf":
        extracted = _extract_pdf_bytes(file_bytes)
        source    = f"PDF: {file_name}"
    else:
        mime      = "jpeg" if extension in (".jpg", ".jpeg") else extension.lstrip(".")
        extracted = _ocr_image_bytes(file_bytes, mime)
        source    = f"Image: {file_name}"

    session_store(chat_id, extracted, source)

    await _safe_reply(update,
        f"✅ *{source}* loaded.\n\n"
        "Now tell me what you'd like:\n"
        "  - Explain how Question 1(b) was solved\n"
        "  - Answer Question 2(a)\n"
        "  - Give me two practice problems like Question 3\n"
        "  - Mark my work against this memo",
        parse_mode="Markdown")

    if caption:
        sess   = session_get(chat_id)
        await _safe_reply(update,
            f"⏳ Acting on your caption: _{caption}_…",
            parse_mode="Markdown")
        prompt   = _route_session_prompt(sess["text"], caption)
        response = _call_llm(prompt)
        await send_llm_response(update, response, f"Answer — {source}",
                                msg_id=update.message.message_id)
        session_clear(chat_id)
        await _safe_reply(update, "_(Session cleared.)_", parse_mode="Markdown")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN  ← only this function was changed (fixed HTTPXRequest conflict)
# ══════════════════════════════════════════════════════════════════════════════
async def main():
    # Two separate HTTPXRequest instances — one for regular API calls,
    # one for get_updates — both with raised timeouts.
    # You CANNOT mix .request()/.get_updates_request() with
    # .http_version()/.get_updates_http_version() — they are mutually exclusive.
    req = tg_request.HTTPXRequest(
        connection_pool_size=8,
        connect_timeout=30.0,
        read_timeout=30.0,
        write_timeout=30.0,
        pool_timeout=30.0,
    )
    req_updates = tg_request.HTTPXRequest(
        connection_pool_size=8,
        connect_timeout=30.0,
        read_timeout=30.0,
        write_timeout=30.0,
        pool_timeout=30.0,
    )

    app = (
        ApplicationBuilder()
        .token(BOT_TOKEN)
        .request(req)
        .get_updates_request(req_updates)
        .build()
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help",  help_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.PHOTO,        handle_photo))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))

    print("✅ Bot is running!")
    await app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    asyncio.run(main())
