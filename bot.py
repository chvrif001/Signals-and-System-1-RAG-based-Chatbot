#Chavalala Rifuwo (CHVRIF001)
#CHVRIF001
#EEE4022F
#Designing a course-specific chatbot
#20 May 2026
#Llama-3 Instruct Lite Powered Chatbot

#Import all packages and libraries to be used this project

import os
import asyncio
import nest_asyncio  
import re
import base64
import requests
import tempfile
import textwrap
import warnings
import time
import unicodedata

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Use the non-interactive backend so matplotlib never tries to open a display window
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


# Congiguring all the tokens/APIs used in this chatbot
# All tokens/APIs are store in shared environments on Railway hosting
# CHROMA_DIR points to /data/ so the vector index survives container restarts on Render's persistent disk.
BOT_TOKEN         = os.environ["BOT_TOKEN"]
TOGETHER_API_KEY  = os.environ["TOGETHER_API_KEY"]
GEMINI_API_KEY    = os.environ.get("GEMINI_API_KEY", "")
ADMIN_USER_ID     = int(os.environ.get("ADMIN_USER_ID", "0"))

LLM_MODEL         = "meta-llama/Meta-Llama-3-8B-Instruct-Lite" #Endpoint String for Llama
TOGETHER_ENDPOINT = "https://api.together.xyz/v1/chat/completions"

PDF_FOLDER  = "./knowledge_base" #This is where all the course material is stored in
CHROMA_DIR  = os.environ.get("CHROMA_DIR", "/data/chroma_db")
PLOT_FOLDER = "/tmp/plots"

os.makedirs(PDF_FOLDER,  exist_ok=True)
os.makedirs(CHROMA_DIR,  exist_ok=True)
os.makedirs(PLOT_FOLDER, exist_ok=True)

os.environ["TOGETHER_API_KEY"] = TOGETHER_API_KEY


# Safe reply wrapper
# Telegram crashes when it takes a long time before receiving a response
# This implementation allows the chatbot to retry 3 times before crashing, this allows the system to handle complex queries that may take a long time before crashing.
async def _safe_reply(update: Update, text: str,
                      parse_mode: str | None = None,
                      retries: int = 3,
                      delay: float = 3.0) -> None:
    """Send a text reply, retrying up to 3 times on network failure."""
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

_send_photo_with_retry = _safe_reply_photo  # Legacy alias kept for compatibility

# Session_Store
# When a user uploads a PDF or photo, the extracted text is stored with a chat_id. The next message sent is treated as an instruction about that upload
# The session is cleared after the response is returned to prevent previous content being used in new queries causing confusion
_session_store: dict[int, dict] = {}

def session_store(chat_id: int, text: str, source: str) -> None:
    _session_store[chat_id] = {"text": text, "source": source}

def session_get(chat_id: int) -> dict | None:
    return _session_store.get(chat_id)

def session_clear(chat_id: int) -> None:
    _session_store.pop(chat_id, None)

def session_has(chat_id: int) -> bool:
    return chat_id in _session_store

# SymPy Symbols, defining all the symbols that may be used by Signals students
t_sym = sp.Symbol("t",     real=True)
s_sym = sp.Symbol("s",     complex=True)
w_sym = sp.Symbol("omega", real=True)   # ω — the Fourier angular frequency variable
n_sym = sp.Symbol("n",     integer=True)
tau   = sp.Symbol("tau",   real=True)   # integration dummy variable for convolution
a_sym = sp.Symbol("a",     positive=True)


# Signal Parser
# This part handles the preprocessing of mathematical expression inputs
# _normalise() rewrites informal strings into something sympy computations accepts and understands,
def _rect(x):
    return sp.Piecewise(
        (sp.Integer(1),      sp.Abs(x) < sp.Rational(1, 2)),
        (sp.Rational(1, 2),  sp.Eq(sp.Abs(x), sp.Rational(1, 2))),
        (sp.Integer(0),      True)
    )

def _normalise(expr: str) -> str:
    """
    Convert informal student notation to valid SymPy syntax.
    """
    s = expr.strip()
    s = s.replace("^", "**").replace("{", "(").replace("}", ")")
    s = re.sub(r'\brect\s*\(',                      'rect(',          s)
    s = re.sub(r'(\d)(u\s*[\(\[])',                  r'\1*\2',         s)
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

# common signal names resolve to the right SymPy objects
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
    """Parse a continuous-time expression string into a SymPy expression."""
    return sp.sympify(_normalise(text), locals=_COMMON_NS)

def _normalise_dt(expr: str) -> str:
    """Normalise discrete-time expressions (uses n, u[n])."""
    s = expr.strip()
    s = s.replace("^", "**").replace("{", "(").replace("}", ")")
    s = re.sub(r'\be\b', 'E', s)
    s = re.sub(r'\bu\s*\[([^\]]+)\]',                r'UnitStep(\1)', s)
    s = re.sub(r'\b(?:delta|δ)\s*\[([^\]]+)\]',      r'KronDelta(\1)', s)
    s = re.sub(r'(\d)(n\b)',                          r'\1*\2', s)
    return s

def _unit_step_dt(val):
    """Discrete unit step: 1 for n >= 0, 0 otherwise."""
    arr = np.asarray(val, dtype=float)
    return np.where(arr >= 0, 1.0, 0.0)

def _kron_delta_dt(val):
    """Kronecker delta: 1 at n=0, 0 elsewhere."""
    arr = np.asarray(val, dtype=float)
    return (arr == 0).astype(float)

def parse_dt_expr(text: str):
    """
    Parse a discrete-time expression string and return a vectorised evaluator function that accepts a numpy array of n values.
    """
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


# defining characters that suggest the user typed a maths expression, not just words
_SIGNAL_CHARS = ['(', '[', 't', 'n', 's', 'sin', 'cos', 'exp',
                 'delta', 'δ', '**', '*', '+', 'sqrt', 'log',
                 'Heaviside', 'DiracDelta', 'rect']

# This is a user query preprocessing that strips question preambles like "what is the", "find the"
_QUESTION_PREFIX = re.compile(
    r'^(?:what\s+is\s+(?:the\s+)?|what\'s\s+(?:the\s+)?|find\s+(?:the\s+)?|'
    r'compute\s+(?:the\s+)?|calculate\s+(?:the\s+)?|determine\s+(?:the\s+)?|'
    r'give\s+me\s+(?:the\s+)?|show\s+me\s+(?:the\s+)?)',
    re.IGNORECASE
)

# This is a user query preprocessing that strips operation prefixes like "laplace of", "plot", "fourier transform of"
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
#This is a user query preprocessing that strips all other text that may come after
_TRAILING_NOISE = re.compile(
    r'\s+(?:looks?\s+like|for\s+me|please|now|here|to\s+me|as\s+well)\s*$',
    re.IGNORECASE
)

def extract_expr(question: str) -> str | None:
    """
    Strip natural language preamble from a question and return the bare mathematical expression, or None if nothing expression-like is found.
    """
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

# This part decides whether a response should be rendered as a maths image or plain text.
# Mathematical calculations should be returned as images so that symbols are well outputted to students
_MATH_TITLE_KEYWORDS = (
    "laplace", "fourier", "convolution", "transform", "series",
    "plot", "signal", "derivation", "calculation", "equation",
    "answer —", "marking feedback", "tutor answer",
    "inverse laplace", "inverse fourier",
)
_PLAIN_TEXT_TITLES = ("tutor answer",)

def _is_math_title(title: str) -> bool:
    """Return True if the response title suggests maths content should be rendered as PNG."""
    t = title.lower()
    if any(kw in t for kw in _PLAIN_TEXT_TITLES):
        return False
    return any(kw in t for kw in _MATH_TITLE_KEYWORDS)


# Llama genrated response cleaner
# During implementation, I realised that Llama returns responses that start with phrases like "Sure, I'll be happy to help you"
# This part strips all of that so responses start directly with the answer, nothing more added on
_LLM_PREAMBLE_RE = re.compile(
    r"(?:"
    r"I(?:'ll|'d| will| would)[\s\S]{0,80}?(?:help|assist|answer)[\s\S]{0,120}?\n+"
    r"|(?:Sure|Certainly|Of course|Absolutely|Great)[!,.][\s\S]{0,120}?\n+"
    r"|(?:\*\*)?Classification(?:\*\*)?:.*?(?:\n|$)"
    r"|(?:\*\*)?Question(?:\*\*)?:.*?(?:\n|$)"
    r"|(?:\*\*)?Answer(?:\*\*)?:\s*"
    r")",
    re.IGNORECASE,
)

def _clean_llm_response(text: str) -> str:
    """Iteratively strips common LLM begin phrase patterns until the text starts cleanly."""
    for _ in range(8):
        stripped = _LLM_PREAMBLE_RE.sub("", text, count=1).lstrip()
        if stripped == text.lstrip():
            break
        text = stripped
    return text.strip()

# Latex to Unicode Converter
# Llama returns math in Latex, however Telegram does not render latex, it renders dollar signs literally, not as maths. 
# This part converts LaTeX to readable Unicode so answers look good in chat for Llama response that contain math expressions
_LATEX_UNICODE = {
    r'\omega':   'ω', r'\Omega':   'Ω', r'\alpha':   'α', r'\beta':    'β',
    r'\gamma':   'γ', r'\delta':   'δ', r'\Delta':   'Δ', r'\epsilon': 'ε',
    r'\zeta':    'ζ', r'\eta':     'η', r'\theta':   'θ', r'\Theta':   'Θ',
    r'\lambda':  'λ', r'\Lambda':  'Λ', r'\mu':      'μ', r'\nu':      'ν',
    r'\xi':      'ξ', r'\pi':      'π', r'\Pi':      'Π', r'\rho':     'ρ',
    r'\sigma':   'σ', r'\Sigma':   'Σ', r'\tau':     'τ', r'\phi':     'φ',
    r'\Phi':     'Φ', r'\chi':     'χ', r'\psi':     'ψ', r'\Psi':     'Ψ',
    r'\infty':   '∞', r'\cdot':    '·', r'\times':   '×', r'\approx':  '≈',
    r'\geq':     '≥', r'\leq':     '≤', r'\neq':     '≠', r'\pm':      '±',
    r'\to':      '→', r'\rightarrow': '→', r'\leftarrow': '←',
    r'\int':     '∫', r'\sum':     'Σ', r'\prod':    'Π', r'\partial': '∂',
    r'\nabla':   '∇', r'\in':      '∈', r'\notin':   '∉', r'\subset':  '⊂',
    r'\cup':     '∪', r'\cap':     '∩', r'\forall':  '∀', r'\exists':  '∃',
    r'\sqrt':    '√', r'\star':    '★',
}

_SUP_DIGITS = str.maketrans('0123456789+-', '⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻')
_SUB_DIGITS = str.maketrans('0123456789+-', '₀₁₂₃₄₅₆₇₈₉₊₋')


def _frac_to_text(num: str, den: str) -> str:
    return f"({num.strip()}/{den.strip()})"


def _latex_to_plain(text: str) -> str:
    """Convert LaTeX math notation to readable plain-text Unicode for Telegram."""
    def _convert_math(expr: str) -> str:
        s = expr
        # cases environment 
        def _cases(m):
            inner = m.group(1)
            branches = [b.strip() for b in re.split(r'\\\\', inner) if b.strip()]
            parts = []
            for b in branches:
                b = re.sub(r'\s*&\s*', ',  ', b).strip()
                parts.append(_convert_math(b))
            return '{ ' + ' ; '.join(parts) + ' }'
        s = re.sub(r'\\begin\s*\{cases\}([\s\S]*?)\\end\s*\{cases\}', _cases, s)
        def _frac(m):
            return _frac_to_text(m.group(1), m.group(2))
        s = re.sub(r'\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}', _frac, s)
        s = re.sub(r'\\(?:mathcal|mathbf|mathrm|text|operatorname)\s*\{([^}]*)\}', r'\1', s)
        s = re.sub(r'\\rm\s+([A-Za-z]+)', r'\1', s)
        for cmd, uni in sorted(_LATEX_UNICODE.items(), key=lambda x: -len(x[0])):
            s = s.replace(cmd, uni)
        s = re.sub(r'\\(?:left|right)\s*', '', s)
        def _sup(m):
            inner = (m.group(1) or m.group(2) or '').strip()
            try:
                return inner.translate(_SUP_DIGITS)
            except Exception:
                return f"^({inner})" if len(inner) > 1 else f"^{inner}"
        s = re.sub(r'\^\{([^}]*)\}|\^([A-Za-z0-9])', _sup, s)
        def _sub(m):
            inner = (m.group(1) or m.group(2) or '').strip()
            try:
                return inner.translate(_SUB_DIGITS)
            except Exception:
                return f"_({inner})" if len(inner) > 1 else f"_{inner}"
        s = re.sub(r'_\{([^}]*)\}|_([A-Za-z0-9])', _sub, s)
        s = re.sub(r'\\(?:quad|qquad|,|;|:|\s)', ' ', s)
        s = re.sub(r'\\([A-Za-z]+)', r'\1', s)
        s = re.sub(r'[{}]', '', s)
        s = re.sub(r'  +', ' ', s).strip()
        return s

    def _replace_display(m):
        return '\n' + _convert_math(m.group(1).strip()) + '\n'
    text = re.sub(r'\$\$([\s\S]+?)\$\$', _replace_display, text)

    def _replace_inline(m):
        return _convert_math(m.group(1))
    text = re.sub(r'\$([^$\n]+?)\$', _replace_inline, text)
    return text

# For math handled by Matplotlib
# This part filters text to only characters Matplotlib's mathtext renderer can handle.
def _strip_non_renderable(text: str) -> str:
    result = []
    for ch in text:
        cp = ord(ch)
        cat = unicodedata.category(ch)
        if cp < 0x0300:
            result.append(ch)
            continue
        if 0x0300 <= cp <= 0x03FF:
            result.append(ch)
        elif 0x2000 <= cp <= 0x23FF:
            result.append(ch)
        elif cat.startswith(('L', 'N', 'P', 'Z', 'S')):
            if cat == 'So' and cp > 0x2FFF:
                continue
            if cp > 0xFFFF:
                continue
            result.append(ch)
    return ''.join(result)


# Latex and Mathtext
# Since SymPy's full LaTeX output and matplotlib's mathtext use different variables for maths term, this parts makes the key conversions to bridge the gap
# For example, imaginary term i is converted to j
def _sanitise_mathtext(s: str) -> str:
    s = _strip_non_renderable(s)
    def _cases_to_inline(m):
        inner = m.group(1)
        branches = [b.strip() for b in re.split(r'\\\\', inner) if b.strip()]
        parts = []
        for b in branches:
            b = re.sub(r'\s*&\s*', r',\\ ', b).strip()
            parts.append(b)
        return r'\{\ ' + r'\ ;\ '.join(parts) + r'\ \}'
    s = re.sub(r'\\begin\s*\{cases\}([\s\S]*?)\\end\s*\{cases\}', _cases_to_inline, s)
    s = re.sub(r'\\(?:begin|end)\s*\{[^}]*\}', '', s)
    # Convert unicode symbols back to mathtext commands
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
    # SymPy sometimes outputs \theta(t) for the Heaviside function — rewrite to u(t)
    s = re.sub(r'\\theta\\left\(([^)]+)\\right\)', r'u(\1)', s)
    s = s.replace(r'\theta\left(t\right)', r'u(t)')
    # Use imaginary unit 'j' instead of maths 'i' 
    s = re.sub(r'(?<![a-zA-Z\\])i(?![a-zA-Z0-9{\\])', 'j', s)
    s = re.sub(r'\\mathcal\{([^}]+)\}',     r'\\mathbf{\1}', s)
    s = re.sub(r'\\mathscr\{([^}]+)\}',     r'\\mathbf{\1}', s)
    s = re.sub(r'\\mathrm\{([^}]+)\}',      r'\\rm \1',      s)
    s = re.sub(r'\\operatorname\{([^}]+)\}', r'\\rm \1',      s)
    s = re.sub(r'\\text\{([^}]*)\}',        r'\\rm \1',      s)
    # Spacing commands
    s = s.replace(r'\qquad', r'\ \ \ \ ')
    s = s.replace(r'\quad',  r'\ \ ')
    s = s.replace(r'\,',     r'\ ')
    s = s.replace(r'\;',     r'\ ')
    s = re.sub(r'\\left\s*',  '', s)
    s = re.sub(r'\\right\s*', '', s)
    # Ensure a space after math commands so they don't merge with the next token
    s = re.sub(
        r'(\\(?:pi|alpha|beta|gamma|delta|epsilon|zeta|eta|theta|iota|kappa|'
        r'lambda|mu|nu|xi|rho|sigma|tau|upsilon|phi|chi|psi|omega|'
        r'Gamma|Delta|Theta|Lambda|Xi|Pi|Sigma|Upsilon|Phi|Psi|Omega|'
        r'infty|cdot|times|star|approx|geq|leq|neq|sum|int|prod|rm|mathbf))'
        r'(?=[A-Za-z])',
        r'\1 ', s
    )
    return s

# Latex math renderer
# Converts a SymPy expression to LaTeX string for use in PNG rendering.
def _sympy_to_latex(expr: sp.Expr) -> str:
    return sp.latex(expr)

def _try_render_row(ax, x_label, x_expr, y, label, latex_str, fontsize=22):
    """Render a single label + maths row onto the axes, with plain-text fallback."""
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
        # Fall back to plain text if mathtext rendering fails
        ax.text(x_expr, y, latex_str,
                transform=ax.transAxes,
                fontsize=fontsize - 3, color="#333333",
                va="top", ha="left", fontfamily="monospace", usetex=False)

def _render_math_png(title: str, steps: list[tuple[str, str]], msg_id: int) -> str | None:
    """
    Render a list of (label, LaTeX) step as a white PNG image. Returns the file path, or None if rendering fails.
    """
    try:
        n      = len(steps)
        FONT   = 22
        ROW_IN = 0.95   # vertical space per row in inches
        PAD_IN = 1.6    # padding for title + divider
        fig_h  = max(4.0, n * ROW_IN + PAD_IN)
        fig_w  = 16.0

        fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="white")
        ax.set_facecolor("white")
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        title_clean = _strip_non_renderable(re.sub(r'[^\x00-\x7F]+', '', title).strip())
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

# Llama response Renderer
# Converts LLM text into a PNG image.
# Split on $...$ and $$...$$ blocks, and label each segment as math and scan those lines for equation like patterns
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

# detect lines that contain inline equations that may not be wrapped in $
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
    """Promote unicode math symbols in plain text to mathtext commands."""
    for pat, rep in _PLAIN_TO_MATHTEXT:
        expr = re.sub(pat, rep, expr)
    return expr

def _extract_math_blocks(text: str) -> list[tuple[str, str]]:
    """Split text into math and prose segments."""
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
    """Further classify prose lines and promote equation-looking lines to math."""
    out = []
    for line in prose.split('\n'):
        stripped = line.strip()
        if _EQ_LINE_RE.match(stripped) and len(stripped) > 5:
            out.append(('math', _plain_to_mt(stripped)))
        else:
            out.append(('prose', line))
    return out

def render_response_png(llm_text: str, title: str, msg_id: int) -> str | None:
    """
    Render llama response as one or more PNG images depending on the length. Extra pages are stored in render_response_png._extra_pages for the caller to send sequentially.
    """
    rows: list[tuple[str, str]] = []
    for kind, content in _extract_math_blocks(llm_text):
        if kind == 'math':
            rows.append(('math', content))
        else:
            rows.extend(_split_prose_lines(content))

    # Strip leading/trailing blank rows
    while rows and rows[0][1].strip() == '':
        rows.pop(0)
    while rows and rows[-1][1].strip() == '':
        rows.pop()

    if not rows:
        return None

    FIG_W      = 22.0
    PROSE_FS   = 22
    MATH_FS    = 28
    LINE_H     = 0.75   # vertical space per prose line
    MATH_H     = 1.10   # vertical space per maths line
    TITLE_H    = 1.10
    PAD        = 0.8
    WRAP_WIDTH = 100    # characters before wrapping prose

    # Estimate total figure height needed
    total_h = TITLE_H + PAD
    for kind, txt in rows:
        if kind == 'math':
            total_h += MATH_H
        else:
            n_lines = max(1, len(textwrap.wrap(txt, width=WRAP_WIDTH)) if txt.strip() else 1)
            total_h += LINE_H * n_lines
    total_h = max(8.0, total_h)

    MAX_PAGE_H = 40.0   # split into multiple images above this height
    render_response_png._extra_pages = []

    if total_h > MAX_PAGE_H:
        # split rows across multiple images
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

        # Render the other pages and save their paths for the caller
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
            p_ax.text(0.5, p_y, f"{_strip_non_renderable(title)}  (page {pi})",
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
                        p_ax.text(0.06, p_y, _strip_non_renderable(txt),
                                  fontsize=MATH_FS - 2, color='#444444',
                                  va='top', ha='left', fontfamily='monospace', usetex=False)
                    p_y -= MATH_H
                else:
                    display = re.sub(r'\*\*?([^*]+)\*\*?', r'\1', txt)
                    display = re.sub(r'__?([^_]+)__?', r'\1', display)
                    display = _strip_non_renderable(display)
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
    ax.text(0.5, y, _strip_non_renderable(title),
            fontsize=18, fontweight='bold', color='#1a1a2e',
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
                ax.text(MATH_INDENT, y, _strip_non_renderable(txt),
                        fontsize=MATH_FS - 2, color='#444444',
                        va='top', ha='left', fontfamily='monospace', usetex=False)
            y -= MATH_H
        else:
            display = re.sub(r'\*\*?([^*]+)\*\*?', r'\1', txt)
            display = re.sub(r'__?([^_]+)__?', r'\1', display)
            display = _strip_non_renderable(display)
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

# Send Llama Respose
# Decides whether to send response as a PNG image (for maths) or plain text but falls back to plain text if image generation or upload fails.
async def send_llm_response(update: Update, response_text: str,
                             title: str, msg_id: int,
                             force_image: bool = False) -> None:
    response_text = _clean_llm_response(response_text)

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

    plain_text = _latex_to_plain(response_text)
    for i in range(0, len(plain_text), 4096):
        await _safe_reply(update, plain_text[i:i + 4096])


# Llama Fallback
# Called when symbolic computation fails or the question is conceptual.
async def _llm_fallback(update: Update, question: str,
                         title: str, msg_id: int) -> None:
    await _safe_reply(update, "Let me work through that for you…")
    try:
        if qa_chain:
            answer = qa_chain.invoke(question)
        else:
            answer = _call_llm(
                f"{_LATEX_INSTRUCTION}\n\nYou are a Signals and Systems tutor.\n\n"
                f"Question: {question}\n\nAnswer:"
            )
        answer = _clean_llm_response(answer)
        answer = _latex_to_plain(answer)
        await send_llm_response(update, answer, title, msg_id)
    except Exception as e:
        await _safe_reply(update, f"Could not answer: {e}")

# Laplace step
# Builds a list of (label, LaTeX) pairs showing the full working for Laplace transform. Uses SymPy's native laplace_transform() engine.
def _identify_laplace_rule_latex(expr: sp.Expr) -> str:
    """Return the LaTeX string for the known transform pair that applies."""
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
    """
    Compute the Laplace transform of expr_str and return step-by-step working.
    """
    steps: list[tuple[str, str]] = []
    try:
        f      = parse_ct_expr(expr_str)
        f_tex  = _sympy_to_latex(f)
        steps.append(("Input",      rf"f(t) = {f_tex}"))
        steps.append(("Definition", r"F(s) = \int_{0}^{\infty} f(t)\,e^{-st}\,dt"))
        steps.append(("Rule / Form", _identify_laplace_rule_latex(f)))

        # Show each additive term separately (linearity of the transform)
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
        return [], f"laplace_failed: {e}"


# Inverse Laplace step
def _identify_inv_laplace_rule_latex(expr: sp.Expr) -> str:
    """Return the known inverse Laplace pair that most closely matches the input."""
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
    """
    Compute the inverse Laplace transform of expr_str and return step-by-step working.
    """
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

        # Attempt partial fraction decomposition to simplify before inverting
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
        return [], f"inv_laplace_failed: {e}"

def compute_inv_laplace(expr_str: str) -> str:
    """Plain-text inverse Laplace computation"""
    lines = ["--- INVERSE LAPLACE TRANSFORM ---\n"]
    lines.append(f"Input:  F(s) = {expr_str}\n")
    lines.append("Definition:  f(t) = (1/2πj) ∫ F(s)·e^(st) ds" + "\n")
    try:
        F = sp.sympify(_normalise(expr_str), locals={**_COMMON_NS, "s": s_sym})
    except Exception as e:
        return f"Could not parse expression: {e}"
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
        lines.append(f"\nf(t) = {sp.pretty(result)}")
    except Exception as e:
        lines.append(f"SymPy could not find a closed form: {e}")
    return "\n".join(lines)

# Fourier step builder
# SymPy's built-in fourier_transform() uses the f-frequency convention struggles with distributions.
# This implemented engine attempts to first integrate the function multiplied by e^{-jωt} over all real values, for simple cos and sin with step function, use the laplace calculation then substitute the s with w. For simple cos and sin, their pairs are predefined

def _identify_fourier_rule_latex(expr: sp.Expr) -> str:
    """Return the known Fourier pair that best matches the input expression."""
    s = str(expr)
    if "DiracDelta" in s:
        return r"\mathcal{F}\{\delta(t)\} = 1"
    if "Heaviside" in s and "exp" not in s:
        return r"\mathcal{F}\{u(t)\} = \pi \delta(\omega) + \frac{1}{j\omega}"
    if "exp" in s and "sin" not in s and "cos" not in s:
        return r"\mathcal{F}\{e^{-at}u(t)\} = \frac{1}{a+j\omega},\quad a>0"
    if "exp" in s and "cos" in s:
        return r"\mathcal{F}\{e^{-at}\cos(\omega_0 t)u(t)\} = \frac{a+j\omega}{(a+j\omega)^2+\omega_0^2}"
    if "exp" in s and "sin" in s:
        return r"\mathcal{F}\{e^{-at}\sin(\omega_0 t)u(t)\} = \frac{\omega_0}{(a+j\omega)^2+\omega_0^2}"
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
    """
    Compute the Fourier transform using a cascade of specialised methods. Raises ValueError if no method succeeds.
    """
    # Pull out any leading scalar so it isn't lost during pattern matching
    coeff = sp.Integer(1)
    core  = f
    if f.is_Mul:
        num_factors = [a for a in f.args if a.is_number]
        if num_factors:
            coeff = sp.Mul(*num_factors)
            core  = f / coeff

    s = str(core)

    # Direct symbolic integration
    # Works for rect/Piecewise and any signal where SymPy can evaluate the integral in closed form.
    try:
        result = sp.integrate(
            core * sp.exp(-sp.I * w_sym * t_sym), (t_sym, -sp.oo, sp.oo)
        )
        if not result.has(sp.Integral):
            return sp.simplify(coeff * result)
    except Exception:
        pass

    # cosine with u(t)
    # Use the Laplace-to-Fourier substitution s → jω. This is more reliablethan direct integration
    if "exp" in s and "cos" in s and "Heaviside" in s:
        try:
            result = sp.laplace_transform(core, t_sym, sp.I * w_sym, noconds=True)
            result = sp.simplify(result)
            if "Integral" not in str(result) and "LaplaceTransform" not in str(result):
                return sp.simplify(coeff * result)
        except Exception:
            pass

        m_cos = re.search(r'cos\(([^,)]+)\*t\b|cos\(t\*([^,)]+)', s)
        m_exp = re.search(r'exp\(-([^*\s)]+)\*t\b|exp\(t\*\((-[^)]+)\)', s)
        if m_cos and m_exp:
            try:
                w0_raw = (m_cos.group(1) or m_cos.group(2) or '').strip()
                a_raw  = (m_exp.group(1) or m_exp.group(2) or '').strip().lstrip('-')
                w0 = sp.sympify(w0_raw, locals=_COMMON_NS)
                a  = sp.sympify(a_raw,  locals=_COMMON_NS)
                num = a + sp.I * w_sym
                den = (a + sp.I * w_sym)**2 + w0**2
                return sp.simplify(coeff * num / den)
            except Exception:
                pass

    # sine with u(t)
    if "exp" in s and "sin" in s and "Heaviside" in s:
        try:
            result = sp.laplace_transform(core, t_sym, sp.I * w_sym, noconds=True)
            result = sp.simplify(result)
            if "Integral" not in str(result) and "LaplaceTransform" not in str(result):
                return sp.simplify(coeff * result)
        except Exception:
            pass
        m_sin = re.search(r'sin\(([^,)]+)\*t\b|sin\(t\*([^,)]+)', s)
        m_exp = re.search(r'exp\(-([^*\s)]+)\*t\b|exp\(t\*\((-[^)]+)\)', s)
        if m_sin and m_exp:
            try:
                w0_raw = (m_sin.group(1) or m_sin.group(2) or '').strip()
                a_raw  = (m_exp.group(1) or m_exp.group(2) or '').strip().lstrip('-')
                w0 = sp.sympify(w0_raw, locals=_COMMON_NS)
                a  = sp.sympify(a_raw,  locals=_COMMON_NS)
                num = w0
                den = (a + sp.I * w_sym)**2 + w0**2
                return sp.simplify(coeff * num / den)
            except Exception:
                pass

    # Damped cosine withou u(t) 
    if "exp" in s and "cos" in s and "Heaviside" not in s:
        # Try the Laplace substitution first (most general)
        try:
            result = sp.laplace_transform(core * sp.Heaviside(t_sym),
                                           t_sym, sp.I * w_sym, noconds=True)
            result = sp.simplify(result)
            if "Integral" not in str(result) and "LaplaceTransform" not in str(result):
                return sp.simplify(coeff * result)
        except Exception:
            pass
        # Direct closed-form formula
        m_cos = re.search(r'cos\(([^,)]+)\*t\b|cos\(t\*([^,)]+)', s)
        m_exp = re.search(r'exp\(-([^*\s)]+)\*t', s)
        if m_cos and m_exp:
            try:
                w0_raw = (m_cos.group(1) or m_cos.group(2) or '').strip()
                a_raw  = m_exp.group(1).strip()
                w0 = sp.sympify(w0_raw, locals=_COMMON_NS)
                a  = sp.sympify(a_raw,  locals=_COMMON_NS)
                num = a + sp.I * w_sym
                den = (a + sp.I * w_sym)**2 + w0**2
                return sp.simplify(coeff * num / den)
            except Exception:
                pass

    # Damped sine without u(t)
    # Same reasoning as abovr, assume causal signal when u(t) is absent.
    if "exp" in s and "sin" in s and "Heaviside" not in s:
        try:
            result = sp.laplace_transform(core * sp.Heaviside(t_sym),
                                           t_sym, sp.I * w_sym, noconds=True)
            result = sp.simplify(result)
            if "Integral" not in str(result) and "LaplaceTransform" not in str(result):
                return sp.simplify(coeff * result)
        except Exception:
            pass
        m_sin = re.search(r'sin\(([^,)]+)\*t\b|sin\(t\*([^,)]+)', s)
        m_exp = re.search(r'exp\(-([^*\s)]+)\*t', s)
        if m_sin and m_exp:
            try:
                w0_raw = (m_sin.group(1) or m_sin.group(2) or '').strip()
                a_raw  = m_exp.group(1).strip()
                w0 = sp.sympify(w0_raw, locals=_COMMON_NS)
                a  = sp.sympify(a_raw,  locals=_COMMON_NS)
                return sp.simplify(coeff * w0 / ((a + sp.I * w_sym)**2 + w0**2))
            except Exception:
                pass

    # Pure exponential decay with u(t)
    # e^{-at}*u(t) → 1/(a + jω)
    if "exp" in s and "Heaviside" in s and "sin" not in s and "cos" not in s:
        m_exp = re.search(r'exp\(-([^*\s)]+)\*t\b|exp\(t\*\((-[^)]+)\)', s)
        if m_exp:
            try:
                a_raw = (m_exp.group(1) or m_exp.group(2) or '').strip().lstrip('-')
                a = sp.sympify(a_raw, locals=_COMMON_NS)
                return sp.simplify(coeff / (a + sp.I * w_sym))
            except Exception:
                pass

    # Pure exponential decay without u(t)
    # Treat as causal — same result as step 6
    if "exp" in s and "Heaviside" not in s and "sin" not in s and "cos" not in s:
        m_exp = re.search(r'exp\(-([^*\s)]+)\*t', s)
        if m_exp:
            try:
                a_raw = m_exp.group(1).strip()
                a = sp.sympify(a_raw, locals=_COMMON_NS)
                return sp.simplify(coeff / (a + sp.I * w_sym))
            except Exception:
                pass

    #Pure cosine, usedthe propertt
    if "cos" in s and "exp" not in s:
        m = re.search(r'cos\(([^,)]+)\*t\b|cos\(t\*([^,)]+)', s)
        if m:
            w0_raw = (m.group(1) or m.group(2) or '').strip()
            try:
                w0 = sp.sympify(w0_raw, locals=_COMMON_NS)
                return coeff * sp.pi * (
                    sp.DiracDelta(w_sym - w0) + sp.DiracDelta(w_sym + w0)
                )
            except Exception:
                pass

    # Pure sine, used the property 
    if "sin" in s and "exp" not in s:
        m = re.search(r'sin\(([^,)]+)\*t\b|sin\(t\*([^,)]+)', s)
        if m:
            w0_raw = (m.group(1) or m.group(2) or '').strip()
            try:
                w0 = sp.sympify(w0_raw, locals=_COMMON_NS)
                return coeff * sp.I * sp.pi * (
                    sp.DiracDelta(w_sym + w0) - sp.DiracDelta(w_sym - w0)
                )
            except Exception:
                pass

    # Unit step, used the property
    if "Heaviside" in s and "exp" not in s:
        return sp.simplify(
            coeff * (sp.pi * sp.DiracDelta(w_sym) + 1 / (sp.I * w_sym))
        )

    # constant
    if core == sp.Integer(1):
        return coeff * 2 * sp.pi * sp.DiracDelta(w_sym)

    raise ValueError("No closed-form Fourier pair found")

def _build_fourier_steps(expr_str: str) -> tuple[list[tuple[str, str]], str]:
    """
    Compute the Fourier transform of expr_str and return step-by-step working.
    """
    steps: list[tuple[str, str]] = []
    try:
        f      = parse_ct_expr(expr_str)
        f_tex  = _sympy_to_latex(f)
        steps.append(("Input",      rf"f(t) = {f_tex}"))
        steps.append(("Definition", r"F(\omega) = \int_{-\infty}^{\infty} f(t)\, e^{-j\omega t}\, dt"))
        steps.append(("Known pair", _identify_fourier_rule_latex(f)))

        result     = _fourier_direct(f)
        result_tex = _sympy_to_latex(result)
        # Use engineering 'j' instead of SymPy's 'i' for imaginary unit
        result_tex = re.sub(r'(?<![a-zA-Z])i(?![a-zA-Z])', 'j', result_tex)
        steps.append(("Result", rf"F(\omega) = {result_tex}"))
        return steps, ""
    except Exception as e:
        return [], f"fourier_failed: {e}"

# Inverse fourier transform
def _identify_inv_fourier_rule_latex(expr: sp.Expr) -> str:
    """Return the known inverse Fourier pair that best matches the input."""
    s = str(expr)
    if "DiracDelta" in s:
        return r"\mathcal{F}^{-1}\{\delta(\omega)\} = \frac{1}{2\pi}"
    if "omega" in s and ("a +" in s or "a+" in s or "j*omega" in s or "j\omega" in s):
        return r"\mathcal{F}^{-1}\left\{\frac{1}{a+j\omega}\right\} = e^{-at}u(t),\quad a>0"
    if "pi" in s and "DiracDelta" in s:
        return r"\mathcal{F}^{-1}\{\pi\delta(\omega\pm\omega_0)\} \to \cos/\sin \text{ terms}"
    return r"f(t) = \frac{1}{2\pi}\int_{-\infty}^{\infty} F(\omega)\,e^{j\omega t}\,d\omega"

def _build_inv_fourier_steps(expr_str: str) -> tuple[list[tuple[str, str]], str]:
    """
    Compute the inverse Fourier transform of expr_str and return step-by-step working.
    """
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

        # Try direct integration first
        result = sp.integrate(
            F * sp.exp(sp.I * w_sym * t_sym) / (2 * sp.pi),
            (w_sym, -sp.oo, sp.oo)
        )
        # Fall back to SymPy's inverse_fourier_transform (different frequency convention)
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
        return [], f"inv_fourier_failed: {e}"

def compute_inv_fourier(expr_str: str) -> str:
    """Plain-text inverse Fourier computation """
    lines = ["--- INVERSE FOURIER TRANSFORM ---\n"]
    lines.append(f"Input:  F(ω) = {expr_str}\n")
    lines.append("Definition:  f(t) = (1/2π) ∫ F(ω)·e^(jωt) dω\n")
    try:
        F = sp.sympify(_normalise(expr_str), locals={**_COMMON_NS, "omega": w_sym})
    except Exception as e:
        return f"Could not parse expression: {e}"
    try:
        result = sp.integrate(
            F * sp.exp(sp.I * w_sym * t_sym) / (2 * sp.pi),
            (w_sym, -sp.oo, sp.oo)
        )
        if not result.has(sp.Integral):
            result = sp.simplify(result)
            lines.append(f"Result:\n   f(t) = {sp.pretty(result)}\n")
            lines.append(f"\nf(t) = {sp.pretty(result)}")
        else:
            lines.append("Integral could not be evaluated in closed form.")
    except Exception as e:
        lines.append(f"Could not evaluate: {e}")
    return "\n".join(lines)

# Periodic summation fourier step builder
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
        return [], f"periodic_fourier_failed: {e}"

def compute_fourier(expr_str: str) -> str:
    """Plain-text Fourier transform computation"""
    lines = ["--- FOURIER TRANSFORM ---\n"]
    lines.append(f"Input:  f(t) = {expr_str}\n")
    try:
        f = parse_ct_expr(expr_str)
        result = _fourier_direct(f)
        lines.append(f"F(ω) = {sp.pretty(result)}")
    except Exception as e:
        lines.append(f"{e}")
    return "\n".join(lines)

def compute_laplace(expr_str: str) -> str:
    """Plain-text Laplace transform computation"""
    lines = ["--- LAPLACE TRANSFORM ---\n"]
    lines.append(f"Input:  f(t) = {expr_str}\n")
    try:
        f      = parse_ct_expr(expr_str)
        result = sp.laplace_transform(f, t_sym, s_sym, noconds=True)
        result = sp.simplify(result)
        lines.append(f"F(s) = {sp.pretty(result)}")
    except Exception as e:
        lines.append(f"{e}")
    return "\n".join(lines)

# Convolution
# Computes the convolution symbolically, with a numerical plot as a fallback (and always as a addition to the respone to show the shape).
def _simplify_heaviside_powers(expr: sp.Expr) -> sp.Expr:
    return expr.replace(
        lambda e: e.is_Pow and isinstance(e.base, sp.Heaviside) and e.exp.is_positive,
        lambda e: e.base
    )

def _extract_heaviside_onset(expr: sp.Expr) -> sp.Expr | None:
    """Find where a causal signal switches on (i.e. the argument of its Heaviside)."""
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
    """
    Determine finite integration limits for the convolution of two causal signals. Returns (lower, upper) for the tau integral, or None if limits can't be determined.
    """
    d_f = _extract_heaviside_onset(f)
    d_g = _extract_heaviside_onset(g)
    if d_f is None or d_g is None:
        return None
    return d_f, t_sym - d_g

def _lambdify_ct(expr: sp.Expr):
    """Convert a SymPy CT expression to a numpy-compatible callable."""
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
    """Generate a three plots, f(t), g(t), and their numerical convolution."""
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
        axes[2].set(title="(f * g)(t)  [numerical]", xlabel="t")
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
    """
    Compute convolution symbolically and return step-by-step working plus a plot.
    """
    steps: list[tuple[str, str]] = []
    try:
        f = parse_ct_expr(expr1_str)
        g = parse_ct_expr(expr2_str)
    except Exception as e:
        return [], f"conv_failed: {e}", None

    steps.append(("f(t)",       _sympy_to_latex(f)))
    steps.append(("g(t)",       _sympy_to_latex(g)))
    steps.append(("Definition", r"(f\star g)(t)=\int_{-\infty}^{\infty}f(\tau)\,g(t-\tau)\,d\tau"))

    f_tau     = f.subs(t_sym, tau)
    g_shift   = g.subs(t_sym, t_sym - tau)
    integrand = sp.expand(f_tau * g_shift)
    steps.append(("Substitution",
                   rf"f(\tau)={_sympy_to_latex(f_tau)},\quad g(t-\tau)={_sympy_to_latex(g_shift)}"))
    steps.append(("Integrand",   _sympy_to_latex(integrand)))

    # Use tighter integration limits for causal signals to help SymPy evaluate
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
        steps.append(("Fallback", r"\text{See numerical plot below}"))
        plot_path = _numerical_convolution_plot(f, g, msg_id)

    return steps, "", plot_path

def _parse_two_signals(text: str):
    """
    Split a convolution request like "convolve f(t) with g(t)" into two expression strings to diferentiate the functions.
    """
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
    """Plain-text convolution computation"""
    lines = ["--- CONVOLUTION ---\n"]
    lines.append(f"f(t) = {expr1_str}\ng(t) = {expr2_str}\n")
    try:
        f = parse_ct_expr(expr1_str)
        g = parse_ct_expr(expr2_str)
    except Exception as e:
        return f"{e}", None
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
        lines.append(f"(f * g)(t) = {sp.pretty(result)}")
        plot_path = _numerical_convolution_plot(f, g, msg_id)
    except Exception as e:
        lines.append(f"Symbolic failed: {e}")
        plot_path = _numerical_convolution_plot(f, g, msg_id)
    return "\n".join(lines), plot_path

# Signal Plotting, plots all the signals
# ══════════════════════════════════════════════════════════════════════════════
PLOT_KEYWORDS     = ["plot", "draw", "graph", "sketch", "show me",
                     "visualise", "visualize", "diagram"]
DISCRETE_KEYWORDS = ["x[n]", "u[n]", "delta[n]", "δ[n]", "h[n]", "y[n]", "[n]"] #Keywords to differentiate the discrete time plots from CT plots

def is_discrete(question: str) -> bool:
    """Return True if the question refers to a discrete-time signal."""
    return any(kw in question for kw in DISCRETE_KEYWORDS)

def plot_ct(expr_str: str, msg_id: int) -> str | None:
    """Plot a continuous-time signal and return the PNG file path."""
    try:
        expr   = parse_ct_expr(expr_str)
        t_vals = np.linspace(-4, 8, 4000)
        y_vals = np.real(_lambdify_ct(expr)(t_vals)).astype(float)
        y_vals = np.clip(y_vals, -10, 10)  # Prevent runaway axes from near-singular signals
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
    """Plot a discrete-time signal as a stem plot and return the PNG file path."""
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
    """Plot a Dirac delta as an upward arrow at t=t0."""
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.axhline(0, color="k", lw=.8)
    ax.annotate("", xy=(t0, 1), xytext=(t0, 0),
                arrowprops=dict(arrowstyle="-|>", color="steelblue", lw=2.5))
    ax.set_xlim(t0 - 3, t0 + 3); ax.set_ylim(-0.2, 1.5)
    label = f"delta(t - {t0})" if t0 != 0 else "delta(t)"
    ax.set(title=label, xlabel="t", ylabel="delta(t)"); ax.grid(True, alpha=.4)
    fig.tight_layout()
    path = os.path.join(PLOT_FOLDER, f"plot_{msg_id}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    return path

def generate_plot(question: str, msg_id: int) -> str | None:
    """Route a plot request to the correct plotter based on signal type."""
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

# Keyword triggers
# This is the routing logic of the system, keywords triggerwhich module to follow

LAPLACE_KEYS     = ["laplace of", "l transform", "l{", "the laplace transform", "laplace of", "the laplace transform of", "LT of"]
INV_LAPLACE_KEYS = ["inverse laplace of", "inv laplace", "ilt", "ilaplace", "inverse l transform", "laplace inverse", "inverse LT", "inverse laplace", "the inverse laplace transform"]
FOURIER_KEYS     = ["fourier transform of", "ft{", "fourier of", "f transform", "ft of", "compute ft", "find ft", "f.t. of", "fourier tf", "the fourier transform"]
INV_FOURIER_KEYS = ["inverse fourier of", "inv fourier", "ift", "ifourier", "inverse ft", "inverse f transform", "fourier inverse", "inverse FT", "the inverse fourier"]
CONV_KEYS        = ["convolution of", "convolve", "f*g", "f★g", "f star g", "convolute", "the convolution", "convolut"]
PERIODIC_FOURIER_KEYS = ["sum", "summation", "periodic summation", "x(t) =", "x(t)=", "k=-inf", "g(t-2k)", "g(t -"]

def is_inv_laplace(q: str) -> bool:  return any(k in q for k in INV_LAPLACE_KEYS)
def is_laplace(q: str) -> bool:
    if is_inv_laplace(q): return False   # Guard: inverse must not match forward
    return any(k in q for k in LAPLACE_KEYS)
def is_inv_fourier(q: str) -> bool:  return any(k in q for k in INV_FOURIER_KEYS)
def is_fourier(q: str) -> bool:
    if is_inv_fourier(q): return False   # Guard: inverse must not match forward
    return any(k in q for k in FOURIER_KEYS)
def is_conv(q: str) -> bool:  return any(k in q for k in CONV_KEYS)
def is_plot(q: str) -> bool:  return any(k in q for k in PLOT_KEYWORDS)
def is_periodic_fourier(q: str) -> bool:
    return any(k in q for k in FOURIER_KEYS) and any(k in q for k in PERIODIC_FOURIER_KEYS)

# This is a precautionary step that prevents purely conceptual questions like "what is a laplace transform?" from being routed into the symbolic engine and producing a parse error.
# the verb prefix si stripped first so only the expression part is checked.
_MATH_INDICATOR_RE = re.compile(
    r'[(){}\[\]]'                          # brackets
    r'|[+\-*/^]'                           # operators
    r'|\b(?:sin|cos|exp|sqrt|log|rect)\b'  # known functions
    r'|\b[eE]\^'                           # e^ notation
    r'|\bt\b|\bn\b|\bs\b'                  # signal variables
    r'|\bu\s*[\(\[]'                       # u(t) / u[n]
    r'|\d+\s*\*'                           # digit followed by multiply
    r'|\b\d+\.\d+\b'                       # decimal number
    r'|delta|δ|omega|ω'                    # common signal terms
    , re.IGNORECASE
)

def _has_math_expr(question: str) -> bool:
    """Return True only if the question contains something that looks like a mathematical expression, not just conceptual words."""
    stripped = _QUESTION_PREFIX.sub('', question.strip())
    stripped = _VERB_PREFIX.sub('', stripped).strip()
    return bool(_MATH_INDICATOR_RE.search(stripped))


# Mark Calculator
# Starts a conversation that collects test/lab/tutorial marks and calculates the minimum exam score needed to pass the module.
WEIGHTS      = {"tutorials": .10, "labs": .10, "tests": .20, "exam": .60}
PASS_MARK    = 50.0
TRIGGER_KEYS = ["to pass", "exam mark", "calculate marks", "how much do i need",
                "calculate mark breakdown", "what do i need", "minimum", "final mark", "course mark"]
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
    """Calculate the required exam mark and format the result message."""
    test_avg = (data["test1"] + data["test2"]) / 2
    weighted = (test_avg * WEIGHTS["tests"] +
                data["labs"] * WEIGHTS["labs"] +
                data["tutorials"] * WEIGHTS["tutorials"])
    req_exam = (PASS_MARK - weighted) / WEIGHTS["exam"]
    lines = [
        "*Your Mark Breakdown*\n",
        f"  Test average:       {test_avg:.1f}%",
        f"  Labs average:       {data['labs']:.1f}%",
        f"  Tutorials average:  {data['tutorials']:.1f}%",
        f"\n  *Marks secured so far: {weighted:.1f}% (out of 40%)*",
        f"  *(Exam carries the remaining 60%)*\n",
    ]
    #This is just a little something for motivation, so the student knows if its possible and how hard they need to work for it
    if req_exam <= 0:
        lines.append("You've already secured enough to pass!")
    elif req_exam > 100:
        lines.append(f"You'd need {req_exam:.1f}% — mathematically impossible. Give it your best!")
    else:
        lines.append(f"*You need at least {req_exam:.1f}% in the exam to pass.*")
        if req_exam <= 50:   lines.append("Very achievable — keep it up!")
        elif req_exam <= 70: lines.append("Tough but doable with a solid plan!")
        else:                lines.append("Hard work required — start now!")
    return "\n".join(lines)

async def handle_mark_session(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """
    Intercept messages while a mark-calculation session is active. Returns True if the message was consumed, False if normal routing should continue.
    """
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
            await _safe_reply(update, "Cancelled.")
            return True
        try:
            value = float(text)
            if not 0 <= value <= 100:
                raise ValueError
        except ValueError:
            await _safe_reply(update, "Please enter a number 0–100, or type *cancel*.",
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

# Handwriting/image recognition — Gemini Flash 2.5
# Used to extract text from handwritten photos and image documents.
def _ocr_image_bytes(image_bytes: bytes, mime: str) -> str:
    """Send an image to Gemini Flash for OCR and return the extracted text."""
    if not image_bytes or len(image_bytes) < 100:
        return f"OCR failed: image too small ({len(image_bytes)} bytes)"
    if not GEMINI_API_KEY:
        return "OCR failed: GEMINI_API_KEY not set."
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
        return f"OCR request failed: {e}"

def _extract_pdf_bytes(pdf_bytes: bytes) -> str:
    """Extract text from all pages of a PDF. Image-only pages are noted."""
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

#Document Parser
# Builds a structured index of questions in a document so we can extract just the relevant section when a student asks about "Question 2(a)".
def _build_question_index(doc_text: str) -> list[dict]:
    """
    Parse the document and return a list of question blocks, each with:
      id    — main question number (string)
      sub   — sub-question letter (string, or None)
      start — character offset in doc_text
      end   — character offset
      text  — the question text for that block
    """
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
    """
    Find the specific question section the student is asking about and return a structured prompt.
    """
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


# When a student sends a photo with no caption, try to detect the operation from the extracted text and route it to the correct symbolic engine.
def auto_route_extracted_text(extracted: str) -> str | None:
    """
    Attempt to convert OCR output into a structured command string that the normal text handler can process. Returns None if no pattern matches.
    """
    lower = extracted.lower()
    # Special case: periodic Fourier (Σ g(t-kT) pattern)
    if re.search(r'sum.*g\s*\(t', lower) and "fourier" in lower:
        g_def = re.search(r'g\s*\(\s*t\s*\)\s*=\s*([^\n,]+)', lower)
        T_def = re.search(r'[Tt]\s*=\s*([\d\.]+)', lower)
        g_part = g_def.group(1).strip() if g_def else "g(t)"
        T_part = T_def.group(1) if T_def else "2"
        return f"fourier transform of x(t) = sum g(t-{T_part}k), g(t) = {g_part}, T={T_part}"
    # Check for known operation patterns in the extracted text
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


# Session Llama Prompts
# All include _LATEX_INSTRUCTION which tells the LLM to write maths in plain unicoder that telegram can render
_LATEX_INSTRUCTION = (
    "FORMATTING RULE: Do NOT use LaTeX dollar-sign notation ($...$) anywhere in your response. "
    "Instead, write all mathematics in plain text using Unicode symbols directly:\n"
    "  - Use ω for omega, π for pi, δ for delta, τ for tau, α β γ θ φ λ σ μ for other Greek letters\n"
    "  - Use ∞ for infinity, ∑ for sum, ∫ for integral, → for arrow\n"
    "  - Use ^ for powers (e.g. e^(-st)), / for fractions (e.g. 1/(s+2))\n"
    "  - Write subscripts inline: omega_0, X_n, a_k\n"
    "  - Write fractions as num/den (e.g. ω/(s²+ω²))\n"
    "  - Use × for multiply where helpful, · for dot product\n"
    "  - Write equations on their own line for clarity\n"
    "Example: write   F(s) = 1/(s+2)   not   $F(s) = \\frac{1}{s+2}$"
)

_SESSION_RULES = (
    "IMPORTANT:\n"
    "1. The uploaded document is the PRIMARY source of truth.\n"
    "2. Use general knowledge only to explain/clarify — never to override the doc.\n"
    "3. If OCR looks garbled, say so and give your best interpretation.\n"
    f"4. {_LATEX_INSTRUCTION}\n"
    "5. Start your response DIRECTLY with the content. "
    "Do NOT begin with phrases like 'I'll be happy to help', 'Sure!', "
    "'Certainly!', 'Classification:', 'Question:', or 'Answer:'. "
    "Go straight to the explanation or solution."
)


def _prompt_explain_memo(doc_text: str, instruction: str) -> str:
    """Build a prompt that re-explains a memo solution with practice problems."""
    return (
        f"{_SESSION_RULES}\n\n"
        f"You are a patient, encouraging Signals and Systems tutor.\n\n"
        f"Uploaded memorandum / model solution:\n\"\"\"\n{doc_text}\n\"\"\"\n\n"
        f"Student instruction: {instruction}\n\n"
        f"Your task:\n"
        f"1. Identify the question or solution section the student is asking about.\n"
        f"2. Re-explain the solution step-by-step using simple language a first-year "
        f"student can follow. Number each step clearly.\n"
        f"3. For EVERY formula or mathematical symbol used, add a one-line plain-English "
        f"explanation of what it means and why it is applied at that step.\n"
        f"4. After the explanation, generate TWO similar practice problems with "
        f"different signal parameters. For each, state the problem clearly, then provide "
        f"the worked solution.\n"
        f"5. End with one short study tip specific to this technique.\n"
        f"Be concise but thorough. Use numbered steps. "
        f"Write all math in plain text / Unicode — NO LaTeX dollar signs."
    )


def _prompt_mark(memo_text: str, student_work: str) -> str:
    """Build a prompt that marks student work against a memo."""
    return (
        f"{_SESSION_RULES}\n\n"
        f"Memo / expected solution (from uploaded file):\n\"\"\"\n{memo_text}\n\"\"\"\n\n"
        f"Student's work:\n\"\"\"\n{student_work}\n\"\"\"\n\n"
        f"Your task:\n"
        f"1. OVERALL VERDICT: State one of CORRECT / PARTIALLY CORRECT / INCORRECT.\n"
        f"2. MARKS: Estimate marks earned out of the total available "
        f"(use the memo's mark allocation if visible, otherwise use your judgement).\n"
        f"3. WHAT IS CORRECT: List every step or element the student got right.\n"
        f"4. ERRORS: For each mistake, state:\n"
        f"   a) What the student wrote\n"
        f"   b) What it should be\n"
        f"   c) The concept they missed\n"
        f"5. IMPROVEMENT: Concise explanation of how to correct each error.\n"
        f"6. ENCOURAGEMENT: End with one specific, genuine encouragement statement.\n"
        f"Format each section clearly with its heading. "
        f"Write all math in plain text / Unicode — NO LaTeX dollar signs."
    )


def _prompt_solve(doc_text: str, instruction: str) -> str:
    """Build a prompt that solves a specific question from a document."""
    return (f"{_SESSION_RULES}\n\nDocument:\n\"\"\"\n{doc_text}\n\"\"\"\n\n"
            f"Student instruction: {instruction}\n\n"
            f"Respond directly. Numbered steps for math. Explain every formula. "
            f"Write all math in plain text / Unicode — NO LaTeX dollar signs.")

def _prompt_explain(doc_text: str, question: str) -> str:
    """Build a prompt for explaining a concept or question from a document."""
    return (f"{_SESSION_RULES}\n\nDocument:\n\"\"\"\n{doc_text}\n\"\"\"\n\n"
            f"Student question: {question}\n\n"
            f"FACTUAL: 2-4 sentences. CONCEPTUAL: paragraph + example. "
            f"CALCULATION: numbered steps. "
            f"Write all math in plain text / Unicode — NO LaTeX dollar signs.")


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
    for attempt in range(3):
        try:
            resp = requests.post(TOGETHER_ENDPOINT, json=payload, headers=headers, timeout=90)
            if resp.status_code in (503, 429):
                wait = 5 * (attempt + 1)
                print(f"[_call_llm] HTTP {resp.status_code} — retrying in {wait}s…")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except requests.exceptions.Timeout:
            print(f"[_call_llm] Timeout on attempt {attempt + 1}")
            time.sleep(5)
        except Exception as e:
            if attempt == 2:
                return f"LLM call failed after 3 attempts: {e}"
            time.sleep(5)
    return "Together AI is temporarily unavailable. Please try again in a minute."


def _route_session_prompt(doc_text: str, instruction: str) -> str:
    """
    Choose the right prompt template based on what the student seems to want
    """
    instr_lower = instruction.lower()
    if any(kw in instr_lower for kw in [
        "explain how", "explain question", "how was", "how is", "step by step",
        "walk me through", "practice", "similar example", "similar problem",
        "simplify", "what does this mean", "break down"
    ]):
        return _prompt_explain_memo(doc_text, instruction)
    if any(kw in instr_lower for kw in [
        "mark", "check", "compare", "correct", "feedback",
        "evaluate", "grade", "is my answer", "did i get", "how many marks"
    ]):
        return _prompt_mark(doc_text, instruction)
    if re.search(r'[Qq](?:uestion)?\s*\d+', instruction):
        ctx = extract_question_with_context(doc_text, instruction)
        if any(kw in instr_lower for kw in ["solve", "answer", "find", "compute",
                                             "calculate", "work out", "determine"]):
            return _prompt_solve(doc_text, ctx)
        return _prompt_explain(doc_text, ctx)
    if any(kw in instr_lower for kw in ["solve", "calculate", "find", "compute",
                                          "work out", "answer", "determine"]):
        return _prompt_solve(doc_text, instruction)
    return _prompt_explain(doc_text, instruction)


# Vector Store
# Loads PDFs from the knowledge base folder, splits them into overlapping chunks, embeds with MiniLM, and stores in ChromaDB on disk.
# On startup we check if the index already exists to skip re-embedding.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def build_vector_store_if_needed(pdf_folder: str, chroma_dir: str):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    if os.path.exists(chroma_dir) and os.listdir(chroma_dir):
        print("Loading existing ChromaDB…")
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

    print("Building ChromaDB from PDFs…")
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
    if not pdf_files:
        print("No PDFs found.")
        return None
    loader   = PyPDFDirectoryLoader(pdf_folder)
    docs     = loader.load()
    # Split on question boundaries first, then on paragraphs and lines, keeps question-answer pairs together for better retrieval
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, chunk_overlap=400,
        separators=["\n\nQuestion", "\n\nQ", "\n\n", "\n"]
    )
    chunks = splitter.split_documents(docs)
    print(f"   {len(docs)} pages → {len(chunks)} chunks")
    client = chromadb.PersistentClient(
        path=chroma_dir, settings=Settings(anonymized_telemetry=False))
    vs = Chroma.from_documents(chunks, embedding=embeddings, client=client)
    print(f"   {vs._collection.count()} vectors saved")
    return vs

# RAG Chain
# Wraps the vector store in a LangChain retrieval chain.
# The prompt instructs Llama to classify questions internally and respond in plain Unicode maths rather than LaTeX.
TUTOR_PROMPT = PromptTemplate.from_template(
    "You are a Signals and Systems tutor assistant.\n\n"
    "FORMATTING RULE: Do NOT use LaTeX dollar-sign notation ($...$) anywhere. "
    "Write all mathematics in plain text using Unicode symbols directly: "
    "ω for omega, π for pi, δ for delta, ∞ for infinity, ∑ for sum, ∫ for integral. "
    "Use ^ for powers, / for fractions. Write equations on their own line.\n\n"
    "Start your response DIRECTLY with the answer. "
    "Do NOT begin with 'I'll be happy to help', 'Sure!', 'Classification:', "
    "'Question:', 'Answer:', or any preamble.\n\n"
    "Classify the question type internally (do not state the classification):\n"
    "  A) FACTUAL → 2-4 sentences.\n"
    "  B) CONCEPTUAL → short paragraph + one example.\n"
    "  C) CALCULATION → numbered steps, explain every symbol.\n\n"
    "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
)

def format_docs(docs):
    """Combines retrieved document chunks into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)

def build_chains(vs):
    """Build the RAG retrieval + generation chain."""
    retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    llm = ChatTogether(model=LLM_MODEL, temperature=0.7, max_tokens=1024)
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | TUTOR_PROMPT | llm | StrOutputParser()
    )

# Build the vector store and chain at startup (or skip if no PDFs exist)
vector_store = build_vector_store_if_needed(PDF_FOLDER, CHROMA_DIR)
qa_chain     = None
if vector_store:
    qa_chain = build_chains(vector_store)
    print("RAG chain ready")
else:
    print("Running without knowledge base.")


# Telegram
async def send_long_code(update: Update, text: str) -> None:
    """Send text as a code block, splitting into 4090-char chunks if needed."""
    chunk_size = 4090
    for i in range(0, len(text), chunk_size):
        await _safe_reply(update, f"```\n{text[i:i+chunk_size]}\n```", parse_mode="Markdown")


# First welcome message on the telegram bot interface
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await _safe_reply(update,
        "Hi there! I'm your *Signals & Systems 1* tutor bot.\n\n"
        "I am here to help you understand any Signals and Systems 1 topic. "
        "I can help you with the following calculations:\n"
        "*Laplace Transforms*\n"
        "*Inverse Laplace Transforms*\n"
        "*Fourier Transform*\n"
        "*Inverse Fourier Transforms*\n"
        "*Convolution*\n"
        "*Plot signals*\n\n"
        "I can also help you calculate how much you need in your exam to pass. "
        "Ask me anything — I am here for you.\n"
        "Use /help for the full guide.",
        parse_mode="Markdown")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await _safe_reply(update,
        "*Full guide:*\n\n"
        "1. *Laplace*           _laplace of e^(-2*t)*u(t)_\n"
        "2. *Inverse Laplace*   _inverse laplace of 1/(s+2)_\n"
        "   Also: _ilt of s/(s^2+4)_\n"
        "3. *Fourier*           _fourier transform of e^(-t)*u(t)_\n"
        "   Also works without u(t): _fourier transform of e^(-2*t)*cos(2*t)_\n"
        "4. *Inverse Fourier*   _inverse fourier of 1/(1+j*omega)_\n"
        "   Also: _ift of 2*pi*DiracDelta(omega)_\n"
        "5. *Convolution*       _convolve e^(-t)*u(t) with u(t)_\n"
        "6. *Plot*              _plot 2*u(t-2)_  /  _draw u[n]-u[n-3]_\n"
        "7. *Upload PDF/image* then ask:\n"
        "     _explain how Question 1(b) was solved_\n"
        "     _answer Question 2(a)_\n"
        "     _mark my work against this memo_\n"
        "8. *Mark calculator* — _how much do I need to pass_\n\n"
        "Use * for multiply, ** for power\n"
        "e.g. e**(-2*t)*u(t)  or  e^-2t*u(t)\n\n"
        "Note: u(t) is optional for Fourier — e^(-2*t)*cos(2*t) is treated as causal.",
        parse_mode="Markdown")

#Period Extracter
def _extract_period(text: str) -> float | None:
    m = re.search(r'[Tt]\s*=\s*([0-9.]+\s*\*?\s*pi|[0-9.]+)', text)
    if not m:
        return None
    raw = m.group(1).replace(" ", "")
    if "pi" in raw:
        num = raw.replace("pi", "").replace("*", "") or "1"
        return float(num) * float(np.pi)
    return float(raw)

# Text handler
# Main routing logic, and Llama as the fallback
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    question = update.message.text.strip()
    q_lower  = question.lower()
    msg_id   = update.message.message_id
    chat_id  = update.effective_chat.id

    #Mark calculator 
    if await handle_mark_session(update, context):
        return

    # Document session
    # User uploaded a file last turn; this message is their instruction about it
    if session_has(chat_id):
        sess = session_get(chat_id)
        await _safe_reply(update,
            f"Working on it using *{sess['source']}* as reference…",
            parse_mode="Markdown")

        pending_work = context.user_data.pop("pending_student_work", None)
        if pending_work:
            # Student sent their own work photo + caption "mark this"
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

    # Inverse Laplace
    if is_inv_laplace(q_lower):
        if not _has_math_expr(question):
            # Conceptual question like "what is the inverse Laplace?" — use LLM
            await _llm_fallback(update, question, "Tutor Answer", msg_id)
            return
        expr_str = extract_expr(question)
        if not expr_str:
            await _safe_reply(update,
                "Please include an expression, e.g.:\n"
                "  _inverse laplace of 1/(s+2)_\n"
                "  _ilt of s/(s^2+4)_", parse_mode="Markdown")
            return
        await _safe_reply(update, "Computing Inverse Laplace transform…")
        steps, err = _build_inv_laplace_steps(expr_str)
        if err:
            await _llm_fallback(update, question, "Inverse Laplace Transform", msg_id)
        else:
            png = _render_math_png("Inverse Laplace Transform", steps, msg_id)
            if png and os.path.exists(png):
                success = await _safe_reply_photo(
                    update, png, f"Inverse Laplace of  F(s) = {expr_str}")
                if not success:
                    await _llm_fallback(update, question, "Inverse Laplace Transform", msg_id)
            else:
                await _llm_fallback(update, question, "Inverse Laplace Transform", msg_id)
        return

    # Inverse Fourier 
    if is_inv_fourier(q_lower):
        if not _has_math_expr(question):
            await _llm_fallback(update, question, "Tutor Answer", msg_id)
            return
        expr_str = extract_expr(question)
        if not expr_str:
            await _safe_reply(update,
                "Please include an expression, e.g.:\n"
                "  _inverse fourier of 1/(1+j*omega)_\n"
                "  _ift of 2*pi*DiracDelta(omega)_", parse_mode="Markdown")
            return
        await _safe_reply(update, "Computing Inverse Fourier transform…")
        steps, err = _build_inv_fourier_steps(expr_str)
        if err:
            await _llm_fallback(update, question, "Inverse Fourier Transform", msg_id)
        else:
            png = _render_math_png("Inverse Fourier Transform", steps, msg_id)
            if png and os.path.exists(png):
                success = await _safe_reply_photo(
                    update, png, f"Inverse Fourier of  F(ω) = {expr_str}")
                if not success:
                    await _llm_fallback(update, question, "Inverse Fourier Transform", msg_id)
            else:
                await _llm_fallback(update, question, "Inverse Fourier Transform", msg_id)
        return

    # Periodic Fourier
    # Must be checked before plain Fourier since it also contains Fourier keywords
    if is_periodic_fourier(q_lower):
        period_val = _extract_period(question) or 2.0
        g_def_q = re.search(r'g\s*\(\s*t\s*\)\s*=\s*([^,\n]+)', question, re.IGNORECASE)
        g_str = g_def_q.group(1).strip() if g_def_q else None
        if not g_str:
            await _safe_reply(update,
                "Please tell me what *g(t)* is, e.g.:\n"
                "  _fourier transform of x(t) = sum g(t-2k), g(t) = e^(-t)*u(t), T=2_",
                parse_mode="Markdown")
            return
        await _safe_reply(update,
            f"Computing periodic summation Fourier (g(t)={g_str}, T={period_val})…")
        steps, err = _build_periodic_fourier_steps(g_str, period_val)
        if err:
            await _llm_fallback(update, question, "Fourier Transform (Periodic)", msg_id)
        else:
            png = _render_math_png("Fourier Transform  (Periodic Summation)", steps, msg_id)
            if png and os.path.exists(png):
                await _safe_reply_photo(update, png,
                    f"Periodic Fourier: x(t)=sum g(t-{period_val}k), g(t)={g_str}, T={period_val}")
            else:
                await _llm_fallback(update, question, "Fourier Transform (Periodic)", msg_id)
        return

    # Laplace 
    if is_laplace(q_lower):
        if not _has_math_expr(question):
            await _llm_fallback(update, question, "Tutor Answer", msg_id)
            return
        expr_str = extract_expr(question)
        if not expr_str:
            await _safe_reply(update,
                "Please include an expression, e.g.:\n"
                "  _laplace of e^(-2*t)*u(t)_", parse_mode="Markdown")
            return
        await _safe_reply(update, "Computing Laplace transform…")
        steps, err = _build_laplace_steps(expr_str)
        if err:
            await _llm_fallback(update, question, "Laplace Transform", msg_id)
        else:
            png = _render_math_png("Laplace Transform", steps, msg_id)
            if png and os.path.exists(png):
                success = await _safe_reply_photo(
                    update, png, f"Laplace Transform of  f(t) = {expr_str}")
                if not success:
                    await _llm_fallback(update, question, "Laplace Transform", msg_id)
            else:
                await _llm_fallback(update, question, "Laplace Transform", msg_id)
        return

    #Fourier 
    if is_fourier(q_lower):
        if not _has_math_expr(question):
            await _llm_fallback(update, question, "Tutor Answer", msg_id)
            return
        expr_str = extract_expr(question)
        if not expr_str:
            await _safe_reply(update,
                "Please include an expression, e.g.:\n"
                "  _fourier transform of e^(-t)*u(t)_\n"
                "  _fourier transform of e^(-2*t)*cos(2*t)_", parse_mode="Markdown")
            return
        await _safe_reply(update, "Computing Fourier transform…")
        steps, err = _build_fourier_steps(expr_str)
        if err:
            await _llm_fallback(update, question, "Fourier Transform", msg_id)
        else:
            png = _render_math_png("Fourier Transform", steps, msg_id)
            if png and os.path.exists(png):
                success = await _safe_reply_photo(
                    update, png, f"Fourier Transform of  f(t) = {expr_str}")
                if not success:
                    await _llm_fallback(update, question, "Fourier Transform", msg_id)
            else:
                await _llm_fallback(update, question, "Fourier Transform", msg_id)
        return

    # Convolution
    if is_conv(q_lower):
        e1, e2 = _parse_two_signals(question)
        if not (e1 and e2):
            await _safe_reply(update,
                "Please specify both signals, e.g.:\n"
                "  _convolve e^(-t)*u(t) with u(t)_", parse_mode="Markdown")
            return
        await _safe_reply(update,
            f"Computing convolution of f(t)={e1} and g(t)={e2}…")
        steps, err, plot_path = _build_convolution_steps(e1, e2, msg_id)
        if err:
            await _llm_fallback(update, question, "Convolution", msg_id)
            return
        png = _render_math_png("Convolution  (f * g)(t)", steps, msg_id)
        if png and os.path.exists(png):
            success = await _safe_reply_photo(
                update, png, f"Convolution: f(t)={e1} * g(t)={e2}")
            if not success:
                await _llm_fallback(update, question, "Convolution", msg_id)
        else:
            await _llm_fallback(update, question, "Convolution", msg_id)
        if plot_path and os.path.exists(plot_path):
            await _safe_reply_photo(update, plot_path, "Numerical convolution (f * g)(t)")
        return

    # Plot 
    if is_plot(q_lower):
        await _safe_reply(update, "Generating plot…")
        fig_path = generate_plot(question, msg_id)
        if fig_path and os.path.exists(fig_path):
            success = await _safe_reply_photo(update, fig_path, f"{question}")
            if not success:
                await _llm_fallback(update, question, "Tutor Answer", msg_id)
        else:
            # Plot failed (expression couldn't be parsed) — fall through to LLM
            await _llm_fallback(update, question, "Tutor Answer", msg_id)
        return

    # Default: RAG / LLM fallback 
    if not qa_chain:
        await _safe_reply(update, "No knowledge base loaded.")
        return
    await _safe_reply(update, "Thinking…")
    try:
        answer = qa_chain.invoke(question)
        answer = _clean_llm_response(answer)
        answer = _latex_to_plain(answer)
        await send_llm_response(update, answer, "Tutor Answer", msg_id)
    except Exception as e:
        await _safe_reply(update, f"Something went wrong: {str(e)}")


# Handles photos sent directly (not as file attachments).
# If the user already has a document session open, the photo is treated as their work to be marked against the loaded document.
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    caption = (update.message.caption or "").strip()
    chat_id = update.effective_chat.id
    msg_id  = update.message.message_id

    await _safe_reply(update, "Got your photo — running OCR… (~15–30s)")

    import io
    photo_file  = await update.message.photo[-1].get_file()  # [-1] = highest resolution
    buf         = io.BytesIO()
    await photo_file.download_to_memory(buf)
    image_bytes = buf.getvalue()

    extracted = _ocr_image_bytes(image_bytes, "jpeg")
    await _safe_reply(update, f"Extracted:\n\n{extracted}")

    sess = session_get(chat_id)

    if caption:
        mark_keywords = ["mark", "check", "compare", "grade", "evaluate", "feedback"]
        if sess and any(kw in caption.lower() for kw in mark_keywords):
            # Mark the photo's work against the loaded document
            await _safe_reply(update,
                f"Marking against *{sess['source']}*…", parse_mode="Markdown")
            prompt   = _prompt_mark(sess["text"], extracted)
            response = _call_llm(prompt)
            response = _clean_llm_response(response)
            response = _latex_to_plain(response)
            await send_llm_response(update, response,
                                    f"Marking Feedback — {sess['source']}", msg_id)
            session_clear(chat_id)
            await _safe_reply(update, "_(Session cleared.)_", parse_mode="Markdown")
        else:
            # Process the photo using the session document as context
            doc_text = sess["text"] if sess else extracted
            source   = sess["source"] if sess else "handwritten photo"
            await _safe_reply(update,
                f"Processing using *{source}* as reference…",
                parse_mode="Markdown")
            prompt   = _route_session_prompt(doc_text, caption)
            response = _call_llm(prompt)
            response = _clean_llm_response(response)
            response = _latex_to_plain(response)
            await send_llm_response(update, response, f"Answer — {source}", msg_id)
            if sess:
                session_clear(chat_id)
                await _safe_reply(update, "_(Session cleared.)_", parse_mode="Markdown")
    else:
        # No caption — try to auto-detect the operation from OCR text
        routed_command = auto_route_extracted_text(extracted)
        if routed_command:
            await _safe_reply(update,
                f"Detected: `{routed_command[:120]}`\nSolving…",
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
                    await _llm_fallback(update, routed_command,
                                        "Inverse Laplace Transform", msg_id)
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
                    await _llm_fallback(update, routed_command,
                                        "Inverse Fourier Transform", msg_id)
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
                    await _llm_fallback(update, routed_command, "Laplace Transform", msg_id)
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
                    await _llm_fallback(update, routed_command, "Fourier Transform", msg_id)
                    return

            elif is_conv(q_lower):
                e1, e2 = _parse_two_signals(routed_command)
                if e1 and e2:
                    steps, err, plot_path = _build_convolution_steps(e1, e2, msg_id)
                    if not err:
                        png = _render_math_png("Convolution", steps, msg_id)
                        if png and os.path.exists(png):
                            await _safe_reply_photo(update, png, f"{e1} * {e2}")
                        if plot_path and os.path.exists(plot_path):
                            await _safe_reply_photo(update, plot_path,
                                                     "Numerical convolution")
                        return
                    await _llm_fallback(update, routed_command, "Convolution", msg_id)
                    return

            elif is_plot(q_lower):
                fig_path = generate_plot(routed_command, msg_id)
                if fig_path and os.path.exists(fig_path):
                    if await _safe_reply_photo(update, fig_path, f"{routed_command}"):
                        return

            # All routed attempts exhausted — fall back to LLM
            await _llm_fallback(update, extracted, "Tutor Answer", msg_id)

        else:
            # No operation detected — store as document session or prompt for instructions
            if sess:
                context.user_data["pending_student_work"] = extracted
                await _safe_reply(update,
                    f"I have *{sess['source']}* loaded.\n\n"
                    "Tell me what you'd like.",
                    parse_mode="Markdown")
            else:
                session_store(chat_id, extracted, "Handwritten photo / diagram")
                await _safe_reply(update,
                    "Photo loaded. What would you like me to do?\n"
                    "  - Solve this\n"
                    "  - Explain step by step\n"
                    "  - What is this question asking?\n"
                    "  - Describe this diagram")


# Handles PDF and image files sent as attachments , Extracts text (PDF: pypdf, image: Gemini OCR) and stores in session.
# If the user included a caption, acts on it immediately.
async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    doc       = update.message.document
    caption   = (update.message.caption or "").strip()
    chat_id   = update.effective_chat.id
    file_name = doc.file_name or "uploaded_file"
    extension = os.path.splitext(file_name)[-1].lower()

    SUPPORTED_IMAGES = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
    if extension not in {".pdf"} | SUPPORTED_IMAGES:
        await _safe_reply(update,
            "Unsupported file type. Please send a PDF or image (PNG, JPG, PDF).")
        return

    await _safe_reply(update,
        f"Received *{file_name}* — extracting content…",
        parse_mode="Markdown")

    # Download to a temp file then read bytes (avoids keeping large files in memory)
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
        f"Content loaded from *{source}*.\n\n"
        "Now tell me what you'd like:\n"
        "  - Explain how Question 1 was solved\n"
        "  - Answer Question 2(a)\n"
        "  - Verify if your work is correct",
        parse_mode="Markdown")

    # If the upload came with a caption, process it immediately
    if caption:
        sess   = session_get(chat_id)
        await _safe_reply(update,
            f"Acting on your caption: _{caption}_…",
            parse_mode="Markdown")
        prompt   = _route_session_prompt(sess["text"], caption)
        response = _call_llm(prompt)
        response = _clean_llm_response(response)
        response = _latex_to_plain(response)
        await send_llm_response(update, response, f"Answer — {source}",
                                msg_id=update.message.message_id)
        session_clear(chat_id)
        await _safe_reply(update, "_(Session cleared.)_", parse_mode="Markdown")

# MAIN — Bot startup
# Uses two separate HTTPXRequest objects so that long-polling (getUpdates) and outgoing sends don't share a connection pool and block each other.
async def main():
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

    print("Bot is running!")
    await app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    asyncio.run(main())

