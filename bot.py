import os
import asyncio
import nest_asyncio
import re
import base64
import requests
import tempfile
import textwrap

# ── Matplotlib non-interactive backend (must be set before pyplot import) ──────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_together import ChatTogether
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import SentenceTransformer

from telegram import Update
from telegram.ext import (
    Application, ApplicationBuilder, CommandHandler,
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

LLM_MODEL    = "meta-llama/Meta-Llama-3-8B-Instruct-Lite"

TOGETHER_ENDPOINT = "https://api.together.xyz/v1/chat/completions"

# ── Paths ─────────────────────────────────────────────────────────────────────
PDF_FOLDER  = "./knowledge_base"
CHROMA_DIR  = os.environ.get("CHROMA_DIR", "/data/chroma_db")
PLOT_FOLDER = "/tmp/plots"

os.makedirs(PDF_FOLDER,  exist_ok=True)
os.makedirs(CHROMA_DIR,  exist_ok=True)
os.makedirs(PLOT_FOLDER, exist_ok=True)

os.environ["TOGETHER_API_KEY"] = TOGETHER_API_KEY

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
    """SymPy rect (rectangular window) function."""
    return sp.Piecewise(
        (sp.Integer(1),      sp.Abs(x) < sp.Rational(1, 2)),
        (sp.Rational(1, 2),  sp.Eq(sp.Abs(x), sp.Rational(1, 2))),
        (sp.Integer(0),      True)
    )


def _normalise(expr: str) -> str:
    s = expr.strip()
    s = s.replace("^", "**").replace("{", "(").replace("}", ")")
    # rect() — keep as-is for sympify with our custom namespace
    s = re.sub(r'\brect\s*\(',                       'rect(',          s)
    s = re.sub(r'\bu\s*\(\s*t\s*\)',                 'Heaviside(t)',   s)
    s = re.sub(r'\bu\s*\(\s*t\s*([+-][^)]+)\)',      r'Heaviside(t\1)', s)
    s = re.sub(r'\bE\*\*(-[^\s\*\+\-\(\),]+)',       r'E**(\1)',       s)
    s = re.sub(r'\be\*\*(-[^\s\*\+\-\(\),]+)',       r'E**(\1)',       s)
    s = re.sub(r'(\d)(t\b)',                          r'\1*\2',         s)
    s = re.sub(r'(\d)(sin|cos|exp|sqrt|rect)',        r'\1*\2',         s)
    s = re.sub(r'(\d)\s*[eE]\*\*',                   r'\1*E**',        s)
    s = re.sub(r'\be\b',                              'E',              s)
    s = re.sub(r'\bu\s*[\(\[]',                       'Heaviside(',     s)
    s = re.sub(r'\]',                                 ')',               s)
    s = re.sub(r'\b(?:delta|δ)\s*[\(\[]',            'DiracDelta(',    s)
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
    s = re.sub(r'\bu\s*\[([^\]]+)\]',     r'UnitStep(\1)', s)
    s = re.sub(r'\b(?:delta|δ)\s*\[([^\]]+)\]', r'KronDelta(\1)', s)
    s = re.sub(r'(\d)(n\b)', r'\1*\2', s)
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


_SIGNAL_CHARS = ['(', '[', 't', 'n', 'sin', 'cos', 'exp',
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
    r'laplace\s+(?:transform\s+)?(?:of\s+)?|(?:inverse\s+)?laplace\s+(?:of\s+)?|'
    r'fourier\s+(?:transform\s+)?(?:of\s+)?|(?:inverse\s+)?fourier\s+(?:of\s+)?|'
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
    "laplace",
    "fourier",
    "convolution",
    "transform",
    "series",
    "plot",
    "signal",
    "derivation",
    "calculation",
    "equation",
    "answer —",        # session-doc answers (may contain heavy maths)
    "marking feedback",
    "tutor answer",    # RAG answers that contain step-by-step maths
)

# These titles are ALWAYS plain text regardless of content
_PLAIN_TEXT_TITLES = (
    "tutor answer",
)


def _is_math_title(title: str) -> bool:
    """
    Return True only when the response title indicates mathematical content
    that benefits from PNG rendering (transforms, convolutions, session docs).
    Plain Q&A / concept explanations return False → plain text reply.
    """
    t = title.lower()
    # Explicit plain-text overrides first
    if any(kw in t for kw in _PLAIN_TEXT_TITLES):
        return False
    return any(kw in t for kw in _MATH_TITLE_KEYWORDS)


# ══════════════════════════════════════════════════════════════════════════════
# LATEX / MATHTEXT SANITISER
# ══════════════════════════════════════════════════════════════════════════════

def _sanitise_mathtext(s: str) -> str:
    # ── Unicode symbols → LaTeX commands ──────────────────────────────────────
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
    # ── Heaviside theta notation ───────────────────────────────────────────────
    s = re.sub(r'\\theta\\left\(([^)]+)\\right\)', r'u(\1)', s)
    s = s.replace(r'\theta\left(t\right)', r'u(t)')
    # ── Imaginary unit i → j ──────────────────────────────────────────────────
    s = re.sub(r'(?<![a-zA-Z\\])i(?![a-zA-Z0-9{\\])', 'j', s)
    # ── Unsupported LaTeX envs → mathtext equivalents ─────────────────────────
    s = re.sub(r'\\mathcal\{([^}]+)\}',      r'\\mathbf{\1}', s)
    s = re.sub(r'\\mathscr\{([^}]+)\}',      r'\\mathbf{\1}', s)
    s = re.sub(r'\\mathrm\{([^}]+)\}',       r'\\rm \1',      s)
    s = re.sub(r'\\operatorname\{([^}]+)\}',  r'\\rm \1',      s)
    s = re.sub(r'\\text\{([^}]*)\}',          r'\\rm \1',      s)
    # ── Spacing macros ────────────────────────────────────────────────────────
    s = s.replace(r'\qquad', r'\ \ \ \ ')
    s = s.replace(r'\quad',  r'\ \ ')
    s = s.replace(r'\,',     r'\ ')
    s = s.replace(r'\;',     r'\ ')
    # ── Remove \left \right (not supported in mathtext) ───────────────────────
    s = re.sub(r'\\left\s*',  '', s)
    s = re.sub(r'\\right\s*', '', s)
    return s


# ══════════════════════════════════════════════════════════════════════════════
# LATEX MATH RENDERER
# ══════════════════════════════════════════════════════════════════════════════

def _sympy_to_latex(expr: sp.Expr) -> str:
    return sp.latex(expr)


def _try_render_row(ax, x_label: float, x_expr: float, y: float,
                    label: str, latex_str: str, fontsize: int = 22) -> None:
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
        n       = len(steps)
        FONT    = 22
        ROW_IN  = 0.95
        PAD_IN  = 1.6
        fig_h   = max(4.0, n * ROW_IN + PAD_IN)
        fig_w   = 16.0

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

        fig.tight_layout(pad=0.5)
        path = os.path.join(PLOT_FOLDER, f"math_{msg_id}.png")
        fig.savefig(path, dpi=180, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        plt.close("all")
        return path
    except Exception as e:
        print(f"[_render_math_png] FAILED: {e}")
        import traceback; traceback.print_exc()
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

    FIG_W     = 22.0
    PROSE_FS  = 22
    MATH_FS   = 28
    LINE_H    = 0.75
    MATH_H    = 1.10
    TITLE_H   = 1.10
    PAD       = 0.8

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
    if total_h > MAX_PAGE_H:
        # Split rows into pages and render each as a separate file
        pages = []
        page_rows = []
        page_h = TITLE_H + PAD
        for row in rows:
            kind, txt = row
            if kind == 'math':
                row_h = MATH_H
            else:
                n_lines = max(1, len(textwrap.wrap(txt, width=WRAP_WIDTH)) if txt.strip() else 1)
                row_h = LINE_H * n_lines
            if page_h + row_h > MAX_PAGE_H and page_rows:
                pages.append(page_rows)
                page_rows = [row]
                page_h = TITLE_H + PAD + row_h
            else:
                page_rows.append(row)
                page_h += row_h
        if page_rows:
            pages.append(page_rows)

        # Render only the first page and save; send remaining pages separately
        # by storing them in a temp attribute on the function
        render_response_png._extra_pages = []
        for pi, p_rows in enumerate(pages[1:], start=2):
            p_h = max(8.0, sum(
                MATH_H if k == 'math'
                else LINE_H * max(1, len(textwrap.wrap(t, width=WRAP_WIDTH)) if t.strip() else 1)
                for k, t in p_rows
            ) + TITLE_H + PAD)
            p_fig, p_ax = plt.subplots(figsize=(FIG_W, p_h), facecolor='white')
            p_ax.set_facecolor('white')
            p_ax.axis('off')
            p_ax.set_xlim(0, 1)
            p_ax.set_ylim(0, p_h)
            p_y = p_h - 0.15
            p_ax.text(0.5, p_y, f"{title}  (page {pi})",
                      fontsize=18, fontweight='bold', color='#1a1a2e',
                      ha='center', va='top', usetex=False)
            p_y -= TITLE_H
            p_ax.plot([0.02, 0.98], [p_y + 0.10, p_y + 0.10],
                      color='#cccccc', linewidth=1.0)
            for kind, txt in p_rows:
                if not txt.strip():
                    p_y -= LINE_H * 0.45
                    continue
                if kind == 'math':
                    safe = _sanitise_mathtext(txt)
                    try:
                        p_ax.text(0.06, p_y, f'${safe}$',
                                  fontsize=MATH_FS, color='#003388',
                                  va='top', ha='left', usetex=False)
                    except Exception:
                        p_ax.text(0.06, p_y, txt,
                                  fontsize=MATH_FS - 2, color='#444444',
                                  va='top', ha='left', fontfamily='monospace', usetex=False)
                    p_y -= MATH_H
                else:
                    display = re.sub(r'\*\*?([^*]+)\*\*?', r'\1', txt)
                    display = re.sub(r'__?([^_]+)__?', r'\1', display)
                    is_heading = bool(re.match(r'Step\s+\d+', txt.strip())
                                      or re.match(r'Problem\s+\d+', txt.strip()))
                    wrapped_lines = textwrap.wrap(display, width=WRAP_WIDTH) or [display]
                    p_ax.text(0.03, p_y, '\n'.join(wrapped_lines),
                              fontsize=PROSE_FS,
                              color='#1a1a2e' if is_heading else '#222222',
                              fontweight='bold' if is_heading else 'normal',
                              va='top', ha='left', usetex=False)
                    p_y -= LINE_H * len(wrapped_lines)
            p_fig.tight_layout(pad=0.3)
            p_path = os.path.join(PLOT_FOLDER, f'response_{msg_id}_p{pi}.png')
            p_fig.savefig(p_path, dpi=200, bbox_inches='tight',
                          facecolor='white', edgecolor='none')
            plt.close('all')
            render_response_png._extra_pages.append(p_path)

        rows = pages[0]
        total_h = max(8.0, sum(
            MATH_H if k == 'math'
            else LINE_H * max(1, len(textwrap.wrap(t, width=WRAP_WIDTH)) if t.strip() else 1)
            for k, t in rows
        ) + TITLE_H + PAD)
    else:
        render_response_png._extra_pages = []

    
    fig, ax = plt.subplots(figsize=(FIG_W, total_h), facecolor='white')
    ax.set_facecolor('white')
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, total_h)

    y = total_h - 0.15
    ax.text(0.5, y, title,
            fontsize=18, fontweight='bold', color='#1a1a2e',
            ha='center', va='top', usetex=False)
    y -= TITLE_H
    ax.plot([0.02, 0.98], [y + 0.10, y + 0.10],
            color='#cccccc', linewidth=1.0)

    INDENT      = 0.03
    MATH_INDENT = 0.06

    for kind, txt in rows:
        if not txt.strip():
            y -= LINE_H * 0.45
            continue

        if kind == 'math':
            safe = _sanitise_mathtext(txt)
            try:
                ax.text(MATH_INDENT, y, f'${safe}$',
                        fontsize=MATH_FS, color='#003388',
                        va='top', ha='left', usetex=False)
            except Exception:
                ax.text(MATH_INDENT, y, txt,
                        fontsize=MATH_FS - 2, color='#444444',
                        va='top', ha='left', fontfamily='monospace', usetex=False)
            y -= MATH_H
        else:
            display = re.sub(r'\*\*?([^*]+)\*\*?', r'\1', txt)
            display = re.sub(r'__?([^_]+)__?',      r'\1', display)
            is_heading = bool(re.match(r'\*\*', txt) or re.match(r'Step\s+\d+', txt.strip())
                              or re.match(r'Problem\s+\d+', txt.strip())
                              or re.match(r'Why it', txt.strip())
                              or re.match(r'Study Tip', txt.strip()))

            if display.strip():
                wrapped_lines = textwrap.wrap(display, width=WRAP_WIDTH)
                if not wrapped_lines:
                    wrapped_lines = [display]
            else:
                wrapped_lines = [display]

            wrapped_display = '\n'.join(wrapped_lines)
            n_wrapped = len(wrapped_lines)

            ax.text(INDENT, y, wrapped_display,
                    fontsize=PROSE_FS,
                    color='#1a1a2e' if is_heading else '#222222',
                    fontweight='bold' if is_heading else 'normal',
                    va='top', ha='left', usetex=False)
            y -= LINE_H * n_wrapped

    fig.tight_layout(pad=0.3)
    path = os.path.join(PLOT_FOLDER, f'response_{msg_id}.png')
    fig.savefig(path, dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close('all')
    return path


# ══════════════════════════════════════════════════════════════════════════════
# PHOTO SEND RETRY HELPER
# ══════════════════════════════════════════════════════════════════════════════

async def _send_photo_with_retry(update: Update, path: str, caption: str,
                                  retries: int = 3, delay: float = 5.0) -> bool:
    """
    Try to send a photo up to `retries` times with exponential back-off.
    Returns True on success, False if all attempts fail.
    """
    for attempt in range(1, retries + 1):
        try:
            with open(path, "rb") as f:
                await update.message.reply_photo(
                    photo=f,
                    caption=caption[:1024]
                )
            return True
        except Exception as exc:
            print(f"[send_photo] attempt {attempt}/{retries} failed: {exc}")
            if attempt < retries:
                await asyncio.sleep(delay * attempt)   # 5 s, 10 s, …
    return False


# ══════════════════════════════════════════════════════════════════════════════
# SEND LLM RESPONSE  (plain text by default; PNG only for maths/transforms)
# ══════════════════════════════════════════════════════════════════════════════

async def send_llm_response(update: Update, response_text: str,
                             title: str, msg_id: int,
                             force_image: bool = False) -> None:
    """
    Routing logic:
      • force_image=True  OR  _is_math_title(title) → attempt PNG render
      • otherwise → plain text reply_text (no image generated at all)

    If PNG rendering is attempted but the photo send fails after retries,
    the function falls back to plain text automatically.
    """
    if force_image or _is_math_title(title):
        png = render_response_png(response_text, title, msg_id)
        if png and os.path.exists(png):
            success = await _send_photo_with_retry(update, png, title)
            if success:
                for extra in getattr(render_response_png, '_extra_pages', []):
                    if os.path.exists(extra):
                        await _send_photo_with_retry(update, extra, f"{title} (cont.)")
                return
            # All retries exhausted — fall through to plain text
            print(f"[send_llm_response] photo send failed after retries, "
                  f"falling back to plain text for title='{title}'")

    # Plain-text path (default for general Q&A, and fallback for failed photos)
    for i in range(0, len(response_text), 4096):
        await update.message.reply_text(response_text[i:i + 4096])


# ══════════════════════════════════════════════════════════════════════════════
# LAPLACE STEP BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def _build_laplace_steps(expr_str: str) -> tuple[list[tuple[str, str]], str]:
    steps: list[tuple[str, str]] = []
    try:
        f      = parse_ct_expr(expr_str)
        f_tex  = _sympy_to_latex(f)
        steps.append(("Input",      rf"f(t) = {f_tex}"))
        steps.append(("Definition", r"F(s) = \int_{0}^{\infty} f(t)\,e^{-st}\,dt"))

        rule = _identify_laplace_rule_latex(f)
        steps.append(("Rule / Form", rule))

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

        result     = sp.laplace_transform(f, t_sym, s_sym, noconds=True)
        result     = sp.simplify(result)
        result_tex = _sympy_to_latex(result)
        steps.append(("Result", rf"F(s) = {result_tex}"))

        return steps, ""
    except Exception as e:
        return [], f"❌ Could not compute Laplace transform: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# FOURIER STEP BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def _build_fourier_steps(expr_str: str) -> tuple[list[tuple[str, str]], str]:
    steps: list[tuple[str, str]] = []
    try:
        f      = parse_ct_expr(expr_str)
        f_tex  = _sympy_to_latex(f)
        steps.append(("Input",      rf"f(t) = {f_tex}"))
        steps.append(("Definition", r"F(\omega) = \int_{-\infty}^{\infty} f(t)\, e^{-j\omega t}\, dt"))

        rule = _identify_fourier_rule_latex(f)
        steps.append(("Known pair", rule))

        result     = _fourier_direct(f)
        result_tex = _sympy_to_latex(result).replace(r"\mathbf{i}", "j").replace(" i ", " j ").replace("{i}", "{j}")
        result_tex = re.sub(r'(?<![a-zA-Z])i(?![a-zA-Z])', 'j', result_tex)
        steps.append(("Result", rf"F(\omega) = {result_tex}"))

        return steps, ""
    except Exception as e:
        return [], f"❌ Could not compute Fourier transform: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# PERIODIC SUMMATION FOURIER STEP BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def _build_periodic_fourier_steps(
    g_expr_str: str,
    period: float = 2.0,
) -> tuple[list[tuple[str, str]], str]:
    steps: list[tuple[str, str]] = []
    try:
        g      = parse_ct_expr(g_expr_str)
        g_tex  = _sympy_to_latex(g)
        T_sym  = sp.Rational(period).limit_denominator(1000)
        w0     = 2 * sp.pi / T_sym
        w0_tex = _sympy_to_latex(w0)

        steps.append((
            "Signal",
            rf"x(t) = \sum_{{k=-\infty}}^{{\infty}} g(t - {_sympy_to_latex(T_sym)}k)"
            rf",\quad g(t) = {g_tex}"
        ))
        steps.append((
            "Property",
            rf"x(t)=\sum_k g(t-kT) \;\xrightarrow{{\mathcal{{F}}}}\;"
            rf"X(\omega)=\omega_0\sum_{{n=-\infty}}^{{\infty}}"
            rf"G(n\omega_0)\,\delta(\omega-n\omega_0)"
        ))
        steps.append((
            "Fund. freq.",
            rf"\omega_0 = \frac{{2\pi}}{{T}} = \frac{{2\pi}}{{{_sympy_to_latex(T_sym)}}} "
            rf"= {w0_tex}\ \mathrm{{rad/s}}"
        ))
        steps.append((
            "Definition",
            r"G(\omega) = \int_{-\infty}^{\infty} g(t)\,e^{-j\omega t}\,dt"
        ))

        try:
            G_result = _fourier_direct(g)
            G_tex    = _sympy_to_latex(G_result)
            G_tex    = re.sub(r'(?<![a-zA-Z\\])i(?![a-zA-Z0-9{])', 'j', G_tex)
            steps.append(("G(ω)", rf"G(\omega) = {G_tex}"))
        except Exception:
            steps.append((
                "G(ω)",
                rf"G(\omega) = \int_{{-\infty}}^{{\infty}} {g_tex}\,e^{{-j\omega t}}\,dt"
            ))

        steps.append((
            "Substitute",
            rf"X(\omega) = {w0_tex}\sum_{{n=-\infty}}^{{\infty}}"
            rf"G(n\cdot {w0_tex})\,\delta(\omega - n\cdot {w0_tex})"
        ))
        steps.append((
            "Result",
            rf"X(\omega) = \omega_0\sum_{{n=-\infty}}^{{\infty}}"
            rf"G(n\omega_0)\,\delta(\omega-n\omega_0)"
        ))
        steps.append((
            "Note",
            rf"\text{{Discrete spectral lines at multiples of }}"
            rf"\omega_0={w0_tex}\ \text{{rad/s}}"
        ))

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

        a0     = sp.simplify(sp.integrate(f, (t_sym, 0, T)) / T)
        steps.append(("a0 (DC)", rf"a_0 = {_sympy_to_latex(a0)}"))

        for k in range(1, n_terms + 1):
            try:
                an = sp.simplify(
                    2 * sp.integrate(f * sp.cos(k * w0 * t_sym), (t_sym, 0, T)) / T)
                bn = sp.simplify(
                    2 * sp.integrate(f * sp.sin(k * w0 * t_sym), (t_sym, 0, T)) / T)
                steps.append((
                    f"n = {k}",
                    rf"a_{{{k}}}={_sympy_to_latex(an)},\quad b_{{{k}}}={_sympy_to_latex(bn)}"
                ))
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


def _build_convolution_steps(expr1_str: str, expr2_str: str,
                              msg_id: int) -> tuple[list[tuple[str, str]], str, str | None]:
    steps: list[tuple[str, str]] = []
    plot_path = None

    try:
        f = parse_ct_expr(expr1_str)
        g = parse_ct_expr(expr2_str)
    except Exception as e:
        return [], f"❌ Could not parse signals: {e}", None

    steps.append(("f(t)",       _sympy_to_latex(f)))
    steps.append(("g(t)",       _sympy_to_latex(g)))
    steps.append(("Definition", r"(f\star g)(t)=\int_{-\infty}^{\infty}f(\tau)\,g(t-\tau)\,d\tau"))

    f_tau   = f.subs(t_sym, tau)
    g_shift = g.subs(t_sym, t_sym - tau)
    integrand = sp.expand(f_tau * g_shift)

    steps.append(("Substitution",
                   rf"f(\tau)={_sympy_to_latex(f_tau)},\quad g(t-\tau)={_sympy_to_latex(g_shift)}"))
    steps.append(("Integrand",   _sympy_to_latex(integrand)))

    causal_limits = _compute_causal_limits(f, g)

    if causal_limits is not None:
        lower, upper = causal_limits
        limits = (tau, lower, upper)
        lower_tex = _sympy_to_latex(lower)
        upper_tex = _sympy_to_latex(upper)
      
    else:
        limits = (tau, -sp.oo, sp.oo)
       

    try:
        result = sp.integrate(integrand, limits)
        result = sp.simplify(result)
        result = _simplify_heaviside_powers(result)
        if result.has(sp.Integral):
            raise ValueError("unevaluated integral")
        steps.append(("Result", rf"(f\star g)(t) = {_sympy_to_latex(result)}"))
        plot_path = _numerical_convolution_plot(f, g, msg_id)
    except Exception as e:
        steps.append(("Note", rf"\text{{Symbolic integration failed: {str(e)[:60]}}}"))
        steps.append(("Fallback", r"\text{See numerical plot}"))
        plot_path = _numerical_convolution_plot(f, g, msg_id)

    return steps, "", plot_path


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL PLOTTER
# ══════════════════════════════════════════════════════════════════════════════
PLOT_KEYWORDS     = ["plot", "draw", "graph", "sketch", "show me",
                     "visualise", "visualize", "diagram"]
DISCRETE_KEYWORDS = ["x[n]", "u[n]", "delta[n]", "δ[n]", "h[n]", "y[n]", "[n]"]


def is_discrete(question: str) -> bool:
    return any(kw in question for kw in DISCRETE_KEYWORDS)


def _lambdify_ct(expr: sp.Expr):
    return sp.lambdify(
        t_sym, expr,
        modules=["numpy", {
            "Heaviside":  lambda x: np.where(np.asarray(x, float) >= 0, 1., 0.),
            "DiracDelta": lambda x: np.zeros_like(np.asarray(x, float)),
            "rect": lambda x: np.where(np.abs(x) < 0.5, 1.0,
                              np.where(np.abs(x) == 0.5, 0.5, 0.0)),
        }]
    )


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
        y_vals    = evaluator(n_vals)
        y_vals    = np.clip(y_vals, -10, 10)
    except Exception as e:
        print(f"[plot_dt] {e}"); return None
    fig, ax = plt.subplots(figsize=(10, 3))
    ml, sl, _ = ax.stem(n_vals, y_vals, linefmt="steelblue",
                         markerfmt="o", basefmt="k-")
    ml.set_markersize(5)
    sl.set_linewidth(1.5)
    ax.axhline(0, color="k", lw=.6)
    ax.set(xlabel="n", ylabel="x[n]", title=f"x[n] = {expr_str}")
    ax.grid(True, alpha=.3)
    ax.set_xticks(n_vals[::2])
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
    ax.set_xlim(t0 - 3, t0 + 3)
    ax.set_ylim(-0.2, 1.5)
    label = f"δ(t − {t0})" if t0 != 0 else "δ(t)"
    ax.set(title=label, xlabel="t", ylabel="δ(t)")
    ax.grid(True, alpha=.4)
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
# LAPLACE TRANSFORM — rule identifiers & plain-text
# ══════════════════════════════════════════════════════════════════════════════
def _identify_laplace_rule(expr: sp.Expr) -> str:
    s = str(expr)
    if "DiracDelta" in s:
        return "Unit impulse:   L{δ(t)} = 1"
    if "Heaviside" in s and "exp" not in s and "sin" not in s and "cos" not in s:
        return "Unit step:      L{u(t)} = 1/s"
    if "exp" in s and "sin" not in s and "cos" not in s:
        return "Damped exponential: L{e^{-at}f(t)} = F(s+a)"
    if "sin" in s:
        return "Sine:           L{sin(ωt)u(t)} = ω/(s²+ω²)"
    if "cos" in s:
        return "Cosine:         L{cos(ωt)u(t)} = s/(s²+ω²)"
    if str(expr) == str(t_sym):
        return "Ramp:           L{t·u(t)} = 1/s²"
    if expr == sp.Integer(1):
        return "Impulse (constant 1): L{δ(t)} = 1"
    return "General transform pair / integration by parts"


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


def compute_laplace(expr_str: str) -> str:
    lines = ["━━━ 📐 LAPLACE TRANSFORM ━━━\n"]
    lines.append(f"Input:  f(t) = {expr_str}\n")
    lines.append("Definition:  F(s) = ∫₀^∞  f(t) · e^(-st) dt\n")
    try:
        f = parse_ct_expr(expr_str)
    except Exception as e:
        return (f"❌ Could not parse expression: {e}\n"
                f"Try: e**(-2*t)*Heaviside(t)  or  sin(3*t)*u(t)")
    rule = _identify_laplace_rule(f)
    lines.append(f"Step 1 — Recognise the signal form:\n   → {rule}\n")
    args = sp.Add.make_args(f)
    if len(args) > 1:
        lines.append("Step 2 — Apply linearity  L{af + bg} = aF(s) + bG(s):")
        for term in args:
            try:
                r = sp.laplace_transform(term, t_sym, s_sym, noconds=True)
                lines.append(f"   L{{{sp.pretty(term)}}} = {sp.pretty(r)}")
            except Exception:
                lines.append(f"   L{{{sp.pretty(term)}}} = (could not evaluate)")
        lines.append("")
    else:
        lines.append("Step 2 — Single term, applying transform directly.\n")
    try:
        result = sp.laplace_transform(f, t_sym, s_sym, noconds=True)
        result = sp.simplify(result)
        lines.append(f"Step 3 — Final result:\n   F(s) = {sp.pretty(result)}\n")
        if "exp" in str(f):
            lines.append("ROC hint: Re(s) > -a  (right-sided exponential signal)")
        elif "Heaviside" in str(f):
            lines.append("ROC hint: Re(s) > 0  (causal signal)")
        lines.append(f"\n✅  F(s) = {sp.pretty(result)}")
    except Exception as e:
        lines.append(f"❌ SymPy could not find a closed form: {e}")
        lines.append("   Try splitting the expression or simplifying first.")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# FOURIER TRANSFORM — rule identifiers & plain-text
# ══════════════════════════════════════════════════════════════════════════════
def _identify_fourier_rule(expr: sp.Expr) -> str:
    s = str(expr)
    if "DiracDelta" in s:
        return "Impulse:   F{δ(t)} = 1"
    if "Heaviside" in s and "exp" not in s:
        return "Step:      F{u(t)} = πδ(ω) + 1/jω"
    if "exp" in s and "sin" not in s and "cos" not in s:
        return "Damped exp: F{e^{-at}u(t)} = 1/(a+jω)  [a>0]"
    if "sin" in s:
        return "Sine:      F{sin(ω₀t)} = jπ[δ(ω+ω₀)−δ(ω−ω₀)]"
    if "cos" in s:
        return "Cosine:    F{cos(ω₀t)} = π[δ(ω+ω₀)+δ(ω−ω₀)]"
    if expr == sp.Integer(1):
        return "Constant 1: F{1} = 2πδ(ω)"
    return "General definition: F(ω) = ∫₋∞^∞ f(t)e^{-jωt} dt"


def _identify_fourier_rule_latex(expr: sp.Expr) -> str:
    s = str(expr)
    if "DiracDelta" in s:
        return r"\mathcal{F}\{\delta(t)\} = 1"
    if "Heaviside" in s and "exp" not in s:
        return r"\mathcal{F}\{u(t)\} = \pi\delta(\omega) + \frac{1}{j\omega}"
    if "exp" in s and "sin" not in s and "cos" not in s:
        return r"\mathcal{F}\{e^{-at}u(t)\} = \frac{1}{a+j\omega},\quad a>0"
    if "sin" in s:
        return r"\mathcal{F}\{\sin(\omega_0 t)\} = j\pi[\delta(\omega+\omega_0)-\delta(\omega-\omega_0)]"
    if "cos" in s:
        return r"\mathcal{F}\{\cos(\omega_0 t)\} = \pi[\delta(\omega+\omega_0)+\delta(\omega-\omega_0)]"
    if expr == sp.Integer(1):
        return r"\mathcal{F}\{1\} = 2\pi\delta(\omega)"
    # rect-based pair
    if "Piecewise" in str(expr):
        return r"\mathcal{F}\{\mathrm{rect}(t/\tau)\} = \tau\,\mathrm{sinc}(\omega\tau/2)"
    return r"F(\omega) = \int_{-\infty}^{\infty} f(t)\,e^{-j\omega t}\,dt"


def _fourier_direct(f: sp.Expr) -> sp.Expr:
    try:
        result = sp.integrate(
            f * sp.exp(-sp.I * w_sym * t_sym),
            (t_sym, -sp.oo, sp.oo)
        )
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
            numeric_factors = []
            other_factors   = []
            for arg in f.args:
                if arg.is_number:
                    numeric_factors.append(arg)
                else:
                    other_factors.append(arg)
            if numeric_factors:
                coeff = sp.Mul(*numeric_factors)
        pair = sp.pi * sp.DiracDelta(w_sym) + 1 / (sp.I * w_sym)
        return sp.simplify(coeff * pair)

    if f == sp.Integer(1):
        return 2 * sp.pi * sp.DiracDelta(w_sym)

    raise ValueError("No closed-form pair found")


def compute_fourier(expr_str: str) -> str:
    lines = ["━━━ 📡 FOURIER TRANSFORM ━━━\n"]
    lines.append(f"Input:  f(t) = {expr_str}\n")
    lines.append("Definition:  F(ω) = ∫₋∞^∞  f(t) · e^(-jωt) dt\n")
    try:
        f = parse_ct_expr(expr_str)
    except Exception as e:
        return (f"❌ Could not parse expression: {e}\n"
                f"Try: e**(-t)*u(t)  or  cos(2*t)")

    rule = _identify_fourier_rule(f)
    lines.append(f"Step 1 — Recognise the signal form:\n   → {rule}\n")

    args = sp.Add.make_args(f)
    if len(args) > 1:
        lines.append("Step 2 — Apply linearity  F{af + bg} = aF(ω) + bG(ω):")
        for term in args:
            try:
                r = _fourier_direct(term)
                lines.append(f"   F{{{sp.pretty(term)}}} = {sp.pretty(r)}")
            except Exception:
                lines.append(f"   F{{{sp.pretty(term)}}} = (could not evaluate term)")
        lines.append("")
    else:
        lines.append("Step 2 — Single term, applying transform directly.\n")

    try:
        result = _fourier_direct(f)
        lines.append(f"Step 3 — Final result:\n   F(ω) = {sp.pretty(result)}\n")
        if "Heaviside" in str(f) and "exp" in str(f):
            lines.append("Magnitude |F(ω)|: low-pass characteristic — decays as 1/ω")
        elif "cos" in str(f) or "sin" in str(f):
            lines.append("Note: Result contains Dirac deltas — spectrum consists of discrete lines.")
        lines.append(f"\n✅  F(ω) = {sp.pretty(result)}")
    except Exception as e:
        lines.append(f"❌ Could not evaluate: {e}")
        lines.append("   Tip: for causal signals include u(t), e.g. cos(2*t)*u(t)")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# FOURIER SERIES — plain-text
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


def compute_fourier_series(expr_str: str, period: float, n_terms: int = 5) -> str:
    lines = ["━━━ 🎵 FOURIER SERIES ━━━\n"]
    lines.append(f"Input:   f(t) = {expr_str}")
    lines.append(f"Period:  T = {period:.4g}\n")
    try:
        f = parse_ct_expr(expr_str)
    except Exception as e:
        return f"❌ Could not parse expression: {e}"
    T  = sp.Rational(period).limit_denominator(1000)
    w0 = 2 * sp.pi / T
    lines.append(f"Step 1 — Fundamental frequency:\n   ω₀ = 2π/T = {sp.pretty(w0)} rad/s\n")
    lines.append("Step 2 — Coefficient formulas:")
    lines.append("   a₀ = (1/T) ∫₀ᵀ f(t) dt")
    lines.append("   aₙ = (2/T) ∫₀ᵀ f(t)·cos(nω₀t) dt")
    lines.append("   bₙ = (2/T) ∫₀ᵀ f(t)·sin(nω₀t) dt\n")
    try:
        a0 = sp.simplify(sp.integrate(f, (t_sym, 0, T)) / T)
        lines.append(f"Step 3 — DC coefficient:\n   a₀ = {sp.pretty(a0)}\n")
    except Exception:
        lines.append("Step 3 — a₀ could not be computed symbolically.\n")
    lines.append(f"Step 4 — First {n_terms} harmonics:")
    for k in range(1, n_terms + 1):
        try:
            an = sp.simplify(
                2 * sp.integrate(f * sp.cos(k * w0 * t_sym), (t_sym, 0, T)) / T)
            bn = sp.simplify(
                2 * sp.integrate(f * sp.sin(k * w0 * t_sym), (t_sym, 0, T)) / T)
            lines.append(f"   n={k}:  a_{k} = {sp.pretty(an)},   b_{k} = {sp.pretty(bn)}")
        except Exception:
            lines.append(f"   n={k}:  could not evaluate")
    try:
        series = sp.fourier_series(f, (t_sym, 0, T))
        trunc  = series.truncate(n_terms)
        lines.append(f"\nStep 5 — Truncated series (first {n_terms} terms):")
        lines.append(f"   f(t) ≈ {sp.pretty(trunc)}")
        lines.append(f"\n✅  Series computed successfully.")
    except Exception as e:
        lines.append(f"\n⚠️  Full series truncation failed: {e}")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# CONVOLUTION — plain-text fallback
# ══════════════════════════════════════════════════════════════════════════════
def _parse_two_signals(text: str):
    m = re.split(r'\bwith\b|\band\b|\*|\bstar\b', text, maxsplit=1,
                 flags=re.IGNORECASE)
    if len(m) == 2:
        e1 = re.sub(r'^.*?(?:convolve|convolution\s+of|f\s*=|f\(t\)\s*=)\s*',
                    '', m[0], flags=re.IGNORECASE).strip()
        e2 = re.sub(r'^.*?(?:g\s*=|g\(t\)\s*=)\s*', '',
                    m[1], flags=re.IGNORECASE).strip()
        e1 = extract_expr(e1) or e1
        e2 = extract_expr(e2) or e2
        return e1, e2
    return None, None


def _numerical_convolution_plot(f_expr: sp.Expr, g_expr: sp.Expr,
                                 msg_id: int) -> str | None:
    try:
        t_vals = np.linspace(-2, 20, 5000)
        dt     = t_vals[1] - t_vals[0]
        f_lam  = _lambdify_ct(f_expr)
        g_lam  = _lambdify_ct(g_expr)
        f_vals = np.real(f_lam(t_vals)).astype(float)
        g_vals = np.real(g_lam(t_vals)).astype(float)
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


def compute_convolution(expr1_str: str, expr2_str: str, msg_id: int = 0):
    lines = ["━━━ 🔁 CONVOLUTION ━━━\n"]
    lines.append(f"f(t) = {expr1_str}")
    lines.append(f"g(t) = {expr2_str}\n")
    lines.append("Definition:  (f ★ g)(t) = ∫₋∞^∞  f(τ) · g(t−τ) dτ\n")
    try:
        f = parse_ct_expr(expr1_str)
        g = parse_ct_expr(expr2_str)
    except Exception as e:
        return f"❌ Could not parse one or both signals: {e}", None
    lines.append("Step 1 — Substitute τ into f(t) and (t−τ) into g(t):")
    f_tau     = f.subs(t_sym, tau)
    g_shift   = g.subs(t_sym, t_sym - tau)
    integrand = sp.expand(f_tau * g_shift)
    lines.append(f"   f(τ)      = {sp.pretty(f_tau)}")
    lines.append(f"   g(t−τ)    = {sp.pretty(g_shift)}")
    lines.append(f"   Integrand = {sp.pretty(integrand)}\n")

    causal_limits = _compute_causal_limits(f, g)
    if causal_limits is not None:
        lower, upper = causal_limits
        limits = (tau, lower, upper)
        lines.append(f"Step 2 — Integration limits from Heaviside support: τ ∈ [{lower}, {upper}]\n")
    else:
        limits = (tau, -sp.oo, sp.oo)
        lines.append("Step 2 — Integrate over the full real line (-∞, ∞).\n")

    plot_path = None

    try:
        result = sp.integrate(integrand, limits)
        result = sp.simplify(result)
        result = _simplify_heaviside_powers(result)
        if result.has(sp.Integral):
            raise ValueError("SymPy returned unevaluated integral")
        lines.append("Step 3 — Evaluate the integral:")
        lines.append(f"   (f ★ g)(t) = {sp.pretty(result)}\n")
        lines.append(f"✅  Result: (f ★ g)(t) = {sp.pretty(result)}")
        plot_path = _numerical_convolution_plot(f, g, msg_id)
        if plot_path:
            lines.append("\n📊  Numerical plot attached for verification.")
    except Exception as e:
        lines.append(f"Step 3 — Symbolic integration failed: {e}")
        lines.append("   → Falling back to numerical convolution (plot attached).\n")
        plot_path = _numerical_convolution_plot(f, g, msg_id)
        if plot_path:
            lines.append("📊  Numerical (f ★ g)(t) plotted successfully.")
        else:
            lines.append("❌  Numerical fallback also failed. Check the expressions.")
    return "\n".join(lines), plot_path


# ══════════════════════════════════════════════════════════════════════════════
# KEYWORD ROUTERS
# ══════════════════════════════════════════════════════════════════════════════
LAPLACE_KEYS = ["laplace", "l transform", "l{", "laplace transform",
                "laplace of", "inverse laplace"]
FOURIER_KEYS = ["fourier transform", "ft{", "fourier of", "f transform",
                "ft of", "compute ft", "find ft", "f.t. of", "fourier tf",
                "inverse fourier"]
FS_KEYS      = ["fourier series", "periodic signal", "series of"]
CONV_KEYS    = ["convolution", "convolve", "f*g", "f★g", "f star g"]

PERIODIC_FOURIER_KEYS = [
    "sum", "summation", "periodic summation", "x(t) =", "x(t)=",
    "k=-inf", "k = -inf", "g(t-2k)", "g(t -", "poisson",
]


def is_laplace(q: str) -> bool:          return any(k in q for k in LAPLACE_KEYS)
def is_fourier(q: str) -> bool:          return any(k in q for k in FOURIER_KEYS)
def is_fs(q: str)      -> bool:          return any(k in q for k in FS_KEYS)
def is_conv(q: str)    -> bool:          return any(k in q for k in CONV_KEYS)
def is_plot(q: str)    -> bool:          return any(k in q for k in PLOT_KEYWORDS)


def is_periodic_fourier(q: str) -> bool:
    has_fourier   = any(k in q for k in FOURIER_KEYS)
    has_summation = any(k in q for k in PERIODIC_FOURIER_KEYS)
    return has_fourier and has_summation


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
    weighted = (test_avg          * WEIGHTS["tests"] +
                data["labs"]      * WEIGHTS["labs"]  +
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
        lines.append(f"❌ You'd need {req_exam:.1f}% — mathematically impossible. "
                     f"Give it your best anyway!")
    else:
        lines.append(f"🎯 *You need at least {req_exam:.1f}% in the exam to pass.*")
        if req_exam <= 50:
            lines.append("💪 Very achievable — keep it up!")
        elif req_exam <= 70:
            lines.append("📚 Tough but doable with a solid plan!")
        else:
            lines.append("⚠️ Hard work required — start now!")
    return "\n".join(lines)


async def handle_mark_session(update: Update,
                               context: ContextTypes.DEFAULT_TYPE) -> bool:
    chat_id = update.effective_chat.id
    text    = update.message.text.strip()
    if is_mark_trigger(text) and chat_id not in mark_sessions:
        mark_sessions[chat_id] = {"step": "test1", "data": {}}
        await update.message.reply_text(
            "Let's calculate what you need to pass!\n\n" + STEP_PROMPTS["test1"],
            parse_mode="Markdown")
        return True
    if chat_id in mark_sessions:
        session = mark_sessions[chat_id]
        step    = session["step"]
        if text.lower() in ["cancel", "stop", "quit", "exit"]:
            del mark_sessions[chat_id]
            await update.message.reply_text("❌ Cancelled.")
            return True
        try:
            value = float(text)
            if not 0 <= value <= 100:
                raise ValueError
        except ValueError:
            await update.message.reply_text(
                "⚠️ Enter a number 0–100, or type *cancel*.", parse_mode="Markdown")
            return True
        session["data"][step] = value
        idx = STEPS_MC.index(step)
        if idx + 1 < len(STEPS_MC):
            nxt             = STEPS_MC[idx + 1]
            session["step"] = nxt
            await update.message.reply_text(STEP_PROMPTS[nxt], parse_mode="Markdown")
        else:
            result = compute_result(session["data"])
            del mark_sessions[chat_id]
            await update.message.reply_text(result, parse_mode="Markdown")
        return True
    return False


# ══════════════════════════════════════════════════════════════════════════════
# OCR — Gemini Flash
# ══════════════════════════════════════════════════════════════════════════════
def _ocr_image_bytes(image_bytes: bytes, mime: str) -> str:
    if not image_bytes or len(image_bytes) < 100:
        return f"❌ OCR failed: image data is empty or too small ({len(image_bytes)} bytes)"

    if not GEMINI_API_KEY:
        return "❌ OCR failed: GEMINI_API_KEY environment variable is not set."

    mime = mime.lower().lstrip(".")
    if mime == "jpg":
        mime = "jpeg"
    if mime not in ("jpeg", "png", "webp", "gif"):
        mime = "jpeg"

    b64 = base64.b64encode(image_bytes).decode("utf-8")

    prompt_text = (
        "You are an expert at reading handwritten academic work and engineering diagrams.\n\n"
        "First, determine what this image primarily contains:\n"
        "  A) HANDWRITTEN TEXT / EQUATIONS — mathematical working, written answers, equations\n"
        "  B) DRAWN DIAGRAM — block diagram, signal flow graph, circuit, system diagram, "
        "graph/plot, flowchart, or any drawn structure\n"
        "  C) BOTH — handwritten text alongside a drawn diagram\n\n"
        "Then respond accordingly:\n\n"
        "If A: Transcribe ALL handwritten text and mathematical expressions exactly as written. "
        "Preserve equations, symbols, numbering, and layout.\n\n"
        "If B: Describe the diagram STRUCTURALLY in detail:\n"
        "  - What type of diagram is it? (block diagram, signal flow graph, etc.)\n"
        "  - List every block/node with its label\n"
        "  - List every connection/arrow: from → to, and any label on the connection\n"
        "  - Note any summing junctions, branch points, or special symbols\n"
        "  - State any transfer functions, gains, or labels shown\n"
        "  - Identify input(s) and output(s)\n\n"
        "If C: Do both — transcribe the text first, then describe the diagram structure.\n\n"
        "Output ONLY the transcribed/described content. No commentary, no preamble."
    )

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt_text},
                    {
                        "inline_data": {
                            "mime_type": f"image/{mime}",
                            "data": b64
                        }
                    }
                ]
            }
        ]
    }

    url = (
        f"https://generativelanguage.googleapis.com/v1/models"
        f"/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
    )

    print(f"[OCR] Sending {len(image_bytes)} bytes to Gemini Flash")

    try:
        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        data    = resp.json()
        content = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        print(f"[OCR] Success — extracted {len(content)} characters")
        return content
    except requests.exceptions.HTTPError as e:
        body = e.response.text[:300] if e.response is not None else ""
        print(f"[OCR] HTTP error: {body}")
        return f"❌ OCR request failed: {e} — {body}"
    except Exception as e:
        print(f"[OCR] Error: {e}")
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
        if text:
            parts.append(f"--- Page {i} ---\n{text}")
        else:
            parts.append(f"--- Page {i} --- [image-only page, no selectable text]")
    return "\n\n".join(parts) if parts else "[No text could be extracted from this PDF]"


# ══════════════════════════════════════════════════════════════════════════════
# DOCUMENT PARSER
# ══════════════════════════════════════════════════════════════════════════════
_Q_MAIN = re.compile(
    r'(?:^|\n)\s*(?:Question|Q\.?)\s*(\d+)\b[^\n]*',
    re.IGNORECASE
)
_Q_SUB  = re.compile(
    r'(?:^|\n)\s*(?:(\d+)[\.\)]\s*([a-zA-Z])[\.\)]|'
    r'\(([a-zA-Z])\)|'
    r'([a-zA-Z])[\.\)])',
    re.IGNORECASE
)


def _build_question_index(doc_text: str) -> list[dict]:
    blocks = []
    lines  = doc_text.split('\n')
    spans  = []
    char_pos = 0

    for line in lines:
        stripped = line.strip()
        m_main = re.match(
            r'^(?:Question|Q\.?)\s*(\d+)\b',
            stripped, re.IGNORECASE
        )
        if m_main:
            spans.append((char_pos, m_main.group(1), None))

        m_sub = re.match(
            r'^\(?([a-zA-Z]{1,2}|[ivxlIVXL]+)\)?[\.\)]\s',
            stripped
        )
        if m_sub and spans:
            spans.append((char_pos, spans[-1][1] if spans else None,
                           m_sub.group(1).lower()))

        char_pos += len(line) + 1

    for i, (start, qid, sub) in enumerate(spans):
        end   = spans[i + 1][0] if i + 1 < len(spans) else len(doc_text)
        text  = doc_text[start:end].strip()
        blocks.append({'id': str(qid) if qid else None,
                       'sub': sub,
                       'start': start,
                       'end':   end,
                       'text':  text})

    return blocks


def extract_question_with_context(doc_text: str, instruction: str) -> str:
    m = re.search(
        r'[Qq](?:uestion)?\s*(\d+)\s*[\.\(]?\s*([a-zA-Z])?',
        instruction
    )
    if not m:
        return instruction

    target_q   = m.group(1)
    target_sub = m.group(2).lower() if m.group(2) else None

    blocks   = _build_question_index(doc_text)
    q_blocks = [b for b in blocks if b['id'] == target_q]
    if not q_blocks:
        return instruction

    preamble_blocks = [b for b in q_blocks if b['sub'] is None]
    preamble_text   = "\n\n".join(b['text'] for b in preamble_blocks).strip()

    if target_sub is None:
        full_text = "\n\n".join(b['text'] for b in q_blocks).strip()
        return (
            f"The student is asking about Question {target_q}.\n\n"
            f"--- Extracted question text ---\n{full_text}\n\n"
            f"--- Student instruction ---\n{instruction}"
        )

    sub_blocks = [b for b in q_blocks if b['sub'] == target_sub]
    if not sub_blocks:
        context = preamble_text or "(No preamble found)"
        return (
            f"The student is asking about Question {target_q}({target_sub}).\n\n"
            f"--- Question {target_q} preamble / definitions ---\n{context}\n\n"
            f"--- Student instruction ---\n{instruction}\n\n"
            f"Note: The specific sub-question text could not be found. "
            f"Use the preamble and the student's instruction to answer."
        )

    sub_text = "\n\n".join(b['text'] for b in sub_blocks).strip()
    parts = [f"The student is asking about Question {target_q}({target_sub})."]
    if preamble_text:
        parts.append(
            f"--- Question {target_q} preamble / definitions "
            f"(definitions and given information) ---\n{preamble_text}"
        )
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
        return (
            f"fourier transform of x(t) = sum g(t-{T_part}k), "
            f"g(t) = {g_part}, T={T_part}"
        )

    m_lap = re.search(
        r'(?:find|compute|determine|calculate)?\s*'
        r'(?:the\s+)?laplace\s+(?:transform\s+)?(?:of\s+)?(.+)',
        lower, re.IGNORECASE | re.DOTALL
    )
    if m_lap:
        expr = m_lap.group(1).strip().split('\n')[0].strip(' .')
        return f"laplace of {expr}"

    m_ilap = re.search(
        r'inverse\s+laplace\s+(?:transform\s+)?(?:of\s+)?(.+)',
        lower, re.IGNORECASE | re.DOTALL
    )
    if m_ilap:
        expr = m_ilap.group(1).strip().split('\n')[0].strip(' .')
        return f"inverse laplace of {expr}"

    m_ft = re.search(
        r'(?:find|compute|determine|calculate)?\s*'
        r'(?:the\s+)?fourier\s+transform\s+(?:of\s+)?(.+)',
        lower, re.IGNORECASE | re.DOTALL
    )
    if m_ft:
        expr = m_ft.group(1).strip().split('\n')[0].strip(' .')
        return f"fourier transform of {expr}"

    m_fs = re.search(
        r'fourier\s+series\s+(?:of\s+)?(.+)',
        lower, re.IGNORECASE | re.DOTALL
    )
    if m_fs:
        expr = m_fs.group(1).strip().split('\n')[0].strip(' .')
        return f"fourier series of {expr}"

    m_conv = re.search(
        r'(?:find|compute|determine)?\s*(?:the\s+)?convolution\s+(?:of\s+)?(.+)',
        lower, re.IGNORECASE | re.DOTALL
    )
    if m_conv:
        expr = m_conv.group(1).strip().split('\n')[0].strip(' .')
        return f"convolve {expr}"

    m_plot = re.search(
        r'(?:sketch|plot|draw|graph)\s+(?:the\s+signal\s+)?(.+)',
        lower, re.IGNORECASE | re.DOTALL
    )
    if m_plot:
        expr = m_plot.group(1).strip().split('\n')[0].strip(' .')
        return f"plot {expr}"

    return None


# ══════════════════════════════════════════════════════════════════════════════
# SESSION-AWARE LLM PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

_LATEX_INSTRUCTION = (
    "FORMATTING RULE: Wrap ALL mathematical expressions, equations, and formulas "
    "using LaTeX notation: $...$ for inline math and $$...$$ for display/standalone equations. "
    "For example, write $G(\\omega) = \\frac{{1}}{{a + j\\omega}}$ not G(omega) = 1/(a+jw). "
    "Use standard LaTeX commands: \\frac{{}}{{}}, \\int, \\sum, \\omega, \\delta, \\pi, "
    "\\mathcal{{L}}, \\mathcal{{F}}, e^{{-st}}, etc. "
    "CRITICAL: For summations always write $$\\sum_{{k=-\\infty}}^{{\\infty}}$$ NOT the "
    "unicode character ∑ or ∞ outside of math delimiters."
)

_SESSION_RULES = (
    "IMPORTANT — follow strictly:\n"
    "1. The uploaded document is the PRIMARY and authoritative source of truth.\n"
    "2. Use your general Signals & Systems knowledge ONLY to explain or clarify —\n"
    "   never to contradict, override, or replace the uploaded content.\n"
    "3. For correctness, marking, and solution verification rely exclusively on\n"
    "   the uploaded document.\n"
    "4. If OCR output looks garbled in a critical spot, say so and give your best\n"
    "   interpretation — do not silently substitute your own answer.\n"
    f"5. {_LATEX_INSTRUCTION}"
)


def _prompt_explain_memo(doc_text: str, instruction: str) -> str:
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
        f"Wrap all math in $...$ or $$...$$. "
        f"Always write summation limits as $$\\sum_{{k=-\\infty}}^{{\\infty}}$$ in LaTeX."
    )


def _prompt_mark(memo_text: str, student_work: str) -> str:
    return (
        f"{_SESSION_RULES}\n\n"
        f"Memo / expected solution (from uploaded file):\n\"\"\"\n{memo_text}\n\"\"\"\n\n"
        f"Student's work:\n\"\"\"\n{student_work}\n\"\"\"\n\n"
        f"Your task:\n"
        f"1. OVERALL VERDICT: State one of CORRECT / PARTIALLY CORRECT / INCORRECT.\n"
        f"2. MARKS: Estimate marks earned out of the total available "
        f"(use the memo's mark allocation if visible, otherwise use your judgement).\n"
        f"3. WHAT IS CORRECT: List every step or element the student got right. "
        f"Acknowledge partial credit where the approach is sound but execution is flawed.\n"
        f"4. ERRORS: For each mistake, state:\n"
        f"   a) What the student wrote\n"
        f"   b) What it should be\n"
        f"   c) The concept they missed\n"
        f"5. IMPROVEMENT: Give a concise explanation of how to correct each error, "
        f"referencing the memo solution.\n"
        f"6. ENCOURAGEMENT: End with one specific, genuine encouragement statement "
        f"based on what the student demonstrated.\n"
        f"Format each section clearly with its heading. "
        f"Wrap all math in $...$ or $$...$$."
    )


def _prompt_solve(doc_text: str, instruction: str) -> str:
    return (
        f"{_SESSION_RULES}\n\n"
        f"Uploaded document:\n\"\"\"\n{doc_text}\n\"\"\"\n\n"
        f"Student instruction: {instruction}\n\n"
        f"Respond directly. Use numbered steps where maths is involved. "
        f"Explain every formula and symbol. "
        f"Wrap all math in $...$ or $$...$$."
    )


def _prompt_explain(doc_text: str, question: str) -> str:
    return (
        f"{_SESSION_RULES}\n\n"
        f"Uploaded document:\n\"\"\"\n{doc_text}\n\"\"\"\n\n"
        f"Student question: {question}\n\n"
        f"Classify your answer silently:\n"
        f"  FACTUAL     → 2-4 sentences.\n"
        f"  CONCEPTUAL  → short paragraph + one example.\n"
        f"  CALCULATION → numbered steps, explain every symbol.\n"
        f"Answer based on the uploaded document. Use general knowledge only to aid clarity. "
        f"Wrap all math in $...$ or $$...$$."
    )


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
        resp = requests.post(
            TOGETHER_ENDPOINT, json=payload, headers=headers, timeout=90)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"❌ LLM call failed: {e}"


def _route_session_prompt(doc_text: str, instruction: str) -> str:
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
        contextual_instruction = extract_question_with_context(doc_text, instruction)
        if any(kw in instr_lower for kw in ["solve", "answer", "find", "compute",
                                             "calculate", "work out", "determine"]):
            return _prompt_solve(doc_text, contextual_instruction)
        return _prompt_explain(doc_text, contextual_instruction)

    if any(kw in instr_lower for kw in ["solve", "calculate", "find", "compute",
                                          "work out", "answer", "determine"]):
        return _prompt_solve(doc_text, instruction)

    return _prompt_explain(doc_text, instruction)


# ══════════════════════════════════════════════════════════════════════════════
# VECTOR STORE
# ══════════════════════════════════════════════════════════════════════════════
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def build_vector_store_if_needed(pdf_folder: str, chroma_dir: str):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    if os.path.exists(chroma_dir) and os.listdir(chroma_dir):
        print("✅ Loading existing ChromaDB from volume...")
        vs    = Chroma(persist_directory=chroma_dir, embedding_function=embeddings)
        count = vs._collection.count()
        print(f"   {count} vectors loaded.")
        if count > 0:
            return vs
        print("   Index was empty — rebuilding...")
    print("📄 Building ChromaDB from PDFs in knowledge_base/...")
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
    if not pdf_files:
        print("⚠️  No PDFs found — bot will answer without context.")
        return None
    loader   = PyPDFDirectoryLoader(pdf_folder)
    docs     = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, chunk_overlap=400,
        separators=["\n\nQuestion", "\n\nQ", "\n\n", "\n"]
    )
    chunks = splitter.split_documents(docs)
    print(f"   {len(docs)} pages → {len(chunks)} chunks")
    vs = Chroma.from_documents(chunks, embedding=embeddings,
                               persist_directory=chroma_dir)
    vs.persist()
    print(f"   ✅ {vs._collection.count()} vectors saved to {chroma_dir}")
    return vs


# ══════════════════════════════════════════════════════════════════════════════
# RAG CHAINS
# ══════════════════════════════════════════════════════════════════════════════

TUTOR_PROMPT = PromptTemplate.from_template(
    "You are a Signals and Systems tutor assistant.\n\n"
    "FORMATTING RULE: Wrap ALL mathematical expressions, equations, and formulas "
    "using LaTeX notation: $...$ for inline math and $$...$$ for display equations. "
    "For example, write $G(\\omega) = \\frac{{1}}{{a + j\\omega}}$ not G(omega) = 1/(a+jw). "
    "Use standard LaTeX commands: \\frac{{}}{{}}, \\int, \\sum, \\omega, \\delta, \\pi, "
    "\\mathcal{{L}}, \\mathcal{{F}}, e^{{-st}}, etc. "
    "CRITICAL: Always write summation limits as $$\\sum_{{k=-\\infty}}^{{\\infty}}$$ "
    "in LaTeX — never use unicode ∑ or ∞ outside of $...$ delimiters.\n\n"
    "First, silently classify the student's question into one of three types:\n"
    "  A) FACTUAL — asking for course info, dates, definitions, or simple facts\n"
    "  B) CONCEPTUAL — asking to understand an idea, theorem, or technique\n"
    "  C) CALCULATION — asking to solve or explain a problem or work through math\n\n"
    "Then respond according to the type:\n"
    "  A) FACTUAL -> Answer in 2-4 sentences. No steps, no examples.\n"
    "  B) CONCEPTUAL -> Explain clearly in a short paragraph of 4-6 sentences. "
    "Give exactly one simple example. No numbered steps.\n"
    "  C) CALCULATION -> Work through or explain it step by step with numbered steps. "
    "Explain every formula and symbol used.\n\n"
    "Do not mention the classification in your response. Just answer.\n"
    "If the answer is not in the context, say so honestly.\n\n"
    "Context from course documents:\n{context}\n\n"
    "Student question:\n{question}\n\n"
    "Answer:"
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_chains(vs):
    retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    llm = ChatTogether(model=LLM_MODEL, temperature=0.7, max_tokens=1024)

    def make_rag(prompt):
        return (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt | llm | StrOutputParser()
        )

    return make_rag(TUTOR_PROMPT)


vector_store = build_vector_store_if_needed(PDF_FOLDER, CHROMA_DIR)
qa_chain     = None

if vector_store:
    qa_chain = build_chains(vector_store)
    print("✅ RAG chain ready (read-only)")
else:
    print("⚠️  Running without knowledge base — add PDFs to knowledge_base/ and redeploy.")

scorer = SentenceTransformer(EMBEDDING_MODEL)


# ══════════════════════════════════════════════════════════════════════════════
# TELEGRAM UTILITY
# ══════════════════════════════════════════════════════════════════════════════
async def send_long(update: Update, text: str) -> None:
    for i in range(0, len(text), 4096):
        await update.message.reply_text(text[i:i + 4096])


async def send_long_code(update: Update, text: str) -> None:
    chunk_size = 4090
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        await update.message.reply_text(f"```\n{chunk}\n```", parse_mode="Markdown")


# ══════════════════════════════════════════════════════════════════════════════
# COMMAND HANDLERS
# ══════════════════════════════════════════════════════════════════════════════
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Hi! I'm your *Signals & Systems* tutor bot.\n\n"
        "Here's what I can do:\n\n"
        "📚 *Q&A* — ask any course question\n"
        "📊 *Plot signals* (continuous & discrete)\n"
        "   _plot 2*u(t-2)_,  _draw u[n] - u[n-3]_\n\n"
        "📐 *Laplace Transform*\n"
        "   _laplace of e^(-2*t)*u(t)_\n\n"
        "📡 *Fourier Transform*\n"
        "   _fourier transform of e^(-t)*u(t)_\n\n"
        "📡 *Periodic Summation Fourier Transform*\n"
        "   _fourier transform of x(t)=sum g(t-2k), g(t)=e^(-t)*u(t), T=2_\n\n"
        "🎵 *Fourier Series*\n"
        "   _fourier series of t, T=2_\n\n"
        "🔁 *Convolution*\n"
        "   _convolve u(t) with e^(-t)*u(t)_\n\n"
        "📷 *Upload a photo* — handwritten work, diagrams, or question images\n"
        "   With caption → I do what you ask\n"
        "   No caption   → I read the question and solve it automatically\n\n"
        "📄 *Upload a PDF or image file* — question paper, memo, textbook\n"
        "   Then ask:\n"
        "   _explain how Question 1 was solved_\n"
        "   _answer Question 1(a)_\n"
        "   _mark my work against this memo_\n"
        "   _give me practice examples like Question 2_\n\n"
        "🧮 *Exam mark calculator* — _how much do I need to pass_\n\n"
        "Use /help for the full guide.",
        parse_mode="Markdown"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🆘 *Full guide:*\n\n"
        "1. *Laplace Transform*\n"
        "   _laplace of e^(-2*t)*u(t)_\n\n"
        "2. *Fourier Transform*\n"
        "   _fourier transform of e^(-t)*u(t)_\n\n"
        "3. *Periodic Summation Fourier*\n"
        "   _fourier transform of x(t)=sum g(t-2k), g(t)=e^(-t)*u(t), T=2_\n\n"
        "4. *Fourier Series*\n"
        "   _fourier series of t**2, T=2*pi_\n\n"
        "5. *Convolution*\n"
        "   _convolve e^(-t)*u(t) with u(t)_\n\n"
        "6. *Plot signals*\n"
        "   Continuous: _plot 2*u(t-2)_  or  _show me what u(t) looks like_\n"
        "   Discrete:   _draw u[n]-u[n-3]_\n\n"
        "7. *Upload a PDF or image file*\n"
        "   Send the file then follow up with:\n"
        "     _explain how Question 1(b) was solved_\n"
        "     _answer Question 2(a)_\n"
        "     _give me two practice problems like Question 3_\n"
        "     _mark my work against this memo_\n"
        "   Uploaded files are session-only — not saved permanently.\n\n"
        "8. *Handwritten photo or question image*\n"
        "   With caption → bot follows your instruction\n"
        "   No caption   → bot reads the question and solves it automatically\n"
        "   Diagrams (block diagrams, signal flow graphs) described structurally\n\n"
        "9. *Mark calculator*\n"
        "   _how much do I need to pass_\n\n"
        "Use * for multiply, ** for power\n"
        "e.g. e**(-2*t)*u(t)  or  e^-2t*u(t)",
        parse_mode="Markdown"
    )


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

        pending_work = context.user_data.pop("pending_student_work", None)
        if pending_work:
            await update.message.reply_text(
                f"⏳ Marking against *{sess['source']}*…", parse_mode="Markdown")
            prompt   = _prompt_mark(sess["text"], pending_work)
            response = _call_llm(prompt)
            title    = f"Marking Feedback — {sess['source']}"
            await send_llm_response(update, response, title, msg_id)
            session_clear(chat_id)
            await update.message.reply_text(
                "_(Session cleared — uploaded file no longer in memory.)_",
                parse_mode="Markdown")
            return

        await update.message.reply_text(
            f"⏳ Working on it using *{sess['source']}* as reference…",
            parse_mode="Markdown")
        prompt   = _route_session_prompt(sess["text"], question)
        response = _call_llm(prompt)
        title    = f"Answer — {sess['source']}"
        await send_llm_response(update, response, title, msg_id)
        session_clear(chat_id)
        await update.message.reply_text(
            "_(Session cleared — uploaded file no longer in memory.)_",
            parse_mode="Markdown")
        return

    # ── 3. Math tools ─────────────────────────────────────────────────────────

    # ── Periodic summation Fourier ────────────────────────────────────────────
    if is_periodic_fourier(q_lower):
        period_val = _extract_period(question) or 2.0

        sess  = session_get(chat_id)
        g_str = None

        if sess:
            g_def = re.search(
                r'g\s*\(\s*t\s*\)\s*=\s*([^\n,]+)',
                sess["text"], re.IGNORECASE
            )
            if g_def:
                g_str = g_def.group(1).strip()

        if not g_str:
            g_def_q = re.search(
                r'g\s*\(\s*t\s*\)\s*=\s*([^,\n]+)',
                question, re.IGNORECASE
            )
            g_str = g_def_q.group(1).strip() if g_def_q else None

        if not g_str:
            await update.message.reply_text(
                "⚠️ I can see you want the Fourier transform of a periodic summation.\n\n"
                "Please also tell me what *g(t)* is, e.g.:\n"
                "  _fourier transform of x(t) = sum g(t-2k), g(t) = e^(-t)*u(t), T=2_",
                parse_mode="Markdown"
            )
            return

        await update.message.reply_text(
            f"⏳ Computing Fourier transform of periodic summation "
            f"(g(t) = {g_str}, T = {period_val})…"
        )
        steps, err = _build_periodic_fourier_steps(g_str, period_val)
        if err:
            await update.message.reply_text(err)
        else:
            png = _render_math_png(
                "Fourier Transform  (Periodic Summation)", steps, msg_id
            )
            if png and os.path.exists(png):
                success = await _send_photo_with_retry(
                    update, png,
                    f"Fourier Transform of  x(t) = Σ g(t − {period_val}k)\n"
                    f"where  g(t) = {g_str},  T = {period_val}"
                )
                if not success:
                    lines = [
                        "📡 Fourier Transform — Periodic Summation\n",
                        f"x(t) = Σ g(t − {period_val}k),   g(t) = {g_str}",
                        f"T = {period_val},   ω₀ = 2π/{period_val} = "
                        f"{2 * 3.14159 / period_val:.4g} rad/s\n",
                        "Property:  X(ω) = ω₀ Σ G(nω₀) · δ(ω − nω₀)\n",
                    ]
                    try:
                        g_sym = parse_ct_expr(g_str)
                        G_sym = _fourier_direct(g_sym)
                        lines.append(f"G(ω) = {sp.pretty(G_sym)}\n")
                    except Exception:
                        lines.append("G(ω) = (see definition above)\n")
                    lines.append("✅  X(ω) = ω₀ Σ_n G(n·ω₀) · δ(ω − n·ω₀)")
                    await send_long_code(update, "\n".join(lines))
        return

    # ── Laplace ───────────────────────────────────────────────────────────────
    if is_laplace(q_lower):
        expr_str = extract_expr(question)
        if not expr_str:
            await update.message.reply_text(
                "⚠️ Please include an expression, e.g.:\n"
                "  _laplace of e^(-2*t)*u(t)_", parse_mode="Markdown")
            return
        await update.message.reply_text("⏳ Computing Laplace transform…")
        steps, err = _build_laplace_steps(expr_str)
        if err:
            await update.message.reply_text(err)
        else:
            png = _render_math_png("Laplace Transform", steps, msg_id)
            if png and os.path.exists(png):
                success = await _send_photo_with_retry(
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
            await update.message.reply_text(
                "⚠️ Please include an expression, e.g.:\n"
                "  _fourier transform of e^(-t)*u(t)_", parse_mode="Markdown")
            return
        await update.message.reply_text("⏳ Computing Fourier transform…")
        steps, err = _build_fourier_steps(expr_str)
        if err:
            await update.message.reply_text(err)
        else:
            png = _render_math_png("Fourier Transform", steps, msg_id)
            if png and os.path.exists(png):
                success = await _send_photo_with_retry(
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
            await update.message.reply_text(
                "⚠️ Please include the period, e.g.:\n"
                "  _fourier series of t, T=2_", parse_mode="Markdown")
            return
        expr_str = extract_expr(question)
        if not expr_str:
            expr_str = re.sub(r',?\s*[Tt]\s*=\s*[^\s]+', '', question).strip()
            expr_str = extract_expr(expr_str) or expr_str
        await update.message.reply_text(
            f"⏳ Computing Fourier series for f(t)={expr_str}, T={period:.4g}…")
        steps, err = _build_fourier_series_steps(expr_str, period)
        if err:
            await update.message.reply_text(err)
        else:
            png = _render_math_png("Fourier Series", steps, msg_id)
            if png and os.path.exists(png):
                success = await _send_photo_with_retry(
                    update, png,
                    f"Fourier Series of  f(t) = {expr_str},  T = {period:.4g}")
                if not success:
                    await send_long_code(update, compute_fourier_series(expr_str, period))
            else:
                await send_long_code(update, compute_fourier_series(expr_str, period))
        return

    # ── Convolution ───────────────────────────────────────────────────────────
    if is_conv(q_lower):
        e1, e2 = _parse_two_signals(question)
        if not (e1 and e2):
            await update.message.reply_text(
                "⚠️ Please specify both signals, e.g.:\n"
                "  _convolve e^(-t)*u(t) with u(t)_", parse_mode="Markdown")
            return
        await update.message.reply_text(
            f"⏳ Computing convolution of  f(t)={e1}  and  g(t)={e2}…")
        steps, err, plot_path = _build_convolution_steps(e1, e2, msg_id)
        if err:
            await update.message.reply_text(err)
        else:
            png = _render_math_png("Convolution  (f ★ g)(t)", steps, msg_id)
            if png and os.path.exists(png):
                success = await _send_photo_with_retry(
                    update, png, f"Convolution:  f(t)={e1}  ★  g(t)={e2}")
                if not success:
                    text_result, plot_path2 = compute_convolution(e1, e2, msg_id)
                    await send_long_code(update, text_result)
                    plot_path = plot_path or plot_path2
            else:
                text_result, plot_path2 = compute_convolution(e1, e2, msg_id)
                await send_long_code(update, text_result)
                plot_path = plot_path or plot_path2
            if plot_path and os.path.exists(plot_path):
                await _send_photo_with_retry(
                    update, plot_path, "📊 Numerical convolution  (f ★ g)(t)")
        return

    # ── Plot ──────────────────────────────────────────────────────────────────
    if is_plot(q_lower):
        await update.message.reply_text("📊 Generating plot…")
        fig_path = generate_plot(question, msg_id)
        if fig_path and os.path.exists(fig_path):
            success = await _send_photo_with_retry(update, fig_path, f"📈 {question}")
            if not success:
                await update.message.reply_text(
                    "⚠️ Plot was generated but could not be sent. Please try again.")
        else:
            await update.message.reply_text(
                "⚠️ Could not parse that expression.\n"
                "Examples:\n  _plot 2*u(t-2)_\n  _draw u[n] - u[n-3]_\n"
                "  _show me what u(t) looks like_",
                parse_mode="Markdown")
        return

    # ── 4. General tutor Q&A via RAG chain ────────────────────────────────────
    if not qa_chain:
        await update.message.reply_text(
            "⚠️ No knowledge base loaded yet. Please ask the administrator.")
        return
    await update.message.reply_text("🤔 Thinking…")
    try:
        answer = qa_chain.invoke(question)
        # General Q&A always returns plain text (title="Tutor Answer" → _is_math_title=False)
        await send_llm_response(update, answer, "Tutor Answer", msg_id)
    except Exception as e:
        await update.message.reply_text(f"❌ Something went wrong: {str(e)}")


# ══════════════════════════════════════════════════════════════════════════════
# PHOTO HANDLER
# ══════════════════════════════════════════════════════════════════════════════
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    caption = (update.message.caption or "").strip()
    chat_id = update.effective_chat.id
    msg_id  = update.message.message_id

    await update.message.reply_text(
        "📷 Got your photo — running handwriting/diagram recognition… (~15–30s)")

    import io
    photo_file  = await update.message.photo[-1].get_file()
    buf         = io.BytesIO()
    await photo_file.download_to_memory(buf)
    image_bytes = buf.getvalue()
    print(f"[handle_photo] Downloaded {len(image_bytes)} bytes")

    extracted = _ocr_image_bytes(image_bytes, "jpeg")
    await update.message.reply_text(f"📝 Extracted content:\n\n{extracted}")

    sess = session_get(chat_id)

    if caption:
        mark_keywords = ["mark", "check", "compare", "grade", "evaluate", "feedback"]
        if sess and any(kw in caption.lower() for kw in mark_keywords):
            await update.message.reply_text(
                f"⏳ Marking against *{sess['source']}*…", parse_mode="Markdown")
            prompt   = _prompt_mark(sess["text"], extracted)
            response = _call_llm(prompt)
            await send_llm_response(update, response,
                                    f"Marking Feedback — {sess['source']}", msg_id)
            session_clear(chat_id)
            await update.message.reply_text(
                "_(Session cleared — uploaded file no longer in memory.)_",
                parse_mode="Markdown")
        else:
            doc_text = sess["text"] if sess else extracted
            source   = sess["source"] if sess else "handwritten photo"
            await update.message.reply_text(
                f"⏳ Processing using *{source}* as reference…",
                parse_mode="Markdown")
            prompt   = _route_session_prompt(doc_text, caption)
            response = _call_llm(prompt)
            await send_llm_response(update, response, f"Answer — {source}", msg_id)
            if sess:
                session_clear(chat_id)
                await update.message.reply_text(
                    "_(Session cleared — uploaded file no longer in memory.)_",
                    parse_mode="Markdown")
    else:
        routed_command = auto_route_extracted_text(extracted)

        if routed_command:
            await update.message.reply_text(
                f"🔍 Detected question: `{routed_command[:120]}`\n"
                f"Solving automatically…", parse_mode="Markdown")

            q_lower = routed_command.lower()

            if is_periodic_fourier(q_lower):
                period_val = _extract_period(routed_command) or 2.0
                g_def_r    = re.search(
                    r'g\s*\(\s*t\s*\)\s*=\s*([^,\n]+)',
                    routed_command, re.IGNORECASE
                )
                g_str = g_def_r.group(1).strip() if g_def_r else None
                if g_str:
                    steps, err = _build_periodic_fourier_steps(g_str, period_val)
                    if not err:
                        png = _render_math_png(
                            "Fourier Transform  (Periodic Summation)", steps, msg_id
                        )
                        if png and os.path.exists(png):
                            success = await _send_photo_with_retry(
                                update, png,
                                f"Fourier Transform of  x(t) = Σ g(t − {period_val}k)\n"
                                f"where  g(t) = {g_str},  T = {period_val}"
                            )
                            if success:
                                return

            elif is_laplace(q_lower):
                expr_str = extract_expr(routed_command)
                if expr_str:
                    steps, err = _build_laplace_steps(expr_str)
                    if not err:
                        png = _render_math_png("Laplace Transform", steps, msg_id)
                        if png and os.path.exists(png):
                            success = await _send_photo_with_retry(
                                update, png,
                                f"Laplace Transform of  f(t) = {expr_str}")
                            if success:
                                return
                    await send_long_code(update, compute_laplace(expr_str or routed_command))
                    return

            elif is_fourier(q_lower):
                expr_str = extract_expr(routed_command)
                if expr_str:
                    steps, err = _build_fourier_steps(expr_str)
                    if not err:
                        png = _render_math_png("Fourier Transform", steps, msg_id)
                        if png and os.path.exists(png):
                            success = await _send_photo_with_retry(
                                update, png,
                                f"Fourier Transform of  f(t) = {expr_str}")
                            if success:
                                return
                    await send_long_code(update, compute_fourier(expr_str or routed_command))
                    return

            elif is_fs(q_lower):
                period = _extract_period(routed_command)
                if period:
                    expr_str = extract_expr(routed_command)
                    if expr_str:
                        steps, err = _build_fourier_series_steps(expr_str, period)
                        if not err:
                            png = _render_math_png("Fourier Series", steps, msg_id)
                            if png and os.path.exists(png):
                                success = await _send_photo_with_retry(
                                    update, png,
                                    f"Fourier Series: f(t)={expr_str}, T={period:.4g}")
                                if success:
                                    return
                        await send_long_code(update, compute_fourier_series(expr_str, period))
                        return

            elif is_conv(q_lower):
                e1, e2 = _parse_two_signals(routed_command)
                if e1 and e2:
                    steps, err, plot_path = _build_convolution_steps(e1, e2, msg_id)
                    if not err:
                        png = _render_math_png("Convolution (f ★ g)(t)", steps, msg_id)
                        if png and os.path.exists(png):
                            await _send_photo_with_retry(
                                update, png,
                                f"Convolution: f(t)={e1} ★ g(t)={e2}")
                        if plot_path and os.path.exists(plot_path):
                            await _send_photo_with_retry(
                                update, plot_path,
                                "📊 Numerical convolution (f ★ g)(t)")
                        return

            elif is_plot(q_lower):
                fig_path = generate_plot(routed_command, msg_id)
                if fig_path and os.path.exists(fig_path):
                    success = await _send_photo_with_retry(
                        update, fig_path, f"📈 {routed_command}")
                    if success:
                        return

            # Math pipeline failed — fall back to LLM
            if qa_chain:
                await update.message.reply_text("🤔 Solving with tutor…")
                try:
                    answer = qa_chain.invoke(extracted)
                    await send_llm_response(update, answer, "Tutor Answer", msg_id)
                except Exception as e:
                    await update.message.reply_text(f"❌ Something went wrong: {str(e)}")
            else:
                await update.message.reply_text(
                    "⚠️ No knowledge base loaded. Try adding a caption with your question.")

        else:
            if sess:
                context.user_data["pending_student_work"] = extracted
                await update.message.reply_text(
                    f"I have *{sess['source']}* loaded as your memo/reference.\n\n"
                    "Reply *mark* to compare your work against it, or tell me what "
                    "else you'd like me to do.",
                    parse_mode="Markdown")
            else:
                session_store(chat_id, extracted, "Handwritten photo / diagram")
                await update.message.reply_text(
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
        await update.message.reply_text(
            "⚠️ Unsupported file type. Please send a PDF or an image (PNG, JPG, WEBP).")
        return

    await update.message.reply_text(
        f"📄 Received *{file_name}* — extracting content…\n",
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

    await update.message.reply_text(
        f"✅ Content loaded from *{source}*.\n\n"
        f"Now tell me what you'd like me to do:\n"
        f"  - Explain how Question 1(b) was solved\n"
        f"  - Answer Question 2(a)\n"
        f"  - Give me two practice problems like Question 3\n"
        f"  - Mark my work against this memo",
        parse_mode="Markdown")

    if caption:
        sess   = session_get(chat_id)
        prompt = _route_session_prompt(sess["text"], caption)
        await update.message.reply_text(
            f"⏳ Also acting on your caption: _{caption}_…",
            parse_mode="Markdown")
        response = _call_llm(prompt)
        await send_llm_response(update, response, f"Answer — {source}",
                                msg_id=update.message.message_id)
        session_clear(chat_id)
        await update.message.reply_text(
            "_(Session cleared — uploaded file no longer in memory.)_",
            parse_mode="Markdown")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN  — HTTP/1.1 forced to avoid write-stall on photo uploads
# ══════════════════════════════════════════════════════════════════════════════
async def main():
    app = (
        ApplicationBuilder()
        .token(BOT_TOKEN)
        .get_updates_http_version("1.1")   # avoid HTTP/2 write-stall on polling
        .http_version("1.1")               # avoid HTTP/2 write-stall on sends
        .build()
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help",  help_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.PHOTO,          handle_photo))
    app.add_handler(MessageHandler(filters.Document.ALL,   handle_document))
    print("✅ Bot is running!")
    await app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    asyncio.run(main())
