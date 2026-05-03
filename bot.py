import os
import asyncio
import nest_asyncio
import re
import base64
import requests
import tempfile

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
    Application, CommandHandler, MessageHandler, filters, ContextTypes
)

nest_asyncio.apply()

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
BOT_TOKEN         = os.environ["BOT_TOKEN"]
TOGETHER_API_KEY  = os.environ["TOGETHER_API_KEY"]
ADMIN_USER_ID     = int(os.environ.get("ADMIN_USER_ID", "0"))

# Vision model — using Llama 3.2 Vision via NVIDIA NIM on Together AI
# Options:
#   "nim/meta/llama-3.2-11b-vision-instruct"  ← faster
#   "nim/meta/llama-3.2-90b-vision-instruct"  ← more accurate
VISION_MODEL = "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo"

LLM_MODEL    = "meta-llama/Meta-Llama-3-8B-Instruct-Lite"

# ── API endpoints ─────────────────────────────────────────────────────────────
TOGETHER_ENDPOINT = "https://api.together.xyz/v1/chat/completions"

def _get_endpoint(model: str) -> str:
    """All models, including nim/* prefixed ones, are served via Together AI."""
    return TOGETHER_ENDPOINT


# ── Paths ─────────────────────────────────────────────────────────────────────
PDF_FOLDER  = "./knowledge_base"
CHROMA_DIR  = os.environ.get("CHROMA_DIR", "/data/chroma_db")
PLOT_FOLDER = "/tmp/plots"

os.makedirs(PDF_FOLDER,  exist_ok=True)
os.makedirs(CHROMA_DIR,  exist_ok=True)
os.makedirs(PLOT_FOLDER, exist_ok=True)

os.environ["TOGETHER_API_KEY"] = TOGETHER_API_KEY

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STORE  — in-memory, per chat_id, never touches the knowledge base
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
def _normalise(expr: str) -> str:
    s = expr.strip()
    s = s.replace("^", "**").replace("{", "(").replace("}", ")")
    s = re.sub(r'\bE\*\*(-[^\s\*\+\-\(\),]+)', r'E**(\1)', s)
    s = re.sub(r'\be\*\*(-[^\s\*\+\-\(\),]+)', r'E**(\1)', s)
    s = re.sub(r'(\d)(t\b)',               r'\1*\2', s)
    s = re.sub(r'(\d)(sin|cos|exp|sqrt)',  r'\1*\2', s)
    s = re.sub(r'(\d)\s*[eE]\*\*',        r'\1*E**', s)
    s = re.sub(r'\be\b', 'E', s)
    s = re.sub(r'\bu\s*[\(\[]', 'Heaviside(', s)
    s = re.sub(r'\]', ')', s)
    s = re.sub(r'\b(?:delta|δ)\s*[\(\[]', 'DiracDelta(', s)
    s = re.sub(
        r'\bsinc\(([^)]+)\)',
        lambda m: f'(sin(pi*({m.group(1)})))/(pi*({m.group(1)}))',
        s
    )
    return s


_COMMON_NS = {
    "t": t_sym, "s": s_sym, "omega": w_sym, "n": n_sym,
    "pi": sp.pi, "E": sp.E, "j": sp.I,
    "Heaviside": sp.Heaviside, "DiracDelta": sp.DiracDelta,
    "exp": sp.exp, "sin": sp.sin, "cos": sp.cos,
    "sqrt": sp.sqrt, "Abs": sp.Abs, "log": sp.log,
}


def parse_ct_expr(text: str) -> sp.Expr:
    return sp.sympify(_normalise(text), locals=_COMMON_NS)


def _normalise_dt(expr: str) -> str:
    s = expr.strip()
    s = s.replace("^", "**").replace("{", "(").replace("}", ")")
    s = re.sub(r'\be\b', 'E', s)
    s = re.sub(r'\bu\s*\[([^\]]+)\]', r'UnitStep(\1)', s)
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
                 'Heaviside', 'DiracDelta']

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


def extract_expr(question: str) -> str | None:
    q = question.strip()
    q = _QUESTION_PREFIX.sub('', q).strip()
    q = _VERB_PREFIX.sub('', q).strip()
    q = q.rstrip("?.")
    if any(c in q for c in _SIGNAL_CHARS):
        return q
    if re.match(r'^-?[\d][\d\.]*$', q.strip()):
        return q
    if re.search(r'\be\b', q):
        return q
    return None


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
# LAPLACE TRANSFORM
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
# FOURIER TRANSFORM
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


def compute_fourier(expr_str: str) -> str:
    lines = ["━━━ 📡 FOURIER TRANSFORM ━━━\n"]
    lines.append(f"Input:  f(t) = {expr_str}\n")
    lines.append("Definition:  F(ω) = ∫₋∞^∞  f(t) · e^(-jωt) dt\n")
    try:
        f = parse_ct_expr(expr_str)
    except Exception as e:
        return (f"❌ Could not parse expression: {e}\n"
                f"Try: e**(-t)*u(t)  or  cos(2*t)*u(t)")
    rule = _identify_fourier_rule(f)
    lines.append(f"Step 1 — Recognise the signal form:\n   → {rule}\n")
    args = sp.Add.make_args(f)
    if len(args) > 1:
        lines.append("Step 2 — Apply linearity  F{af + bg} = aF(ω) + bG(ω):")
        for term in args:
            try:
                r = sp.fourier_transform(term, t_sym, w_sym / (2 * sp.pi), noconds=True)
                lines.append(f"   F{{{sp.pretty(term)}}} = {sp.pretty(r)}")
            except Exception:
                lines.append(f"   F{{{sp.pretty(term)}}} = (could not evaluate term)")
        lines.append("")
    else:
        lines.append("Step 2 — Single term, applying transform directly.\n")
    try:
        result = sp.fourier_transform(f, t_sym, w_sym / (2 * sp.pi), noconds=True)
        result = sp.simplify(result)
        lines.append(f"Step 3 — Final result:\n   F(ω) = {sp.pretty(result)}\n")
        if "exp" in str(f) and "Heaviside" in str(f):
            lines.append("Magnitude |F(ω)|: decreases as 1/ω — low-pass character")
        lines.append(f"\n✅  F(ω) = {sp.pretty(result)}")
    except Exception as e:
        lines.append(f"❌ SymPy could not evaluate: {e}")
        lines.append("   Tip: make sure causal signals include u(t) / Heaviside(t)")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# FOURIER SERIES
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
# CONVOLUTION
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
    lines.append("Step 2 — Determine integration limits from signal support:")
    lines.append("   (Causal signals: limits become 0 to t)\n")
    plot_path   = None
    both_causal = ("Heaviside" in str(f) or str(f) == str(sp.Heaviside(t_sym))) and \
                  ("Heaviside" in str(g) or str(g) == str(sp.Heaviside(t_sym)))
    limits = (tau, 0, t_sym) if both_causal else (tau, -sp.oo, sp.oo)
    try:
        result = sp.integrate(integrand, limits)
        result = sp.simplify(result)
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
FS_KEYS   = ["fourier series", "periodic signal", "series of"]
CONV_KEYS = ["convolution", "convolve", "f*g", "f★g", "f star g"]


def is_laplace(q: str) -> bool: return any(k in q for k in LAPLACE_KEYS)
def is_fourier(q: str) -> bool: return any(k in q for k in FOURIER_KEYS)
def is_fs(q: str)      -> bool: return any(k in q for k in FS_KEYS)
def is_conv(q: str)    -> bool: return any(k in q for k in CONV_KEYS)
def is_plot(q: str)    -> bool: return any(k in q for k in PLOT_KEYWORDS)


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
# OCR — Llama 3.2 Vision via NVIDIA NIM on Together AI
# ══════════════════════════════════════════════════════════════════════════════

# Phrases that indicate the model did not receive or cannot see the image
_OCR_FAILURE_SIGNALS = [
    "[object", "看起来", "无法", "图片", "I cannot see",
    "no image", "didn't receive", "can't see the image",
    "don't see any image", "no image was provided",
    "I don't have the ability to view",
    "I'm unable to view", "cannot view",
    "there is no image", "not able to see",
]


def _ocr_image_bytes(image_bytes: bytes, mime: str) -> str:
    """Send image bytes to the vision model and return extracted text."""

    # Guard: empty bytes means the download failed
    if not image_bytes or len(image_bytes) < 100:
        return (f"❌ OCR failed: image data is empty or too small "
                f"({len(image_bytes)} bytes)")

    # Normalise mime type
    mime = mime.lower().lstrip(".")
    if mime in ("jpg",):
        mime = "jpeg"
    if mime not in ("jpeg", "png", "webp", "gif"):
        mime = "jpeg"

    b64      = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:image/{mime};base64,{b64}"

    payload = {
        "model": VISION_MODEL,
        "max_tokens": 2048,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    },
                    {
                        "type": "text",
                        "text": (
                            "You are an OCR assistant. Transcribe ALL handwritten text "
                            "and mathematical expressions in this image exactly as written. "
                            "Preserve equations, symbols, numbering, and layout. "
                            "Output ONLY the transcribed content — "
                            "no commentary, no explanations, no translations."
                        ),
                    },
                ],
            }
        ],
    }

    endpoint = _get_endpoint(VISION_MODEL)
    headers  = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type":  "application/json",
    }

    print(f"[OCR] Sending {len(image_bytes)} bytes to {endpoint} "
          f"using model {VISION_MODEL}")

    try:
        resp = requests.post(endpoint, json=payload, headers=headers, timeout=90)
        resp.raise_for_status()
        data    = resp.json()
        content = data["choices"][0]["message"]["content"].strip()

        # Check for known failure phrases
        if any(sig.lower() in content.lower() for sig in _OCR_FAILURE_SIGNALS):
            print(f"[OCR] Suspicious response: {content[:200]}")
            return ("❌ OCR failed: the model did not receive the image correctly. "
                    "Please try again.")

        print(f"[OCR] Success — extracted {len(content)} characters")
        return content

    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response is not None else "?"
        body   = e.response.text[:300] if e.response is not None else ""
        print(f"[OCR] HTTP {status}: {body}")
        # 422 often means the model doesn't accept this image format via NIM
        if status == 422:
            return ("❌ OCR failed: the vision model rejected the image "
                    f"(HTTP 422). Try sending a plain JPEG. Details: {body}")
        return f"❌ OCR request failed (HTTP {status}): {e}"

    except requests.exceptions.RequestException as e:
        print(f"[OCR] Request error: {e}")
        return f"❌ OCR request failed: {e}"

    except (KeyError, IndexError) as e:
        print(f"[OCR] Unexpected response format: {e}")
        return f"❌ OCR response could not be parsed: {e}"


def _extract_pdf_bytes(pdf_bytes: bytes) -> str:
    """
    Extract text from PDF bytes using pypdf.
    Runs entirely in memory — never touches the knowledge base.
    """
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
# SESSION-AWARE LLM CALLS
# ══════════════════════════════════════════════════════════════════════════════
_SESSION_RULES = """IMPORTANT — follow strictly:
1. The uploaded document is the PRIMARY and authoritative source of truth.
2. Use your general Signals & Systems knowledge ONLY to explain or clarify —
   never to contradict, override, or replace the uploaded content.
3. For correctness, marking, and solution verification rely exclusively on
   the uploaded document.
4. If OCR output looks garbled in a critical spot, say so and give your best
   interpretation — do not silently substitute your own answer."""


def _prompt_solve(doc_text: str, instruction: str) -> str:
    return (
        f"{_SESSION_RULES}\n\n"
        f"Uploaded document:\n\"\"\"\n{doc_text}\n\"\"\"\n\n"
        f"Student instruction: {instruction}\n\n"
        f"Respond directly. Use numbered steps where maths is involved. "
        f"Explain every formula and symbol."
    )


def _prompt_mark(memo_text: str, student_work: str) -> str:
    return (
        f"{_SESSION_RULES}\n\n"
        f"Memo / expected solution (from uploaded file):\n\"\"\"\n{memo_text}\n\"\"\"\n\n"
        f"Student's work (OCR extracted):\n\"\"\"\n{student_work}\n\"\"\"\n\n"
        f"Your task:\n"
        f"1. State CORRECT, PARTIALLY CORRECT, or INCORRECT.\n"
        f"2. Identify every error or missing step clearly.\n"
        f"3. Explain the correct approach for each error.\n"
        f"4. Acknowledge what the student did right.\n"
        f"Be encouraging and specific."
    )


def _prompt_explain(doc_text: str, question: str) -> str:
    return (
        f"{_SESSION_RULES}\n\n"
        f"Uploaded document:\n\"\"\"\n{doc_text}\n\"\"\"\n\n"
        f"Student question: {question}\n\n"
        f"Classify your answer silently:\n"
        f"  FACTUAL     → 2–4 sentences.\n"
        f"  CONCEPTUAL  → short paragraph + one example.\n"
        f"  CALCULATION → numbered steps, explain every symbol.\n"
        f"Answer based on the uploaded document. Use general knowledge only to aid clarity."
    )


def _call_llm(prompt: str, max_tokens: int = 1500) -> str:
    """Direct Together AI call — independent of the RAG chain."""
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
    if any(kw in instr_lower for kw in ["mark", "check", "compare", "correct",
                                          "feedback", "evaluate", "grade"]):
        return _prompt_mark(doc_text, instruction)
    if any(kw in instr_lower for kw in ["solve", "calculate", "find", "compute",
                                          "work out", "answer", "determine"]):
        return _prompt_solve(doc_text, instruction)
    return _prompt_explain(doc_text, instruction)


# ══════════════════════════════════════════════════════════════════════════════
# VECTOR STORE  (read-only after startup)
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


# ── Boot up vector store ───────────────────────────────────────────────────────
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
        "🎵 *Fourier Series*\n"
        "   _fourier series of t, T=2_\n\n"
        "🔁 *Convolution*\n"
        "   _convolve u(t) with e^(-t)*u(t)_\n\n"
        "📷 *Upload a photo* — send handwritten work\n"
        "   With caption → I do what you ask\n"
        "   No caption   → compared to a loaded memo\n\n"
        "📄 *Upload a PDF or image* — question paper, memo, textbook\n"
        "   Sent as a file → loaded as session context\n"
        "   Then ask: _solve question 3_, _mark my work_, _explain this_\n\n"
        "🧮 *Exam mark calculator* — _how much do I need to pass_\n\n"
        "Use /help for the full guide.",
        parse_mode="Markdown"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🆘 *Full guide:*\n\n"
        "1️⃣ *Laplace Transform*\n"
        "   _laplace of e^(-2*t)*u(t)_\n\n"
        "2️⃣ *Fourier Transform*\n"
        "   _fourier transform of e^(-t)*u(t)_\n\n"
        "3️⃣ *Fourier Series*\n"
        "   _fourier series of t**2, T=2*pi_\n\n"
        "4️⃣ *Convolution*\n"
        "   _convolve e^(-t)*u(t) with u(t)_\n\n"
        "5️⃣ *Plot signals*\n"
        "   Continuous: _plot 2*u(t-2)_\n"
        "   Discrete:   _draw u[n]-u[n-3]_\n\n"
        "6️⃣ *Upload a PDF or image file*\n"
        "   Send the file → bot loads it as session context\n"
        "   Then send a follow-up message:\n"
        "     _solve question 2b_\n"
        "     _mark my work against this memo_\n"
        "     _explain what question 3 is asking_\n"
        "   ⚠️ Uploaded files are session-only — not saved permanently.\n\n"
        "7️⃣ *Handwritten photo*\n"
        "   📌 With caption → bot follows your instruction\n"
        "   📌 No caption   → compared against loaded session memo\n\n"
        "8️⃣ *Mark calculator*\n"
        "   _how much do I need to pass_\n\n"
        "💡 Use * for multiply, ** for power\n"
        "   e.g. e**(-2*t)*u(t)  or  e^-2t*u(t)",
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
            await send_long(update, response)
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
        await send_long(update, response)
        session_clear(chat_id)
        await update.message.reply_text(
            "_(Session cleared — uploaded file no longer in memory.)_",
            parse_mode="Markdown")
        return

    # ── 3. Math tools ─────────────────────────────────────────────────────────
    if is_laplace(q_lower):
        expr_str = extract_expr(question)
        if not expr_str:
            await update.message.reply_text(
                "⚠️ Please include an expression, e.g.:\n"
                "  _laplace of e^(-2*t)*u(t)_", parse_mode="Markdown")
            return
        await update.message.reply_text("⏳ Computing Laplace transform…")
        await send_long_code(update, compute_laplace(expr_str))
        return

    if is_fourier(q_lower):
        expr_str = extract_expr(question)
        if not expr_str:
            await update.message.reply_text(
                "⚠️ Please include an expression, e.g.:\n"
                "  _fourier transform of e^(-t)*u(t)_", parse_mode="Markdown")
            return
        await update.message.reply_text("⏳ Computing Fourier transform…")
        await send_long_code(update, compute_fourier(expr_str))
        return

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
        await send_long_code(update, compute_fourier_series(expr_str, period))
        return

    if is_conv(q_lower):
        e1, e2 = _parse_two_signals(question)
        if not (e1 and e2):
            await update.message.reply_text(
                "⚠️ Please specify both signals, e.g.:\n"
                "  _convolve e^(-t)*u(t) with u(t)_", parse_mode="Markdown")
            return
        await update.message.reply_text(
            f"⏳ Computing convolution of  f(t)={e1}  and  g(t)={e2}…")
        text_result, plot_path = compute_convolution(e1, e2, msg_id)
        await send_long_code(update, text_result)
        if plot_path and os.path.exists(plot_path):
            await update.message.reply_photo(
                photo=open(plot_path, "rb"),
                caption="📊 Numerical convolution (f ★ g)(t)")
        return

    if is_plot(q_lower):
        await update.message.reply_text("📊 Generating plot…")
        fig_path = generate_plot(question, msg_id)
        if fig_path and os.path.exists(fig_path):
            await update.message.reply_photo(
                photo=open(fig_path, "rb"), caption=f"📈 {question}")
        else:
            await update.message.reply_text(
                "⚠️ Could not parse that expression.\n"
                "Examples:\n  _plot 2*u(t-2)_\n  _draw u[n] - u[n-3]_",
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
        await send_long(update, answer)
    except Exception as e:
        await update.message.reply_text(f"❌ Something went wrong: {str(e)}")


# ══════════════════════════════════════════════════════════════════════════════
# PHOTO HANDLER
# ══════════════════════════════════════════════════════════════════════════════
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    caption = (update.message.caption or "").strip()
    chat_id = update.effective_chat.id

    await update.message.reply_text(
        "📷 Got your photo — running handwriting recognition… (~15–30s)")

    import io
    photo_file  = await update.message.photo[-1].get_file()
    buf         = io.BytesIO()
    await photo_file.download_to_memory(buf)
    image_bytes = buf.getvalue()
    print(f"[handle_photo] Downloaded {len(image_bytes)} bytes")

    extracted = _ocr_image_bytes(image_bytes, "jpeg")
    await update.message.reply_text(
        f"📝 *Extracted handwriting:*\n\n{extracted}", parse_mode="Markdown")

    sess = session_get(chat_id)

    if caption:
        mark_keywords = ["mark", "check", "compare", "grade", "evaluate", "feedback"]
        if sess and any(kw in caption.lower() for kw in mark_keywords):
            await update.message.reply_text(
                f"⏳ Marking against *{sess['source']}*…", parse_mode="Markdown")
            prompt   = _prompt_mark(sess["text"], extracted)
            response = _call_llm(prompt)
            await send_long(update, response)
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
            await send_long(update, response)
            if sess:
                session_clear(chat_id)
                await update.message.reply_text(
                    "_(Session cleared — uploaded file no longer in memory.)_",
                    parse_mode="Markdown")
    else:
        if sess:
            context.user_data["pending_student_work"] = extracted
            await update.message.reply_text(
                f"I have *{sess['source']}* loaded as your memo/reference.\n\n"
                "Reply *mark* to compare your work against it, or tell me what "
                "else you'd like me to do.",
                parse_mode="Markdown")
        else:
            session_store(chat_id, extracted, "Handwritten photo")
            await update.message.reply_text(
                "Photo loaded. What would you like me to do?\n"
                "  • _Solve this_\n"
                "  • _Explain step by step_\n"
                "  • _What is this question asking?_")


# ══════════════════════════════════════════════════════════════════════════════
# DOCUMENT HANDLER
# ══════════════════════════════════════════════════════════════════════════════
async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Any user may upload a file (PDF or image) as temporary session context.
    The file is read into memory and then discarded — it is NEVER written to
    PDF_FOLDER, CHROMA_DIR, or any other persistent location.
    """
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
        f"📄 Received *{file_name}* — extracting content…\n"
        f"_(This file is used for this session only and will not be saved permanently.)_",
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

    preview = extracted[:300].replace("\n", " ")
    await update.message.reply_text(
        f"✅ Content loaded from *{source}*.\n\n"
        f"Preview: _{preview}…_\n\n"
        f"Now tell me what you'd like me to do:\n"
        f"  • _Solve question 3_\n"
        f"  • _Mark my work against this memo_\n"
        f"  • _Explain what this question is asking_",
        parse_mode="Markdown")

    if caption:
        sess   = session_get(chat_id)
        prompt = _route_session_prompt(sess["text"], caption)
        await update.message.reply_text(
            f"⏳ Also acting on your caption: _{caption}_…",
            parse_mode="Markdown")
        response = _call_llm(prompt)
        await send_long(update, response)
        session_clear(chat_id)
        await update.message.reply_text(
            "_(Session cleared — uploaded file no longer in memory.)_",
            parse_mode="Markdown")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
async def main():
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help",  help_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.PHOTO,          handle_photo))
    app.add_handler(MessageHandler(filters.Document.ALL,   handle_document))
    print("✅ Bot is running!")
    await app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    asyncio.run(main())
