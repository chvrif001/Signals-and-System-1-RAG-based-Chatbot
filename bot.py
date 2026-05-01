import os
import asyncio
import nest_asyncio
import re
import base64
import requests
import shutil

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
from langchain_huggingface import HuggingFaceEmbeddings
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
BOT_TOKEN        = os.environ["BOT_TOKEN"]
TOGETHER_API_KEY = os.environ["TOGETHER_API_KEY"]
ADMIN_USER_ID    = int(os.environ.get("ADMIN_USER_ID", "0"))
QWEN_VISION_MODEL = "Qwen/Qwen2.5-7B-Instruct-Turbo"   # used for optional vision OCR fallback

# ── Paths ─────────────────────────────────────────────────────────────────────
PDF_FOLDER  = "./knowledge_base"
CHROMA_DIR  = os.environ.get("CHROMA_DIR", "/data/chroma_db")
IMG_FOLDER  = "/tmp/student_images"
PLOT_FOLDER = "/tmp/plots"

os.makedirs(PDF_FOLDER,  exist_ok=True)
os.makedirs(CHROMA_DIR,  exist_ok=True)
os.makedirs(IMG_FOLDER,  exist_ok=True)
os.makedirs(PLOT_FOLDER, exist_ok=True)

os.environ["TOGETHER_API_KEY"] = TOGETHER_API_KEY

# ══════════════════════════════════════════════════════════════════════════════
# SYMPY SYMBOLS
# ══════════════════════════════════════════════════════════════════════════════
t_sym  = sp.Symbol("t",     real=True)
s_sym  = sp.Symbol("s",     complex=True)
w_sym  = sp.Symbol("omega", real=True)
n_sym  = sp.Symbol("n",     integer=True)
tau    = sp.Symbol("tau",   real=True)
a_sym  = sp.Symbol("a",     positive=True)

# ══════════════════════════════════════════════════════════════════════════════
# ROBUST SIGNAL PARSER  (ported from Colab version)
# ══════════════════════════════════════════════════════════════════════════════
def _normalise(expr: str) -> str:
    """Clean up student shorthand before SymPy parsing."""
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
    "t":          t_sym,
    "s":          s_sym,
    "omega":      w_sym,
    "n":          n_sym,
    "pi":         sp.pi,
    "E":          sp.E,
    "j":          sp.I,
    "Heaviside":  sp.Heaviside,
    "DiracDelta": sp.DiracDelta,
    "exp":        sp.exp,
    "sin":        sp.sin,
    "cos":        sp.cos,
    "sqrt":       sp.sqrt,
    "Abs":        sp.Abs,
    "log":        sp.log,
}


def parse_ct_expr(text: str) -> sp.Expr:
    return sp.sympify(_normalise(text), locals=_COMMON_NS)


def parse_dt_expr(text: str) -> sp.Expr:
    return sp.sympify(_normalise(text), locals=_COMMON_NS)


_SIGNAL_CHARS = ['(', '[', 't', 'n', 'sin', 'cos', 'exp',
                 'delta', 'δ', '**', '*', '+', 'sqrt', 'log',
                 'Heaviside', 'DiracDelta']

_QUESTION_PREFIX = re.compile(
    r'^(?:'
    r'what\s+is\s+(?:the\s+)?|'
    r'what\'s\s+(?:the\s+)?|'
    r'find\s+(?:the\s+)?|'
    r'compute\s+(?:the\s+)?|'
    r'calculate\s+(?:the\s+)?|'
    r'determine\s+(?:the\s+)?|'
    r'give\s+me\s+(?:the\s+)?|'
    r'show\s+me\s+(?:the\s+)?'
    r')',
    re.IGNORECASE
)

_VERB_PREFIX = re.compile(
    r'^(?:'
    r'plot|draw|graph|sketch|visualis[ae]|diagram|'
    r'laplace\s+(?:transform\s+)?(?:of\s+)?|'
    r'(?:inverse\s+)?laplace\s+(?:of\s+)?|'
    r'fourier\s+(?:transform\s+)?(?:of\s+)?|'
    r'(?:inverse\s+)?fourier\s+(?:of\s+)?|'
    r'(?:i)?ft\s+(?:of\s+)?|'
    r'f\.?t\.?\s+(?:of\s+)?|'
    r'fourier\s+series\s+(?:of\s+)?|'
    r'convolve\s+|convolution\s+(?:of\s+)?|'
    r'compute\s+|calculate\s+|find\s+'
    r')',
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
# SIGNAL PLOTTER  (continuous + discrete)
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
        expr  = parse_dt_expr(expr_str)
        f_lam = sp.lambdify(
            n_sym, expr,
            modules=["numpy", {
                "Heaviside":  lambda x: np.where(np.asarray(x, float) >= 0, 1., 0.),
                "DiracDelta": lambda x: (np.asarray(x, float) == 0).astype(float),
            }]
        )
        n_vals = np.arange(-10, 21)
        y_vals = np.real(f_lam(n_vals)).astype(float)
        y_vals = np.clip(y_vals, -10, 10)
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
# LAPLACE TRANSFORM WITH STEPS
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
# FOURIER TRANSFORM WITH STEPS
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
# FOURIER SERIES WITH STEPS
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
# CONVOLUTION WITH STEPS
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
        t_vals = np.linspace(-5, 10, 2000)
        dt     = t_vals[1] - t_vals[0]
        f_lam  = _lambdify_ct(f_expr)
        g_lam  = _lambdify_ct(g_expr)
        f_vals = np.real(f_lam(t_vals)).astype(float)
        g_vals = np.real(g_lam(t_vals)).astype(float)
        conv   = np.convolve(f_vals, g_vals, mode="full") * dt
        t_conv = np.linspace(t_vals[0] * 2, t_vals[-1] * 2, len(conv))

        fig, axes = plt.subplots(3, 1, figsize=(9, 7))
        axes[0].plot(t_vals, f_vals, color="steelblue")
        axes[0].set(title="f(t)", xlabel="t")
        axes[0].grid(True, alpha=.3)
        axes[1].plot(t_vals, g_vals, color="darkorange")
        axes[1].set(title="g(t)", xlabel="t")
        axes[1].grid(True, alpha=.3)
        axes[2].plot(t_conv, conv, color="green")
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

    plot_path = None
    try:
        result = sp.integrate(integrand, (tau, -sp.oo, sp.oo))
        result = sp.simplify(result)
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


def is_laplace(q: str) -> bool:
    return any(k in q for k in LAPLACE_KEYS)

def is_fourier(q: str) -> bool:
    return any(k in q for k in FOURIER_KEYS)

def is_fs(q: str) -> bool:
    return any(k in q for k in FS_KEYS)

def is_conv(q: str) -> bool:
    return any(k in q for k in CONV_KEYS)

def is_plot(q: str) -> bool:
    return any(k in q for k in PLOT_KEYWORDS)


# ══════════════════════════════════════════════════════════════════════════════
# MARK CALCULATOR
# ══════════════════════════════════════════════════════════════════════════════
WEIGHTS      = {"tutorials": .10, "labs": .10, "tests": .20, "exam": .60}
PASS_MARK    = 50.0
TRIGGER_KEYS = ["pass", "exam mark", "calculate marks", "how much do i need",
                "calculate mark breakdown", "what do i need", "minimum",
                "final mark"]
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
    weighted = (test_avg            * WEIGHTS["tests"] +
                data["labs"]        * WEIGHTS["labs"]  +
                data["tutorials"]   * WEIGHTS["tutorials"])
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
                "⚠️ Enter a number 0–100, or type *cancel*.",
                parse_mode="Markdown")
            return True

        session["data"][step] = value
        idx = STEPS_MC.index(step)
        if idx + 1 < len(STEPS_MC):
            nxt            = STEPS_MC[idx + 1]
            session["step"] = nxt
            await update.message.reply_text(STEP_PROMPTS[nxt], parse_mode="Markdown")
        else:
            result = compute_result(session["data"])
            del mark_sessions[chat_id]
            await update.message.reply_text(result, parse_mode="Markdown")
        return True
    return False


# ══════════════════════════════════════════════════════════════════════════════
# TrOCR — Handwriting Recognition (lazy-loaded to save startup RAM)
# ══════════════════════════════════════════════════════════════════════════════
trocr_processor = None
trocr_model     = None
device          = "cuda" if torch.cuda.is_available() else "cpu"


def _load_trocr_if_needed():
    global trocr_processor, trocr_model
    if trocr_model is None:
        print("Loading TrOCR model on first photo request...")
        trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
        trocr_model     = VisionEncoderDecoderModel.from_pretrained(
            "microsoft/trocr-large-handwritten")
        trocr_model     = trocr_model.to(device)
        print(f"TrOCR ready on {device}")


def extract_handwritten_text(image_path: str) -> str:
    """
    Process full image first; fall back to horizontal strips for tall images.
    """
    _load_trocr_if_needed()

    img           = Image.open(image_path).convert("RGB")
    width, height = img.size

    if height <= 400:
        pixel_values = trocr_processor(
            images=img, return_tensors="pt"
        ).pixel_values.to(device)
        with torch.no_grad():
            ids = trocr_model.generate(pixel_values, max_new_tokens=512)
        return trocr_processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

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
        model="meta-llama/Meta-Llama-3-8B-Instruct-Lite",
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

scorer = SentenceTransformer(EMBEDDING_MODEL)


# ══════════════════════════════════════════════════════════════════════════════
# TELEGRAM HANDLERS
# ══════════════════════════════════════════════════════════════════════════════
async def send_long(update: Update, text: str):
    for i in range(0, len(text), 4096):
        await update.message.reply_text(text[i:i + 4096])


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
        "   _fourier transform of e^(-t)*u(t)_\n"
        "   _FT of 1_,  _what is the Fourier transform of u(t)_\n\n"
        "🎵 *Fourier Series*\n"
        "   _fourier series of t, T=2_\n\n"
        "🔁 *Convolution*\n"
        "   _convolve u(t) with e^(-t)*u(t)_\n\n"
        "📷 *Mark handwritten work* — send a photo\n"
        "📄 *Add documents* — send a PDF (admin only)\n"
        "🧮 *Exam mark calculator* — _how much do I need to pass_\n\n"
        "Use /help for the full guide.",
        parse_mode="Markdown"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🆘 *Full guide:*\n\n"
        "1️⃣ *Laplace Transform*\n"
        "   _laplace of e^(-2*t)*u(t)_\n"
        "   _what is the Laplace transform of 1_\n\n"
        "2️⃣ *Fourier Transform*\n"
        "   _fourier transform of e^(-t)*u(t)_\n"
        "   _FT of 1_,  _what is the FT of u(t)_\n"
        "   _compute FT of Heaviside(t)_\n\n"
        "3️⃣ *Fourier Series*\n"
        "   _fourier series of t, T=2_\n"
        "   _fourier series of t**2, T=2*pi_\n\n"
        "4️⃣ *Convolution*\n"
        "   _convolve e^(-t)*u(t) with u(t)_\n"
        "   _convolution of u(t) and u(t-2)_\n\n"
        "5️⃣ *Plot signals*\n"
        "   Continuous: _plot 2*u(t-2)_\n"
        "   Discrete:   _draw u[n]-u[n-3]_\n\n"
        "6️⃣ *Handwritten work* — send a photo\n"
        "   📌 With caption → I do what you ask\n"
        "   📌 No caption → compared to memo\n\n"
        "7️⃣ *Add document* — send a PDF (admin only)\n\n"
        "8️⃣ *Mark calculator*\n"
        "   _how much do I need to pass_\n\n"
        "💡 Use * for multiply, ** for power\n"
        "   e.g. e**(-2*t)*u(t)  or  e^-2t*u(t)",
        parse_mode="Markdown"
    )


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    question = update.message.text.strip()
    q_lower  = question.lower()
    msg_id   = update.message.message_id

    # ── Mark calculator (checked first — intercepts numeric replies in session) ─
    if await handle_mark_session(update, context):
        return

    # ── Laplace Transform ──────────────────────────────────────────────────────
    if is_laplace(q_lower):
        expr_str = extract_expr(question)
        if not expr_str:
            await update.message.reply_text(
                "⚠️ Please include an expression, e.g.:\n"
                "  _laplace of e^(-2*t)*u(t)_\n"
                "  _what is the Laplace transform of 1_",
                parse_mode="Markdown")
            return
        await update.message.reply_text("⏳ Computing Laplace transform…")
        result = compute_laplace(expr_str)
        await send_long(update, result)
        return

    # ── Fourier Transform ──────────────────────────────────────────────────────
    if is_fourier(q_lower):
        expr_str = extract_expr(question)
        if not expr_str:
            await update.message.reply_text(
                "⚠️ Please include an expression, e.g.:\n"
                "  _fourier transform of e^(-t)*u(t)_\n"
                "  _FT of 1_",
                parse_mode="Markdown")
            return
        await update.message.reply_text("⏳ Computing Fourier transform…")
        result = compute_fourier(expr_str)
        await send_long(update, result)
        return

    # ── Fourier Series ─────────────────────────────────────────────────────────
    if is_fs(q_lower):
        period = _extract_period(question)
        if not period:
            await update.message.reply_text(
                "⚠️ Please include the period, e.g.:\n"
                "  _fourier series of t, T=2_",
                parse_mode="Markdown")
            return
        expr_str = extract_expr(question)
        if not expr_str:
            expr_str = re.sub(r',?\s*[Tt]\s*=\s*[^\s]+', '', question).strip()
            expr_str = extract_expr(expr_str) or expr_str
        await update.message.reply_text(
            f"⏳ Computing Fourier series for f(t)={expr_str}, T={period:.4g}…")
        result = compute_fourier_series(expr_str, period)
        await send_long(update, result)
        return

    # ── Convolution ────────────────────────────────────────────────────────────
    if is_conv(q_lower):
        e1, e2 = _parse_two_signals(question)
        if not (e1 and e2):
            await update.message.reply_text(
                "⚠️ Please specify both signals, e.g.:\n"
                "  _convolve e^(-t)*u(t) with u(t)_",
                parse_mode="Markdown")
            return
        await update.message.reply_text(
            f"⏳ Computing convolution of  f(t)={e1}  and  g(t)={e2}…")
        text_result, plot_path = compute_convolution(e1, e2, msg_id)
        await send_long(update, text_result)
        if plot_path and os.path.exists(plot_path):
            await update.message.reply_photo(
                photo=open(plot_path, "rb"),
                caption="📊 Numerical convolution (f ★ g)(t)")
        return

    # ── Plot ───────────────────────────────────────────────────────────────────
    if is_plot(q_lower):
        await update.message.reply_text("📊 Generating plot…")
        fig_path = generate_plot(question, msg_id)
        if fig_path and os.path.exists(fig_path):
            await update.message.reply_photo(
                photo=open(fig_path, "rb"), caption=f"📈 {question}")
        else:
            await update.message.reply_text(
                "⚠️ Could not parse that expression.\n"
                "Examples:\n"
                "  _plot 2*u(t-2)_\n"
                "  _draw u[n] - u[n-3]_\n"
                "  _sketch e^(-2*t)*u(t)_",
                parse_mode="Markdown")
        return

    # ── General tutor Q&A ──────────────────────────────────────────────────────
    if not qa_chain:
        await update.message.reply_text(
            "⚠️ No knowledge base loaded yet. The admin needs to add PDFs.")
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
                    "⚠️ Bot not fully initialised yet.")
                return
            response = photo_chain.invoke({"task": caption, "question": extracted})
        else:
            if not check_chain:
                await update.message.reply_text(
                    "⚠️ No memo loaded. Add a caption to tell me what you'd like me to do.")
                return
            response = check_chain.invoke(extracted)

        await send_long(update, response)
    except Exception as e:
        await update.message.reply_text(f"❌ OCR failed: {str(e)}")


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Admin-only: upload a new PDF to the knowledge base."""
    global vector_store, qa_chain, check_chain, photo_chain

    if update.effective_user.id != ADMIN_USER_ID:
        await update.message.reply_text(
            "⚠️ Only the administrator can upload documents to the knowledge base.")
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
