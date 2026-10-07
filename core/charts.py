"""Small, dependency-free chart renderers (HTML / inline SVG).

All output is single-line HTML without indentation or blank lines, which is what
st.markdown(..., unsafe_allow_html=True) needs to avoid treating it as a code block.
"""
from __future__ import annotations

import base64
from html import escape as e

from .theme import BORDER, INDIGO, LAVENDER, MUTED, NAVY, TEXT


def _one_line(s: str) -> str:
    return "".join(line.strip() for line in s.splitlines())


def bar_rows(rows: list[tuple[str, float, str, str]], max_value: float | None = None, top_index: int | None = None) -> str:
    """rows: (label, value, colour, value_text). Value is scaled against max_value (default max)."""
    mx = max_value if max_value else max((r[1] for r in rows), default=1) or 1
    out = []
    for i, (label, value, color, text) in enumerate(rows):
        width = max(0.0, min(100.0, value / mx * 100))
        cls = "prob-row top" if i == top_index else "prob-row"
        out.append(
            f'<div class="{cls}"><div class="name">{e(label)}</div>'
            f'<div class="prob-track"><div class="prob-fill" style="width:{width:.1f}%;background:{color}"></div></div>'
            f'<div>{e(text)}</div></div>'
        )
    return _one_line("".join(out))


def confusion_matrix_html(cm: list[list[int]], labels: list[str]) -> str:
    """Rows = true class, columns = predicted class. Shade = share of the TRUE class (row-normalised)."""
    head = "".join(f'<th style="text-align:center">{e(l)}</th>' for l in labels)
    body = []
    for i, row in enumerate(cm):
        total = sum(row) or 1
        cells = []
        for j, v in enumerate(row):
            share = v / total
            alpha = 0.08 + 0.82 * share
            color = "#fff" if alpha > 0.55 else TEXT
            border = f"2px solid {NAVY}" if i == j else f"1px solid {BORDER}"
            cells.append(
                f'<td style="text-align:center;background:rgba(91,91,214,{alpha:.2f});color:{color};border:{border}">'
                f'<div style="font-weight:800;font-size:16px">{v:,}</div>'
                f'<div style="font-size:12px;opacity:.9">{share * 100:.1f}% of true class</div></td>'
            )
        body.append(f'<tr><th>{e(labels[i])}<div class="small">{total:,} cases</div></th>{"".join(cells)}</tr>')
    return _one_line(
        f'<div class="table-wrap"><table class="clean"><thead><tr><th>True ↓ / Predicted →</th>{head}</tr></thead>'
        f'<tbody>{"".join(body)}</tbody></table></div>'
    )


def roc_svg(curves: list[tuple[str, list[float], list[float], float, str]]) -> str:
    """curves: (label, fpr, tpr, auc, colour). Returns an <img> tag with a data-URI SVG."""
    W, H, L, B, T, R = 560, 420, 52, 46, 16, 16
    pw, ph = W - L - R, H - T - B

    def x(v):
        return L + v * pw

    def y(v):
        return T + (1 - v) * ph

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Inter,Arial,sans-serif">',
             f'<rect width="{W}" height="{H}" rx="14" fill="#ffffff"/>']
    for k in range(0, 6):
        v = k / 5
        parts.append(f'<line x1="{x(0)}" x2="{x(1)}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="#eeeafd"/>')
        parts.append(f'<line x1="{x(v):.1f}" x2="{x(v):.1f}" y1="{y(0)}" y2="{y(1)}" stroke="#eeeafd"/>')
        parts.append(f'<text x="{L - 8}" y="{y(v) + 4:.1f}" font-size="11" text-anchor="end" fill="{MUTED}">{v:.1f}</text>')
        parts.append(f'<text x="{x(v):.1f}" y="{H - B + 18}" font-size="11" text-anchor="middle" fill="{MUTED}">{v:.1f}</text>')
    parts.append(f'<line x1="{x(0)}" y1="{y(0)}" x2="{x(1)}" y2="{y(1)}" stroke="{LAVENDER}" stroke-dasharray="5 5"/>')
    for label, fpr, tpr, auc, color in curves:
        pts = " ".join(f"{x(a):.1f},{y(b):.1f}" for a, b in zip(fpr, tpr))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.6" stroke-linejoin="round"/>')
    parts.append(f'<text x="{L + pw / 2}" y="{H - 8}" font-size="12" text-anchor="middle" fill="{NAVY}" font-weight="700">False positive rate</text>')
    parts.append(f'<text transform="translate(14 {T + ph / 2}) rotate(-90)" font-size="12" text-anchor="middle" fill="{NAVY}" font-weight="700">True positive rate</text>')
    # legend
    for i, (label, _, _, auc, color) in enumerate(curves):
        ly = T + ph - 62 + i * 20          # bottom-right: the empty corner of a ROC plot
        parts.append(f'<rect x="{x(1) - 200}" y="{ly - 9}" width="14" height="4" rx="2" fill="{color}"/>')
        parts.append(f'<text x="{x(1) - 180}" y="{ly}" font-size="12" fill="{TEXT}">{e(label)} (AUC {auc:.3f})</text>')
    parts.append("</svg>")
    svg = "".join(parts)
    uri = "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()
    return f'<img src="{uri}" alt="ROC curves" style="width:100%;max-width:640px;display:block;margin:0 auto;border-radius:14px;border:1px solid {BORDER}"/>'


def metrics_table(rows: list[tuple[str, str, str, str, str, str]]) -> str:
    """rows: (class label, colour, precision, recall, f1, support)"""
    body = "".join(
        f'<tr><td><span class="badge" style="background:{c}">{e(name)}</span></td>'
        f"<td>{p}</td><td>{r}</td><td>{f}</td><td>{s}</td></tr>"
        for name, c, p, r, f, s in rows
    )
    return _one_line(
        '<div class="table-wrap"><table class="clean"><thead><tr><th>Class</th><th>Precision</th><th>Recall</th>'
        f"<th>F1-score</th><th>Test cases</th></tr></thead><tbody>{body}</tbody></table></div>"
    )
