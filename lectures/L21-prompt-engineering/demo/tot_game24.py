"""Tree-of-Thought for the Game of 24 — visualization.

Implements ToT (Yao 2023) for the canonical 24-game problem:
given 4 numbers, use +, -, *, / to make 24.

Reproduces a small search tree and saves a tree-diagram PNG showing
explored vs pruned vs solution paths.

Run:  python tot_game24.py
"""
from __future__ import annotations
import os
from itertools import permutations
from dataclasses import dataclass, field
from typing import Optional
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams['font.sans-serif'] = [
    'Microsoft YaHei', 'SimHei', 'PingFang SC',
    'Microsoft JhengHei', 'Segoe UI Emoji', 'DejaVu Sans',
]
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)


def safe_eval(expr: str) -> Optional[float]:
    try:
        v = eval(expr, {"__builtins__": {}}, {})
        if isinstance(v, (int, float)):
            return float(v)
    except (ZeroDivisionError, SyntaxError):
        return None
    return None


def solve_24(nums: list[int]) -> Optional[str]:
    """Brute search; returns the first expression evaluating to 24."""
    ops = ["+", "-", "*", "/"]
    for perm in permutations(nums):
        a, b, c, d = perm
        for o1 in ops:
            for o2 in ops:
                for o3 in ops:
                    for tmpl in [
                        "(({a}{o1}{b}){o2}{c}){o3}{d}",
                        "({a}{o1}({b}{o2}{c})){o3}{d}",
                        "({a}{o1}{b}){o2}({c}{o3}{d})",
                        "{a}{o1}(({b}{o2}{c}){o3}{d})",
                        "{a}{o1}({b}{o2}({c}{o3}{d}))",
                    ]:
                        e = tmpl.format(a=a, b=b, c=c, d=d,
                                        o1=o1, o2=o2, o3=o3)
                        v = safe_eval(e)
                        if v is not None and abs(v - 24) < 1e-6:
                            return e
    return None


@dataclass
class Node:
    label: str
    score: float            # 0..1 confidence (LLM evaluator)
    state: str              # "explore" | "prune" | "solution"
    children: list = field(default_factory=list)
    x: float = 0.0
    y: float = 0.0


def build_demo_tree() -> Node:
    """Hand-crafted ToT tree for nums=[4, 6, 8, 8] -> e.g. (8-4)*6=24 OR
    (8/(1-...))."""
    root = Node("4, 6, 8, 8\n(start)", 1.0, "explore")

    # Step 1: choose first pair
    pairs = [
        ("4+6=10 -> [10, 8, 8]", 0.4, "explore"),
        ("4*6=24 -> [24, 8, 8]\n再用 8,8=0 即可\n8-8=0, 24+0=24 [solved]",
         0.95, "solution"),
        ("8+8=16 -> [4, 6, 16]", 0.6, "explore"),
        ("8-4=4  -> [4, 6, 8]", 0.5, "explore"),
        ("8/4=2  -> [2, 6, 8]", 0.3, "prune"),
    ]
    for label, score, state in pairs:
        root.children.append(Node(label, score, state))

    # Expand the high-score branch (the "(4+6)=10" path)
    p10 = root.children[0]
    p10.children = [
        Node("10*8=80 -> [80, 8]\n80-8=72 (no)", 0.1, "prune"),
        Node("10-8=2 -> [2, 8]\n 2*8=16 (no)", 0.2, "prune"),
        Node("10+8=18 -> [18, 8]\n18+8=26 (no)", 0.15, "prune"),
    ]
    # Expand 8+8=16 branch
    p16 = root.children[2]
    p16.children = [
        Node("4*6=24, 16 unused\n  24+16-16=24 [trick]\n  (artificial)", 0.85,
             "solution"),
        Node("16-4-6=6 (no)", 0.1, "prune"),
        Node("16/(6-4)=8 (no)", 0.2, "prune"),
    ]
    # Expand 8-4=4 branch
    p84 = root.children[3]
    p84.children = [
        Node("4+6+8=18 (no)", 0.1, "prune"),
        Node("4*(6+8)=56 (no)", 0.1, "prune"),
        Node("4+8*6=52 (no)", 0.1, "prune"),
    ]
    return root


def layout_tree(root: Node) -> None:
    """Simple horizontal layout: depth = x, sibling order = y."""
    def assign_y(node: Node, depth: int, y_acc: list[float]) -> float:
        node.x = depth * 1.0
        if not node.children:
            node.y = y_acc[0]
            y_acc[0] += 1.0
            return node.y
        ys = [assign_y(c, depth + 1, y_acc) for c in node.children]
        node.y = (min(ys) + max(ys)) / 2
        return node.y
    assign_y(root, 0, [0.0])


def draw_tree(root: Node, out_path: str) -> None:
    layout_tree(root)
    fig, ax = plt.subplots(figsize=(15, 8))
    color_map = {"explore": "#9E9E9E", "prune": "#D62728",
                 "solution": "#2CA02C"}

    def walk(node: Node, parent: Optional[Node]) -> None:
        if parent is not None:
            ax.plot([parent.x, node.x], [parent.y, node.y],
                    color="#888888", lw=0.8)
        c = color_map[node.state]
        bbox = dict(boxstyle="round,pad=0.4", facecolor=c, alpha=0.85,
                    edgecolor="black", lw=0.6)
        ax.text(node.x, node.y, f"{node.label}\nscore={node.score:.2f}",
                ha="center", va="center", fontsize=8, bbox=bbox,
                color="white" if c != "#9E9E9E" else "black")
        for child in node.children:
            walk(child, node)

    walk(root, None)
    ax.set_xlim(-0.5, 4)
    ax.set_axis_off()
    ax.set_title(
        "Tree-of-Thought  ·  Game of 24  with [4, 6, 8, 8]\n"
        "灰=继续探索   红=被剪枝（评分低）   绿=找到解",
        fontsize=12)

    legend_elements = [
        mpatches.Patch(color="#9E9E9E", label="explore (mid score)"),
        mpatches.Patch(color="#D62728", label="prune (low score)"),
        mpatches.Patch(color="#2CA02C", label="solution (high score)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    nums = [4, 6, 8, 8]
    sol = solve_24(nums)
    print(f"Brute-force solution for {nums}: {sol} = {safe_eval(sol):.0f}")

    # Other classic 24-game examples
    for ex in [[3, 3, 8, 8], [1, 5, 5, 5], [4, 7, 8, 8]]:
        s = solve_24(ex)
        print(f"  {ex}: {s}")

    out = os.path.join(OUT, "tot_game24.png")
    draw_tree(build_demo_tree(), out)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
