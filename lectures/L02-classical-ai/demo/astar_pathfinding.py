"""A* 寻路可视化 Demo — 第 2 讲：经典 AI 的黄金年代

交互式网格地图上的 A* 搜索可视化。
展示 open/closed 节点扩展过程，支持与 BFS 对比。

用法：
    python astar_pathfinding.py              # A* 默认演示
    python astar_pathfinding.py --algo bfs   # BFS 对比
    python astar_pathfinding.py --size 30    # 更大的网格
"""

import argparse
import heapq
import time

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[4] / "design" / "meta"))
import cjk_font  # noqa: F401  - applies CJK font config
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

SEED = 1337

# ---------- Grid Setup ----------

def create_grid(size: int, obstacle_ratio: float = 0.25) -> np.ndarray:
    """Create a grid with random obstacles. 0=free, 1=obstacle."""
    rng = np.random.RandomState(SEED)
    grid = (rng.random((size, size)) < obstacle_ratio).astype(int)
    grid[0, 0] = 0          # start
    grid[size-1, size-1] = 0  # goal
    # Clear a corridor to guarantee reachability
    for i in range(size):
        grid[i, 0] = 0
        grid[size-1, i] = 0
    return grid


# ---------- A* Algorithm ----------

def heuristic(a: tuple, b: tuple) -> float:
    """Manhattan distance heuristic."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(grid: np.ndarray, start: tuple, goal: tuple):
    """A* search returning (path, visited_order, open_set_snapshots)."""
    size = grid.shape[0]
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    open_heap = [(heuristic(start, goal), 0, start)]
    came_from = {}
    g_score = {start: 0}
    visited = []
    counter = 1

    while open_heap:
        f, _, current = heapq.heappop(open_heap)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1], visited

        visited.append(current)

        for dx, dy in neighbors:
            nx, ny = current[0] + dx, current[1] + dy
            if 0 <= nx < size and 0 <= ny < size and grid[nx, ny] == 0:
                tentative_g = g_score[current] + 1
                neighbor = (nx, ny)
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_heap, (f_score, counter, neighbor))
                    counter += 1

    return [], visited  # No path found


# ---------- BFS Algorithm ----------

def bfs(grid: np.ndarray, start: tuple, goal: tuple):
    """BFS search returning (path, visited_order)."""
    from collections import deque
    size = grid.shape[0]
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    queue = deque([start])
    came_from = {start: None}
    visited = []

    while queue:
        current = queue.popleft()
        visited.append(current)

        if current == goal:
            path = []
            while current is not None:
                path.append(current)
                current = came_from[current]
            return path[::-1], visited

        for dx, dy in neighbors:
            nx, ny = current[0] + dx, current[1] + dy
            neighbor = (nx, ny)
            if (0 <= nx < size and 0 <= ny < size
                    and grid[nx, ny] == 0 and neighbor not in came_from):
                came_from[neighbor] = current
                queue.append(neighbor)

    return [], visited


# ---------- Visualization ----------

def visualize(grid, path, visited, algo_name, start, goal):
    """Render the search result with color-coded cells."""
    size = grid.shape[0]
    canvas = np.zeros((size, size, 3))

    # Obstacles: dark gray
    canvas[grid == 1] = [0.2, 0.2, 0.2]
    # Free: white
    canvas[grid == 0] = [1.0, 1.0, 1.0]

    # Visited: light blue (with gradient for order)
    for i, (r, c) in enumerate(visited):
        t = i / max(len(visited), 1)
        canvas[r, c] = [0.7 + 0.3*t, 0.85, 1.0 - 0.3*t]

    # Path: yellow
    for r, c in path:
        canvas[r, c] = [1.0, 0.85, 0.0]

    # Start: green, Goal: red
    canvas[start[0], start[1]] = [0.0, 0.8, 0.0]
    canvas[goal[0], goal[1]] = [0.9, 0.1, 0.1]

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(canvas, interpolation='nearest')
    ax.set_title(
        f"{algo_name}  |  Path length: {len(path)}  |  Nodes explored: {len(visited)}",
        fontsize=13, fontweight='bold'
    )
    ax.set_xticks([])
    ax.set_yticks([])

    # Grid lines
    for x in range(size + 1):
        ax.axhline(x - 0.5, color='gray', linewidth=0.3)
        ax.axvline(x - 0.5, color='gray', linewidth=0.3)

    plt.tight_layout()
    safe_name = algo_name.lower().replace("*", "star")
    plt.savefig(f"figures/{safe_name}_result.png", dpi=150, bbox_inches='tight')
    print(f"[saved] figures/{safe_name}_result.png")
    plt.show()


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(description="A* vs BFS pathfinding visualization")
    parser.add_argument("--algo", choices=["astar", "bfs", "both"], default="astar",
                        help="Algorithm to visualize (default: astar)")
    parser.add_argument("--size", type=int, default=20,
                        help="Grid size NxN (default: 20)")
    parser.add_argument("--obstacle-ratio", type=float, default=0.25,
                        help="Ratio of obstacles (default: 0.25)")
    args = parser.parse_args()

    import os
    os.makedirs("figures", exist_ok=True)

    grid = create_grid(args.size, args.obstacle_ratio)
    start = (0, 0)
    goal = (args.size - 1, args.size - 1)

    algos = []
    if args.algo in ("astar", "both"):
        algos.append(("A*", astar))
    if args.algo in ("bfs", "both"):
        algos.append(("BFS", bfs))

    for name, func in algos:
        t0 = time.perf_counter()
        path, visited = func(grid, start, goal)
        elapsed = time.perf_counter() - t0
        print(f"\n{'='*40}")
        print(f"  {name}")
        print(f"  Grid: {args.size}x{args.size}")
        print(f"  Path length: {len(path)}")
        print(f"  Nodes explored: {len(visited)}")
        print(f"  Time: {elapsed*1000:.1f} ms")
        print(f"{'='*40}")
        visualize(grid, path, visited, name, start, goal)


if __name__ == "__main__":
    main()
