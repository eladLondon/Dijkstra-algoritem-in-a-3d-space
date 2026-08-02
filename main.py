"""
3D Dijkstra Shortest Path with a live-updating 3D GUI.

- Enter points and edges in the console; the 3D plot builds itself as you type.
- Use the LEFT / RIGHT arrow keys (while the plot window is focused) to spin the view.
- Once you give a start and end point, Dijkstra's algorithm runs and the
  shortest path is highlighted in red on the plot.

Requires matplotlib:  pip install matplotlib
"""

import heapq
import math
import threading

import matplotlib

# Try to grab a common interactive backend; fall back to whatever is default.
for backend in ("TkAgg", "Qt5Agg", "MacOSX"):
    try:
        matplotlib.use(backend)
        break
    except Exception:
        continue

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)


# ----------------------------- Shared state ----------------------------- #

class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.points = []       # list of (x, y, z)
        self.edges = []        # list of (i, j)
        self.path = []         # list of point indices in shortest-path order
        self.start = None
        self.end = None
        self.distance = None
        self.message = "Enter points in the console..."
        self.done = False      # True once the algorithm has finished
        self.azim = -60        # current camera azimuth
        self.elev = 20         # current camera elevation


# ----------------------------- Graph / Dijkstra ----------------------------- #

def euclidean_distance(p1, p2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


def build_graph(points, edges):
    graph = {i: [] for i in range(len(points))}
    for a, b in edges:
        w = euclidean_distance(points[a], points[b])
        graph[a].append((b, w))
        graph[b].append((a, w))  # undirected
    return graph


def dijkstra(graph, start, end):
    n = len(graph)
    dist = {i: math.inf for i in range(n)}
    prev = {i: None for i in range(n)}
    dist[start] = 0
    visited = set()
    pq = [(0, start)]

    while pq:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        if u == end:
            break
        for v, w in graph[u]:
            if v in visited:
                continue
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    if dist[end] == math.inf:
        return None, math.inf

    path = []
    node = end
    while node is not None:
        path.append(node)
        node = prev[node]
    path.reverse()
    return path, dist[end]


def path_edge_set(path):
    return {frozenset((path[i], path[i + 1])) for i in range(len(path) - 1)}


# ----------------------------- Drawing ----------------------------- #

def redraw(ax, state):
    with state.lock:
        points = list(state.points)
        edges = list(state.edges)
        path = list(state.path)
        azim = state.azim
        elev = state.elev
        start = state.start
        end = state.end
        message = state.message
        distance = state.distance

    ax.cla()

    if points:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        zs = [p[2] for p in points]
        ax.scatter(xs, ys, zs, color="steelblue", s=60, depthshade=True)
        for i, (x, y, z) in enumerate(points):
            color = "black"
            if i == start:
                color = "green"
            elif i == end:
                color = "crimson"
            ax.text(x, y, z, f" {i}", color=color, fontsize=10)

    pedges = path_edge_set(path) if path else set()

    # Draw normal edges first
    for a, b in edges:
        if frozenset((a, b)) in pedges:
            continue
        xs = [points[a][0], points[b][0]]
        ys = [points[a][1], points[b][1]]
        zs = [points[a][2], points[b][2]]
        ax.plot(xs, ys, zs, color="gray", linewidth=1.2, alpha=0.6)

    # Draw highlighted shortest-path edges on top
    for a, b in edges:
        if frozenset((a, b)) in pedges:
            xs = [points[a][0], points[b][0]]
            ys = [points[a][1], points[b][1]]
            zs = [points[a][2], points[b][2]]
            ax.plot(xs, ys, zs, color="red", linewidth=3.5)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    title = "3D Points & Edges  (use ← / → to spin)"
    if path:
        title += f"\nShortest path: {' -> '.join(map(str, path))}  (distance: {distance:.4f})"
    elif message:
        title += f"\n{message}"
    ax.set_title(title)

    ax.view_init(elev=elev, azim=azim)


def on_key(event, ax, fig, state):
    if event.key not in ("left", "right"):
        return
    with state.lock:
        if event.key == "left":
            state.azim -= 5
        else:
            state.azim += 5
    redraw(ax, state)
    fig.canvas.draw_idle()


# ----------------------------- Console input thread ----------------------------- #

def input_thread(state):
    print("=== 3D Dijkstra Shortest Path (with live 3D view) ===\n")

    while True:
        try:
            n = int(input("How many 3D points? "))
            if n > 0:
                break
            print("Enter a positive number.")
        except ValueError:
            print("Please enter a valid integer.")

    for i in range(n):
        while True:
            raw = input(f"Point {i} coordinates (x y z): ").strip().split()
            if len(raw) != 3:
                print("Please enter exactly 3 numbers separated by spaces.")
                continue
            try:
                x, y, z = (float(v) for v in raw)
            except ValueError:
                print("Please enter valid numbers.")
                continue
            with state.lock:
                state.points.append((x, y, z))
            break

    print()
    while True:
        try:
            m = int(input("How many lines (edges) connect the points? "))
            if m >= 0:
                break
            print("Enter a non-negative number.")
        except ValueError:
            print("Please enter a valid integer.")

    print(f"Enter each edge as two point indices (0 to {n - 1}), e.g. '0 3'")
    for i in range(m):
        while True:
            raw = input(f"Edge {i}: ").strip().split()
            if len(raw) != 2:
                print("Please enter exactly 2 indices separated by a space.")
                continue
            try:
                a, b = int(raw[0]), int(raw[1])
            except ValueError:
                print("Please enter valid integers.")
                continue
            if not (0 <= a < n and 0 <= b < n) or a == b:
                print(f"Indices must be two different values between 0 and {n - 1}.")
                continue
            with state.lock:
                state.edges.append((a, b))
            break

    print()
    while True:
        try:
            start = int(input(f"Start point index (0 to {n - 1}): "))
            end = int(input(f"End point index (0 to {n - 1}): "))
            if 0 <= start < n and 0 <= end < n:
                break
            print(f"Indices must be between 0 and {n - 1}.")
        except ValueError:
            print("Please enter valid integers.")

    with state.lock:
        state.start = start
        state.end = end
        graph = build_graph(state.points, state.edges)

    path, distance = dijkstra(graph, start, end)

    with state.lock:
        if path is None:
            state.message = f"No path exists between point {start} and point {end}."
            print(f"\n{state.message}")
        else:
            state.path = path
            state.distance = distance
            print(f"\nShortest path from {start} to {end}: {' -> '.join(map(str, path))}")
            print(f"Total distance: {distance:.4f}")
        state.done = True

    print("\nYou can keep spinning the plot with the arrow keys. Close the window to exit.")


# ----------------------------- Main ----------------------------- #

def main():
    state = State()

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Waiting for input in the console...")

    fig.canvas.mpl_connect("key_press_event", lambda event: on_key(event, ax, fig, state))

    t = threading.Thread(target=input_thread, args=(state,), daemon=True)
    t.start()

    plt.ion()
    plt.show()

    # Live-update the plot while the user is still typing in the console.
    while t.is_alive():
        redraw(ax, state)
        plt.pause(0.2)

    # Final draw with the completed path, then hand control fully to the GUI
    # event loop so arrow-key rotation keeps working until the window closes.
    redraw(ax, state)
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()