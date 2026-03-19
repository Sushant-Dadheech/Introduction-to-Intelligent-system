# BFS, DFS and A* Algorithms for TSP
# Name: Sushant Dadheech

graph = {
    0: {1: 12, 2: 10, 3: 19},
    1: {0: 12, 2: 8,  3: 15},
    2: {0: 10, 1: 8,  3: 6},
    3: {0: 19, 1: 15, 2: 6}
}
cities = list(graph.keys())
start = 0

# 1️⃣ BFS
def bfs_tsp():
    queue = [([start], 0)]
    min_cost = float('inf')
    best_path = None
    while queue:
        path, cost = queue.pop(0)
        if len(path) == len(cities):
            total_cost = cost + graph[path[-1]][start]
            if total_cost < min_cost:
                min_cost = total_cost
                best_path = path + [start]
            continue
        for city in cities:
            if city not in path:
                queue.append((path + [city], cost + graph[path[-1]][city]))
    return best_path, min_cost

print("🔵 BFS Output:", bfs_tsp())

# 2️⃣ DFS
def dfs_tsp():
    stack = [([start], 0)]
    min_cost = float('inf')
    best_path = None
    while stack:
        path, cost = stack.pop()
        if len(path) == len(cities):
            total_cost = cost + graph[path[-1]][start]
            if total_cost < min_cost:
                min_cost = total_cost
                best_path = path + [start]
            continue
        for city in cities:
            if city not in path:
                stack.append((path + [city], cost + graph[path[-1]][city]))
    return best_path, min_cost

print("🟢 DFS Output:", dfs_tsp())

# 3️⃣ A*
import heapq

def heuristic(city, unvisited):
    if not unvisited:
        return graph[city][start]
    return min(graph[city][u] for u in unvisited)

def astar_tsp():
    priority_queue = [(0, [start], 0)]
    min_cost = float('inf')
    best_path = None
    while priority_queue:
        f, path, g = heapq.heappop(priority_queue)
        if len(path) == len(cities):
            total_cost = g + graph[path[-1]][start]
            if total_cost < min_cost:
                min_cost = total_cost
                best_path = path + [start]
            continue
        unvisited = [c for c in cities if c not in path]
        for city in unvisited:
            new_g = g + graph[path[-1]][city]
            h = heuristic(city, [c for c in unvisited if c != city])
            heapq.heappush(priority_queue, (new_g + h, path + [city], new_g))
    return best_path, min_cost

print("🔴 A* Output:", astar_tsp())
