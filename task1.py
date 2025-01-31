import networkx as nx
import random
import matplotlib.pyplot as plt
from collections import deque
import heapq

# Завдання 1: Створення соціального графа

G = nx.Graph()

# Додаємо вершини (людей)

people = ["Аня", "Богдан", "Влад", "Галина", "Дмитро", "Єва", "Женя"]

G.add_nodes_from(people)

# Додаємо ребра (наприклад, відстань між людьми)

edges = [("Аня", "Богдан"), ("Аня", "Влад"), ("Богдан", "Галина"),
("Влад", "Дмитро"), ("Галина", "Єва"), ("Єва", "Женя"),
("Женя", "Богдан"), ("Аня", "Галина"), ("Дмитро", "Єва")]

G.add_edges_from(edges)

# Аналіз характеристик графа

print("Кількість вершин:", G.number_of_nodes())
print("Кількість ребер:", G.number_of_edges())
print("Ступені вершин:", dict(G.degree()))

# Завдання 2: Алгоритми DFS і BFS 

def dfs(graph, start, goal, path=None):

    if path is None:
        path = [start]

    if start == goal:
        return path
    
    for neighbor in graph.neighbors(start):
        if neighbor not in path:
            new_path = dfs(graph, neighbor, goal, path + [neighbor])

        if new_path:
            return new_path
    return None

def bfs(graph, start, goal):

    queue = deque([[start]])
    visited = set()

    while queue:
        path = queue.popleft()
        node = path[-1]
    
        if node == goal:
            return path
    
        if node not in visited:
            visited.add(node)
            
            for neighbor in graph.neighbors(node):
                new_path = path + [neighbor]
                queue.append(new_path)

    return None

# Приклад знаходження найкоротшого шляху від Аня до Єва

start_person = "Аня"
goal_person = "Єва"

dfs_path = dfs(G, start_person, goal_person)
bfs_path = bfs(G, start_person, goal_person)

print(f"Шлях DFS від {start_person} до {goal_person}: {dfs_path}")
print(f"Шлях BFS від {start_person} до {goal_person}: {bfs_path}")

# Завдання 3: Алгоритм Дейкстри

# Додаємо ваги до ребер

for edge in edges:
    G[edge[0]][edge[1]]['weight'] = random.randint(1, 10)  # випадкові ваги від 1 до 10


def dijkstra(graph, start):
    distances = {node: float('inf') for node in graph.nodes}
    distances[start] = 0
    priority_queue = [(0, start)]
    previous_nodes = {node: None for node in graph.nodes}
    
    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)
        
        for neighbor in graph.neighbors(current_node):
            weight = graph[current_node][neighbor]['weight']
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous_nodes[neighbor] = current_node
                heapq.heappush(priority_queue, (distance, neighbor))
    
    return distances, previous_nodes

def shortest_path_dijkstra(previous_nodes, start, goal):
    path = []
    current = goal
    while current is not None:
        path.insert(0, current)
        current = previous_nodes[current]
    return path if path[0] == start else None

# Виконання алгоритму Дейкстри для кожної вершини
print("\nНайкоротші шляхи між усіма вершинами (Дейкстра):")
for start in people:
    distances, previous_nodes = dijkstra(G, start)
    for goal in people:
        if start != goal:
            path = shortest_path_dijkstra(previous_nodes, start, goal)
            print(f"{start} -> {goal}: {path} (Довжина: {distances[goal]})")

# Візуалізація графа

pos = nx.spring_layout(G)
plt.figure(figsize=(8, 6))
nx.draw(G, pos, with_labels=True, node_color="skyblue", edge_color="gray", node_size=2000, font_size=12)
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.title("Соціальна мережа з вагами на ребрах")
plt.show()