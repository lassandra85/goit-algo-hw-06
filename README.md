Пояснення до Завдання 2 (DFS vs BFS)

Ось порівняння двох алгоритмів пошуку шляхів:

---DFS (глибина): шукає шлях, занурюючись углиб графа. Він може знайти шлях, але не гарантує, що він найкоротший.

---BFS (ширина): шукає всі варіанти рівень за рівнем. Він завжди знаходить найкоротший шлях у ненаправлених графах.

Чому шляхи різні?

DFS може йти випадковим глибоким маршрутом, поки не знайде ціль, а BFS проходить рівнями, тому знаходить оптимальний шлях. У нашому випадку BFS може знайти коротший шлях, ніж DFS.
