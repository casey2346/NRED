import random

def synthetic_parity(n=100):
    tasks = []
    for _ in range(n):
        x = random.randint(1, 200)
        ans = "even" if x % 2 == 0 else "odd"
        tasks.append((f"Is {x} even or odd?", ans))
    return tasks


def synthetic_arithmetic(n=100):
    tasks = []
    for _ in range(n):
        a,b,c = random.randint(1,20), random.randint(1,20), random.randint(1,10)
        ans = str(a + b - c)
        tasks.append((f"Compute ({a}+{b})-{c} = ?", ans))
    return tasks
