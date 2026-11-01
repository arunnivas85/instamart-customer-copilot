def fibonacci(n):
    """Return the first n Fibonacci numbers."""
    sequence = []
    a, b = 0, 1
    for _ in range(max(0, n)):
        sequence.append(a)
        a, b = b, a + b
    return sequence


if __name__ == "__main__":
    count = int(input("How many terms? "))
    print(*fibonacci(count))