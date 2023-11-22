from threading import Thread
from time import perf_counter
from matplotlib import pyplot as plt


def bench(queue, n = 1000):
    def producer():
        for i in range(int(n)):
            queue.enqueue(i)
        queue.enqueue(None)

    def consumer():
        while queue.dequeue() is not None:
            pass

    threads = Thread(target=producer), Thread(target=consumer)

    print(f"Starting {queue.__class__.__name__} with {int(n)} items")
    start = perf_counter()
    [th.start() for th in threads]
    [th.join() for th in threads]
    end = perf_counter()

    res = end - start
    print(f"{queue.__class__.__name__}: {res:.3f}s")
    return res


class List:
    def __init__(self) -> None:
        self._queue = []

    def enqueue(self, item):
        self._queue.insert(0, item)

    def dequeue(self):
        self._queue.pop()


class Queue:
    def __init__(self) -> None:
        from queue import Queue
        self._queue = Queue()

    def enqueue(self, item):
        self._queue.put(item)

    def dequeue(self):
        self._queue.get()


class Deque:
    def __init__(self) -> None:
        from collections import deque
        self._queue = deque()

    def enqueue(self, item):
        self._queue.append(item)

    def dequeue(self):
        self._queue.pop()


def main():
    sizes = 1e3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5, 2e5, 5e5, 1e6, 2e6, 5e6
    sizes_list = sizes[:8]
    list_times = [bench(List(), n) for n in sizes_list]
    list_times += [float('nan')]*(len(sizes) - len(sizes_list))
    queue_times = [bench(Queue(), n) for n in sizes]
    deque_times = [bench(Deque(), n) for n in sizes]
    plt.gca().set_xscale('log')
    plt.plot(sizes, list_times, 'r.-')
    plt.plot(sizes, queue_times, 'm.-')
    plt.plot(sizes, deque_times, 'b.-')
    plt.title('Queue VS Deque')
    plt.ylabel('time (s)')
    plt.xlabel('items')
    plt.legend(('List', 'Queue', 'Deque'))
    plt.show()


if __name__ == "__main__":
    main()
