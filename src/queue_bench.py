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

    start = perf_counter()
    [th.start() for th in threads]
    [th.join() for th in threads]
    end = perf_counter()

    return end - start


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
    sizes = 1e3, 1e4, 1e5, 1e6
    queue_times = [bench(Queue(), n) for n in sizes]
    deque_times = [bench(Deque(), n) for n in sizes]
    plt.gca().set_xscale('log')
    plt.plot(sizes, queue_times, 'r.-')
    plt.plot(sizes, deque_times, 'b.-')
    plt.title('Queue VS Deque')
    plt.ylabel('time (s)')
    plt.xlabel('items')
    plt.legend(('Queue', 'Deque'))
    plt.show()


if __name__ == "__main__":
    main()
