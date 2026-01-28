import random
import numpy as np
from tqdm import tqdm
import heapq
from collections import defaultdict
import json


class DijkstraWithStats:
    """
    dijkstra's algorithm implementation that tracks decrease-key operations.

    uses a min-heap priority queue. since python's heapq doesn't support
    decrease-key directly, we use lazy deletion: when we find a shorter path,
    we add a new entry to the heap and mark the old distance as stale.
    we count a "decrease-key" operation whenever we successfully relax an edge
    (i.e., find a shorter path to a vertex).
    """

    def __init__(self):
        self.decrease_key_count = 0
        self.total_relaxations = 0

    def run(self, adj_list, source, n):
        """
        run dijkstra's algorithm from source vertex.

        args:
            adj_list: adjacency list where adj_list[u] = [(v, weight), ...]
            source: starting vertex (0-indexed)
            n: number of vertices

        returns:
            dist: array of shortest distances from source
        """
        self.decrease_key_count = 0
        self.total_relaxations = 0

        # initialize distances to infinity, source to 0
        dist = [float("inf")] * n
        dist[source] = 0

        # min-heap: (distance, vertex)
        pq = [(0, source)]
        visited = [False] * n

        while pq:
            d, u = heapq.heappop(pq)

            # skip if already processed (stale entry)
            if visited[u]:
                continue
            visited[u] = True

            # relax all outgoing edges
            for v, weight in adj_list[u]:
                self.total_relaxations += 1
                new_dist = d + weight

                # if we found a shorter path, this is a decrease-key operation
                if new_dist < dist[v]:
                    self.decrease_key_count += 1
                    dist[v] = new_dist
                    heapq.heappush(pq, (new_dist, v))

        return dist


def generate_complete_graph(n, seed=None):
    """
    generate a complete directed graph with n vertices.
    edge weights are uniformly random in [0, 1].
    """
    if seed is not None:
        random.seed(seed)

    adj_list = defaultdict(list)

    for u in range(n):
        for v in range(n):
            if u != v:
                weight = random.random()
                adj_list[u].append((v, weight))

    return adj_list


def run_experiments(n_values, num_trials=30):
    """
    run dijkstra's algorithm on complete graphs of various sizes.

    args:
        n_values: list of graph sizes to test
        num_trials: number of random graphs per size

    returns:
        results: dict with statistics for each n
    """
    results = {
        "n_values": n_values,
        "avg_decrease_keys": [],
        "std_decrease_keys": [],
        "min_decrease_keys": [],
        "max_decrease_keys": [],
        "worst_case": [],
        "all_trials": [],  # raw data for each trial
    }

    dijkstra = DijkstraWithStats()

    for n in n_values:
        print(f"\ntesting n = {n}")
        decrease_key_counts = []

        for trial in tqdm(range(num_trials), desc=f"n={n}"):
            adj_list = generate_complete_graph(n, seed=trial * 1000 + n)
            dijkstra.run(adj_list, source=0, n=n)
            decrease_key_counts.append(dijkstra.decrease_key_count)

        avg = np.mean(decrease_key_counts)
        std = np.std(decrease_key_counts)
        num_edges = n * (n - 1)

        results["avg_decrease_keys"].append(float(avg))
        results["std_decrease_keys"].append(float(std))
        results["min_decrease_keys"].append(int(min(decrease_key_counts)))
        results["max_decrease_keys"].append(int(max(decrease_key_counts)))
        results["worst_case"].append(num_edges)
        results["all_trials"].append(decrease_key_counts)

        print(
            f"  avg decrease-keys: {avg:.1f}, worst-case: {num_edges}, ratio: {avg/num_edges:.4f}"
        )

    return results


def save_results(results, filepath="out/hw2_dijkstra_results.json"):
    """save results to json for later analysis."""
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"results saved to {filepath}")


def load_results(filepath="out/hw2_dijkstra_results.json"):
    """load results from json."""
    with open(filepath, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    # test sizes for clear scaling analysis
    n_values = [50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 2500, 3000]
    num_trials = 30

    print("=" * 70)
    print("CSE 202 HW2 - Dijkstra Decrease-Key Analysis")
    print("=" * 70)
    print(f"graph type: complete directed graph")
    print(f"edge weights: uniform random in [0, 1]")
    print(f"sizes: {n_values}")
    print(f"trials per size: {num_trials}")
    print("=" * 70)

    results = run_experiments(n_values, num_trials)
    save_results(results)

    print("\n" + "=" * 70)
    print("data collection complete. run the notebook for visualization.")
    print("=" * 70)
