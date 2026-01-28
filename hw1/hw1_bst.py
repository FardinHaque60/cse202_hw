import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class BSTNode:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None

class BST:
    def __init__(self):
        self.root = None
        self.comparisons = 0

    def insert(self, key):
        def _insert(node, key):
            if node is None:
                return BSTNode(key)
            self.comparisons += 1
            if key < node.key:
                node.left = _insert(node.left, key) # set node to either my existing left child or new node
            else:
                node.right = _insert(node.right, key)
            return node
        self.root = _insert(self.root, key)

    def max_depth(self):
        def _max_depth(node):
            if node is None:
                return 0
            left_depth = _max_depth(node.left)
            right_depth = _max_depth(node.right)
            return max(left_depth, right_depth) + 1
        return _max_depth(self.root)

def bst_simulate(bst, n):
    nums = list(range(1, n + 1))

    depth_data = []
    comp_data = []
    print("simulating runs for n: ", n)
    for _ in tqdm(range(int(10*np.log(n)))):
        random.shuffle(nums)
        for num in nums:
            bst.insert(num)
        depth_data.append(bst.max_depth())
        comp_data.append(bst.comparisons)

    return depth_data, comp_data

if __name__ == "__main__":
    num_range = [10, 20, 30, 100, 500, 750, 1000, 2500, 4000, 5000, 7500, 10000, 15000, 20000, 40000, 50000]
    max_depth_data = [] # 2d list, each entry is a list of 10 max depths achieved from each run
    comp_data = [] # 2d list with similar data as above

    ''' test if bst works 
    bst = BST()
    nums = [1, -5, 10, -2, -3, -4]
    for num in nums:
        bst.insert(num)
    print(bst.max_depth())
    print(bst.comparisons)
    '''

    for num in num_range:
        bst = BST()
        max_depths, comps = bst_simulate(bst, num)
        max_depth_data.append(max_depths)
        comp_data.append(comps)

    avg_depth_data = []
    avg_comp_data = []
    for depth_data, comps in zip(max_depth_data, comp_data):
        avg_depth_data.append(np.mean(depth_data))
        avg_comp_data.append(np.mean(comps))
    comps_per_el = np.array(avg_comp_data) / np.array(num_range)

    print(avg_comp_data)
    print(avg_depth_data)
    print(comps_per_el)

    # plot data
    plt.figure(figsize=(12, 5))

    # avg depth vs. n (number of items inserted into tree)
    plt.subplot(1, 2, 1)
    plt.plot(num_range, avg_depth_data, marker='o')
    plt.xlabel('n')
    plt.ylabel('Average Max Depth')
    plt.title('Average Max Depth vs n')
    plt.grid(True)

    # avg # of comparisons vs. n
    plt.subplot(1, 2, 2)
    plt.plot(num_range, avg_comp_data, marker='o', color='orange')
    plt.xlabel('n')
    plt.ylabel('Average Comparisons')
    plt.title('Average Comparisons vs n')
    plt.grid(True)

    plt.tight_layout()
    save_path = "out/temp1.png"
    plt.savefig(save_path)
    print(f"saved figure 1 to {save_path}")
    plt.close()

    plt.plot(num_range, comps_per_el, marker='o', color='red')
    plt.xlabel('n')
    plt.ylabel('Average Comparisons per Element')
    plt.title('Average Comparisons per Element vs n')
    plt.grid(True)
    save_path = "out/temp2.png"
    plt.savefig(save_path)
    print(f"saved figure 2 to {save_path}")
