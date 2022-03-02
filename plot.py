import os
import re
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import argparse


def parse_file(file):
    with open(file, 'r') as f:
        lines = [line.strip() for line in f.readlines()]

    data = []
    pattern = "Average elasped time: \((.*?)\) second, performance: \((.*?)\) GFLOPS. size: \((.*?)\)."
    for line in lines:
        r = re.match(pattern, line)
        if r:
            gflops = float(r.group(2))
            data.append(gflops)
    return data


def plot(num1, num2, y1, y2, save_dir):
    x = [(i + 1) * 256 for i in range(len(y1))]
    fig = plt.figure(figsize=(12, 10))
    if num1 == 0:
        num1 = "culas"

    plt.plot(x, y1, c='k', linewidth=2, label=f"kernel_{num1}")
    plt.plot(x, y2, c='b', linewidth=2, label=f"kernel_{num2}")
    plt.legend()

    plt.scatter(x, y1, marker="s", s=60, c='', edgecolors='k', linewidth=2)
    plt.scatter(x, y2, marker="^", s=60, c='', edgecolors='b', linewidth=2)

    plt.tick_params(labelsize=10)
    plt.xlabel("Matrix size (M=N=K)", fontsize=12, fontweight='bold')
    plt.ylabel("Performance (GFLOPS)", fontsize=12, fontweight='bold')

    plt.title(f"Comparison bewteen: kernel_{num1} and kernel_{num2}", fontsize=16, fontweight='bold')

    x_major_locator = MultipleLocator(256)
    plt.gca().xaxis.set_major_locator(x_major_locator)

    plt.savefig(f"{save_dir}/kernel_{num1}_vs_{num2}.png")


def main(args):
    root = os.path.dirname(os.path.abspath(__file__))
    data1 = parse_file(os.path.join(root, f'test/test_kernel_{args.one}.txt'))
    data2 = parse_file(os.path.join(root, f'test/test_kernel_{args.another}.txt'))
    plot(args.one, args.another, data1, data2, args.save_dir)


def parse_args():
    parser = argparse.ArgumentParser(description='plot kernel performance')
    parser.add_argument('one', type=int, help='one kernel num')
    parser.add_argument('another', type=int, help='another kernel num')
    parser.add_argument('--save_dir', default='images')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)

# python plot.py 0 1
