import os
import argparse

parser = argparse.ArgumentParser()
maps = ['mazes-s40_wc4_od30',
            'mazes-s41_wc5_od50',
            'mazes-s42_wc7_od30',
            'mazes-s43_wc2_od45',
            'mazes-s44_wc2_od30',
            'mazes-s45_wc4_od55',
            'mazes-s46_wc2_od55',
            'mazes-s47_wc2_od25',
            'mazes-s48_wc3_od65',
            'mazes-s49_wc2_od50']
if not os.path.exists('./results'):
    os.makedirs('./results')
parser.add_argument('-s', '--seed', nargs='+', required=True, type=int)
parser.add_argument('-n', '--num_agents', nargs='+', required=True, type=int)
seeds = parser.parse_args().seed
num_agents = parser.parse_args().num_agents

for m in maps:
    for s in seeds:
        for n in num_agents:
            os.system(f'python3 eval.py --map_name {m} --seed {s} --num_agents {n}')