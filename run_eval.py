import os
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
seeds = [_ for _ in range(10)]
num_agents = [4,8,16]
if not os.path.exists('./results'):
    os.makedirs('./results')
for m in maps:
    for s in seeds:
        for n in num_agents:
            os.system(f'python3 eval.py --map_name {m} --seed {s} --num_agents {n}')