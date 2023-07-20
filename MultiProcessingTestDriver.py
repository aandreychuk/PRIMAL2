import json
import warnings
import multiprocessing
import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore', category=Warning)
from pathos.multiprocessing import ProcessPool as Pool
import argparse
import subprocess

def run_tests(args):
    num_core = args.num_worker
    parallel_pool = Pool(num_core)

    with open('instances_den.json', 'r') as f:
        data = json.load(f)
    instances = []
    for n in args.num_agents:
        for d in data:
            instances.append((d['map_name'], d['seed'], n))

    print('start testing with ' + str(num_core) + ' processes...')
    allResults = []
    for map_name, seed, num_agents in instances:
        result = parallel_pool.apipe(run_1_test_wrapper, args, map_name, seed, num_agents)
        allResults.append(result)

    totalJobs = len(allResults)
    jobsCompleted = 0
    while len(allResults) > 0:
        for i in range(len(allResults)):
            if allResults[i].ready():
                jobsCompleted += 1
                print("{} / {}".format(jobsCompleted, totalJobs))
                allResults[i].get()
                allResults.pop(i)
                break

    parallel_pool.close()


def run_1_test_wrapper(args, name, seed, num_agents):
    """
    Calls TestingEnv.py in a subprocess.
    This approach avoids any multiprocessing issues with tensorflow
    """
    s = "python3 TestingEnv.py -r {resume_testing} -g {GIF_prob} " \
        + "-p {planner} -m {mapName} -s {seed} -n {num_agents}"

    s = s.format(resume_testing=args.resume_testing, GIF_prob=args.GIF_prob,
                 planner=args.planner, mapName=name, seed=seed, num_agents=num_agents)
    if args.printInfo:
        subprocess.run(s, stderr=subprocess.STDOUT, shell=True)
    else:
        try:
            devNull = open('/dev/null', 'w')
            subprocess.run(s, stderr=devNull, shell=True)
        except Exception:
            Warning('cannot mute info, reset printInfo to True')
            subprocess.run(s, stderr=subprocess.STDOUT, shell=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_worker", default=5, type=int)
    parser.add_argument("--printInfo", default=True, type=bool)
    parser.add_argument("-r", "--resume_testing", default=True, help="resume testing")
    parser.add_argument("-g", "--GIF_prob", default=0., help="prob to write GIF")
    parser.add_argument("-p", "--planner", default='mstar', help="choose between mstar and RL")
    parser.add_argument('-n', '--num_agents', nargs='+', required=True, type=int)
    args = parser.parse_args()

    run_tests(args)
