import os
import argparse
from PRIMAL2_Observer import PRIMAL2_Observer
from Observer_Builder import DummyObserver
import tensorflow as tf
from ACNet import ACNet
from Map_Generator import *
from Env_Builder import *
import yaml
import json
from os.path import exists
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore', category=Warning)


class RL_Planner(MAPFEnv):
    """
    result saved for NN Continuous Planner:
        target_reached      [int ]: num_target that is reached during the episode.
                                    Affected by timeout or non-solution
        computing_time_list [list]: a computing time record of each run of M*
        num_crash           [int ]: number of crash during the episode
        episode_status      [str ]: whether the episode is 'succeed', 'timeout' or 'no-solution'
        succeed_episode     [bool]: whether the episode is successful (i.e. no timeout, no non-solution) or not
        step_count          [int ]: num_step taken during the episode. The 64 timeout step is included
        frames              [list]: list of GIP frames
    """

    def __init__(self, observer, model_path, IsDiagonal=False, isOneShot=True, frozen_steps=0,
                 gpu_fraction=0.04):
        super().__init__(observer=observer, map_generator=DummyGenerator(), num_agents=1,
                         IsDiagonal=IsDiagonal, frozen_steps=frozen_steps, isOneShot=isOneShot)

        self._set_testType()
        self._set_tensorflow(model_path, gpu_fraction)

    def _set_testType(self):
        self.ACTION_COST, self.GOAL_REWARD, self.COLLISION_REWARD = 0, 0.5, 1
        self.test_type = 'oneShot' if self.isOneShot else 'continuous'
        self.method = '_' + self.test_type + 'RL'

    def _set_tensorflow(self, model_path, gpu_fraction):
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = gpu_fraction
        self.sess = tf.Session(config=config)

        # todo:HAS TO BE ENTERED MANUALLY TO MATCH THE MODEL, to be read from DRLMAPF...
        self.num_channels = 11

        self.network = ACNet("global", a_size=5, trainer=None, TRAINING=False,
                             NUM_CHANNEL=self.num_channels,
                             OBS_SIZE=self.observer.observation_size,
                             GLOBAL_NET_SCOPE="global")

        # load the weights from the checkpoint (only the global ones!)
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver = tf.train.Saver()
        saver.restore(self.sess, ckpt.model_checkpoint_path)

        self.agent_states = []
        for i in range(self.num_agents):
            rnn_state = self.network.state_init
            self.agent_states.append(rnn_state)

    def set_world(self):
        return

    def give_moving_reward(self, agentID):
        collision_status = self.world.agents[agentID].status
        if collision_status == 0:
            reward = self.ACTION_COST
            self.isStandingOnGoal[agentID] = False
        elif collision_status == 1:
            reward = self.ACTION_COST + self.GOAL_REWARD
            self.isStandingOnGoal[agentID] = True
            self.world.agents[agentID].dones += 1
        else:
            reward = self.ACTION_COST + self.COLLISION_REWARD
            self.isStandingOnGoal[agentID] = False
        self.individual_rewards[agentID] = reward

    def listValidActions(self, agent_ID, agent_obs):
        return

    def _reset(self, map_generator=None, worldInfo=None):
        self.map_generator = map_generator
        if worldInfo is not None:
            self.world = TestWorld(self.map_generator, world_info=worldInfo, isDiagonal=self.IsDiagonal,
                                   isConventional=False)
        else:
            self.world = World(self.map_generator, num_agents=self.num_agents, isDiagonal=self.IsDiagonal)
            raise UserWarning('you are using re-computing env mode')
        self.num_agents = self.world.num_agents
        self.observer.set_env(self.world)
        self.fresh = True
        if self.viewer is not None:
            self.viewer = None
        self.agent_states = []
        for i in range(self.num_agents):
            rnn_state = self.network.state_init
            self.agent_states.append(rnn_state)

    def step_greedily(self, o):
        def run_network(o):
            inputs, goal_pos, rnn_out = [], [], []

            for agentID in range(1, self.num_agents + 1):
                agent_obs = o[agentID]
                inputs.append(agent_obs[0])
                goal_pos.append(agent_obs[1])
            # compute up to LSTM in parallel
            h3_vec = self.sess.run([self.network.h3],
                                   feed_dict={self.network.inputs: inputs,
                                              self.network.goal_pos: goal_pos})
            h3_vec = h3_vec[0]
            # now go all the way past the lstm sequentially feeding the rnn_state
            for a in range(0, self.num_agents):
                rnn_state = self.agent_states[a]
                lstm_output, state = self.sess.run([self.network.rnn_out, self.network.state_out],
                                                   feed_dict={self.network.inputs: [inputs[a]],
                                                              self.network.h3: [h3_vec[a]],
                                                              self.network.state_in[0]: rnn_state[0],
                                                              self.network.state_in[1]: rnn_state[1]})
                rnn_out.append(lstm_output[0])
                self.agent_states[a] = state
            # now finish in parallel
            policy_vec = self.sess.run([self.network.policy],
                                       feed_dict={self.network.rnn_out: rnn_out})
            policy_vec = policy_vec[0]
            action_dict = {agentID: np.argmax(policy_vec[agentID - 1]) for agentID in range(1, self.num_agents + 1)}
            return action_dict

        numCrashedAgents, computing_time = 0, 0

        start_time = time.time()
        action_dict = run_network(o)
        computing_time = time.time() - start_time

        next_o, reward = self.step_all(action_dict)

        for agentID in reward.keys():
            if reward[agentID] // 1 != 0:
                numCrashedAgents += 1
        assert numCrashedAgents <= self.num_agents

        return numCrashedAgents, computing_time, next_o

    def find_path(self, max_length, saveImage, time_limit=np.Inf):
        assert max_length > 0
        step_count, num_crash, computing_time_list, frames = 0, 0, [], []
        episode_status = 'no early stop'

        obs = self._observe()
        for step in range(1, max_length + 1):
            if saveImage:
                frames.append(self._render(mode='rgb_array'))
            numCrash_AStep, computing_time, obs = self.step_greedily(obs)

            computing_time_list.append(computing_time)
            num_crash += numCrash_AStep
            step_count = step

            if time_limit < computing_time:
                episode_status = "timeout"
                break

        if saveImage:
            frames.append(self._render(mode='rgb_array'))

        target_reached = 0
        for agentID in range(1, self.num_agents + 1):
            target_reached += self.world.getDone(agentID)
        return [target_reached,  # target_reached
                computing_time_list,  # computing_time_list
                num_crash,  # num_crash
                episode_status,  # episode_status
                episode_status == 'no early stop',  # succeed_episode
                step_count,  # step_count
                frames]


class MstarContinuousPlanner(MAPFEnv):
    def __init__(self, IsDiagonal=False, frozen_steps=0):
        super().__init__(observer=DummyObserver(), map_generator=DummyGenerator(), num_agents=1,
                         IsDiagonal=IsDiagonal, frozen_steps=frozen_steps, isOneShot=False)
        self._set_testType()

    def set_world(self):
        return

    def give_moving_reward(self, agentID):
        collision_status = self.world.agents[agentID].status
        if collision_status == 0:
            reward = self.ACTION_COST
            self.isStandingOnGoal[agentID] = False
        elif collision_status == 1:
            reward = self.ACTION_COST + self.GOAL_REWARD
            self.isStandingOnGoal[agentID] = True
            self.world.agents[agentID].dones += 1
        else:
            reward = self.ACTION_COST + self.COLLISION_REWARD
            self.isStandingOnGoal[agentID] = False
        self.individual_rewards[agentID] = reward

    def listValidActions(self, agent_ID, agent_obs):
        return

    def _set_testType(self):
        self.ACTION_COST, self.GOAL_REWARD, self.COLLISION_REWARD = 0, 0.5, 1
        self.test_type = 'continuous'
        self.method = '_' + self.test_type + 'mstar'

    def _reset(self, map_generator=None, starts=None, goals=None, all_goals=None):
        self.map_generator = map_generator
        self.world = World(self.map_generator, starts=starts, goals=goals, all_goals=all_goals, num_agents=self.num_agents, isDiagonal=self.IsDiagonal)
        self.num_agents = self.world.num_agents
        self.observer.set_env(self.world)
        self.fresh = True
        if self.viewer is not None:
            self.viewer = None

    def find_path(self, max_length, saveImage, time_limit=300):
        """
        end episode when 1. max_length is reached immediately, or
                         2. 64 steps after the first timeout, or
                         3. non-solution occurs immediately

        target_reached      [int ]: num_target that is reached during the episode.
                                    Affected by timeout or non-solution
        computing_time_list [list]: a computing time record of each run of M*
        num_crash           [int ]: zero crash in M* mode
        episode_status      [str ]: whether the episode is 'succeed', 'timeout' or 'no-solution'
        succeed_episode     [bool]: whether the episode is successful or not
        step_count          [int ]: num_step taken during the episode. The 64 timeout step is included
        frames              [list]: list of GIP frames
        """

        def parse_path(path, step_count):
            on_goal = False
            path_step = 0
            while step_count < max_length and not on_goal:
                actions = {}
                for i in range(self.num_agents):
                    agent_id = i + 1
                    next_pos = path[path_step][i]
                    diff = tuple_minus(next_pos, self.world.getPos(agent_id))
                    actions[agent_id] = dir2action(diff)

                    if self.world.agents[agent_id].goal_pos == next_pos and not on_goal:
                        on_goal = True

                self.step_all(actions, check_col=False)
                if saveImage:
                    frames.append(self._render(mode='rgb_array'))

                step_count += 1
                path_step += 1
            return step_count if step_count <= max_length else max_length

        def compute_path_piece(time_limit):
            succeed = True
            start_time = time.time()
            path = self.expert_until_first_goal(inflation=3.0, time_limit=time_limit / 5.0)
            # /5 bc we first try C++ M* with 5x less time, then fall back on python if need be where we remultiply by 5
            c_time = time.time() - start_time
            if c_time > time_limit or path is None:
                succeed = False
            return path, succeed, c_time

        assert max_length > 0
        frames, computing_time_list = [], []
        target_reached, step_count, episode_status = 0, 0, 'succeed'

        while step_count < max_length:
            path_piece, succeed_piece, c_time = compute_path_piece(time_limit)
            computing_time_list.append(c_time)
            if not succeed_piece:  # no solution, skip out of loop
                if c_time > time_limit:  # timeout, make a last computation and skip out of the loop
                    episode_status = 'timeout'
                    break
                else:  # no solution
                    episode_status = 'no-solution'
                    break
            else:
                step_count = parse_path(path_piece, step_count)

        for agentID in range(1, self.num_agents + 1):
            target_reached += self.world.getDone(agentID)

        return target_reached, computing_time_list, 0, episode_status, episode_status == 'succeed', step_count, frames


class ContinuousTestsRunner:
    """
    metrics:
        target_reached      [int ]: num_target that is reached during the episode.
                                    Affected by timeout or non-solution
        computing_time_list [list]: a computing time record of each run of M*
        num_crash           [int ]: number of crash during the episode
        episode_status      [str ]: whether the episode is 'succeed', 'timeout' or 'no-solution'
        succeed_episode     [bool]: whether the episode is successful (i.e. no timeout, no non-solution) or not
        step_count          [int ]: num_step taken during the episode. The 64 timeout step is included
        frames              [list]: list of GIP frames
    """

    def __init__(self, env_path, result_path, Planner, resume_testing=False, GIF_prob=0.):
        print('starting {}...'.format(self.__class__.__name__))
        self.env_path = env_path
        self.result_path = result_path
        self.resume_testing = resume_testing
        self.GIF_prob = float(GIF_prob)
        self.worker = Planner

        self.test_method = self.worker.method

        if not os.path.exists(self.result_path):
            os.mkdir(self.result_path)

    def read_single_env(self, name):
        root = self.env_path
        # assert self.worker.test_type == self.test_type
        assert name.split('.')[-1] == 'npy'
        print('loading a single testing env...')
        if self.resume_testing:
            env_name = name[:name.rfind('.')]
            if os.path.exists(self.result_path + env_name + self.test_method + ".txt"):
                return None
        maps = np.load(root + name, allow_pickle=True)
        return maps

    def run_1_test(self, maps, instance, num_agents, seed):
        self.worker.num_agents = num_agents
        self.worker._reset(map_generator=manual_generator(maps[instance['map_name']]),
                           starts={a['agent_id'] + 1: a['start'] for a in instance['task'] if a['agent_id'] < num_agents},
                           goals={a['agent_id'] + 1: a['goals'][0] for a in instance['task'] if a['agent_id'] < num_agents},
                           all_goals={a['agent_id'] + 1: a['goals'][1:] for a in instance['task'] if a['agent_id'] < num_agents})
        max_length = 512
        results = dict()


        print("working on " + instance['map_name'])

        result = self.worker.find_path(max_length=int(max_length), saveImage=False)#np.random.rand() < self.GIF_prob)

        target_reached, computing_time_list, num_crash, episode_status, succeed_episode, step_count, frames = result
        if exists('results.json'):
            with open('results.json', 'r') as f:
                all_results = json.load(f)
                f.close()
        else:
            all_results = []
        results['results'] = {'avg_throughput':target_reached/512, 'reached_goals':target_reached}
        results['resolved_vars'] = {'algo': 'ODrM*', 'map_name': instance['map_name'], 'num_agents': num_agents, 'seed': seed}
        results['id'] = len(all_results)
        all_results.append(results)
        with open('results.json', 'w') as f:
            json.dump(all_results, f, indent=1)
            f.close()

        self.make_gif(frames, instance['map_name'], self.test_method)
        self.write_files(results, instance['map_name'], "_"+str(num_agents))
        return

    def make_gif(self, image, env_name, ext):
        if image:
            gif_name = self.result_path + env_name + ext + ".gif"
            images = np.array(image)
            make_gif(images, gif_name)

    def write_files(self, results, env_name, ext):
        txt_filename = self.result_path + env_name + ext + ".txt"
        f = open(txt_filename, 'w')
        f.write(json.dumps(results))
        f.close()


if __name__ == "__main__":
    import time

    model_path = './model_primal2_continuous/'
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", default="./testing_result/")
    parser.add_argument("--env_path", default='./new_testing_envs/')
    parser.add_argument("-r", "--resume_testing", default=True, help="resume testing")
    parser.add_argument("-g", "--GIF_prob", default=1., help="write GIF")
    parser.add_argument("-t", "--type", default='continuous', help="choose between oneShot and continuous")
    parser.add_argument("-p", "--planner", default='mstar', help="choose between mstar and RL")
    parser.add_argument("-m", "--map_name", default=None, help="single map name for multiprocessing")
    parser.add_argument("-s", "--seed", default=0, type=int)
    parser.add_argument("-n", "--num_agents", default=4, type=int)

    args = parser.parse_args()

    # set a tester--------------------------------------------
    if args.planner == 'mstar':
        print("Starting {} {} tests...".format(args.planner, args.type))
        tester = ContinuousTestsRunner(args.env_path,
                                       args.result_path,
                                       Planner=MstarContinuousPlanner(),
                                       resume_testing=args.resume_testing,
                                       GIF_prob=args.GIF_prob
                                       )

    elif args.planner == 'RL':
        print("Starting {} {} tests...".format(args.planner, args.type))
        tester = ContinuousTestsRunner(args.env_path,
                                       args.result_path,
                                       Planner=RL_Planner(
                                           observer=PRIMAL2_Observer(observation_size=11, num_future_steps=3),
                                           model_path=model_path,
                                           isOneShot=False),
                                       resume_testing=args.resume_testing,
                                       GIF_prob=args.GIF_prob
                                       )
    else:
        raise NameError('invalid planner type')
    # run the tests---------------------------------------------------------
    with open('tasks.json', 'r') as f:
        instances = json.load(f)
    filtered_instances = []

    instance = None
    for i in instances:
        if i['map_name'] == args.map_name and i['seed'] == int(args.seed):
            instance = i
            break
    if instance is None:
        print(f'Required instance with map_name:{args.map_name} and seed:{args.seed} not found')
        exit(1)
    print('running ', args.map_name, args.num_agents, args.seed)
    with open("maps.yaml", 'r') as stream:
        maps = yaml.safe_load(stream)
    tester.run_1_test(maps, instance, args.num_agents, args.seed)
