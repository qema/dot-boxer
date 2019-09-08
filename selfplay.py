from mcts import *

class SelfPlayWorker(mp.Process):
    def __init__(self, game_queue, policy, zero_temp_t=10, n_search_steps=100):
        super(SelfPlayWorker, self).__init__()
        self.zero_temp_t = zero_temp_t
        self.n_search_steps = n_search_steps
        self.game_queue = game_queue
        self.policy = policy

    def run(self):
        agent_a = MCTSAgent(True, self.policy, 0.5, 1, 3,
            use_dirichlet_noise=False)
        agent_b = MCTSAgent(False, self.policy, 0.5, 1, 3,
            use_dirichlet_noise=False)
        while True:
            board = dotboxes.Board()
            agent_a.tau = 1
            agent_b.tau = 1
            t = 0
            moves, dists = [], []
            while not board.is_game_over():
                #if t == self.zero_temp_t:
                #    agent_a.tau = 0
                #    agent_b.tau = 0
                agent = agent_a if board.turn else agent_b
                agent.search(self.n_search_steps)
                move = agent.choose_move()
                dist = agent.move_dist()
                moves.append(move)
                dists.append(dist)

                board.push(move)
                agent_a.commit_move(move)
                agent_b.commit_move(move)
                print(board)
                t += 1
            agent_a.reset()
            agent_b.reset()
            self.game_queue.put((moves, dists, np.sign(board.result())))

class SelfPlayManager(mp.Process):
    def __init__(self, alpha_queue, game_queue):
        super(SelfPlayManager, self).__init__()
        self.alpha_queue = alpha_queue
        self.game_queue = game_queue

    def run(self):
        policy = Policy()
        policy.share_memory()

        search_worker = SelfPlayWorker(self.game_queue, policy)
        search_worker.start()

        while True:
            alpha = self.alpha_queue.get()
            policy.load_state_dict(alpha)

        search_worker.join()
