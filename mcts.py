from common import *
import numpy as np
import threading

class GameTreeNode:
    def __init__(self):
        self.ns = np.array([])
        self.ws = np.array([])
        self.qs = np.array([])
        self.ps = np.array([])
        self.moves = []
        self.move_to_idx = {}
        self.children = []

    def add_edge(self, move, n, w, q, p):
        self.move_to_idx[move] = len(self.moves)
        self.moves.append(move)
        self.ns = np.append(self.ns, n)
        self.ws = np.append(self.ws, w)
        self.qs = np.append(self.qs, q)
        self.ps = np.append(self.ps, p)
        self.children.append(GameTreeNode())

class MCTSAgent:
    def __init__(self, side, eval_queue, eval_outbox, eval_idx,
        c_puct, tau, n_virtual_loss, use_dirichlet_noise=False):
        self.side = side
        self.eval_queue = eval_queue
        self.eval_outbox = eval_outbox
        self.eval_idx = eval_idx
        self.board = dotboxes.Board()
        self.root = GameTreeNode()
        self.c_puct = c_puct
        self.tau = tau
        self.n_virtual_loss = n_virtual_loss
        self.use_dirichlet_noise = use_dirichlet_noise

    def select(self):
        cur = self.root
        board = self.board.clone()
        path = []
        while cur.children:
            frac = np.sqrt(np.sum(cur.ns)) / (1 + cur.ns)
            u = self.c_puct * cur.ps * frac
            scores = cur.qs + u
            best_idx = np.argmax(scores)
            cur.ns[best_idx] += self.n_virtual_loss
            cur.ws[best_idx] -= self.n_virtual_loss
            move = cur.moves[best_idx]
            cur = cur.children[best_idx]
            board.push(move)
            path.append(best_idx)
        return cur, board, path

    def backup(self, path, leaf_value):
        cur = self.root
        for idx in path:
            cur.ns[idx] += 1 - self.n_virtual_loss
            cur.ws[idx] += leaf_value + self.n_virtual_loss
            cur.qs[idx] = cur.ws[idx] / cur.ns[idx]
            cur = cur.children[idx]

    def search_step(self):
        # search for leaf
        leaf, leaf_board, path = self.select()
        # eval leaf
        if not leaf_board.is_game_over():
            self.eval_queue.put((leaf_board, self.eval_idx))
            action, value = self.eval_outbox.recv()
            value = value.item()
            value *= (1 if self.side == leaf_board.turn else -1)
            # expand leaf
            action = torch.exp(action)
            legal_moves = leaf_board.legal_moves
            if self.use_dirichlet_noise and leaf is self.root:
                eta = np.random.dirichlet([0.03], len(legal_moves))
            for i, move in enumerate(legal_moves):
                prior_p = action[move_to_action_idx(move)].item()
                if self.use_dirichlet_noise and leaf is self.root:
                    prior_p = 0.75*prior_p + 0.25*eta[i]
                leaf.add_edge(move, 0, 0, 0, prior_p)
        else:
            value = reward_for_side(leaf_board, self.side)
        # backup
        self.backup(path, value)

    def move_dist(self):
        policy = np.zeros(action_space_size())
        if self.tau != 0:
            dist = np.power(self.root.ns, 1 / self.tau)
            dist /= np.sum(dist)
            for p, move in zip(dist, self.root.moves):
                policy[move_to_action_idx(move)] = p
        else:
            best_idx = np.argmax(self.root.ns)
            policy[move_to_action_idx(self.root.moves[best_idx])] = 1
        return policy

    def choose_move(self):
        if self.tau != 0:
            dist = np.power(self.root.ns, 1 / self.tau)
            dist /= np.sum(dist)
            out = np.random.multinomial(1, dist)
            out = np.argmax(out)
        else:
            out = np.argmax(self.root.ns)
        out = self.root.moves[out]
        return out

    def commit_move(self, move):
        if move in self.root.move_to_idx:
            self.root = self.root.children[self.root.move_to_idx[move]]
        else:
            self.root = GameTreeNode()

        self.board.push(move)

    def reset(self):
        self.board = dotboxes.Board()
        self.root = GameTreeNode()

class SelfPlayWorker(mp.Process):
    def __init__(self, eval_queue, game_queue, eval_pipe, eval_idx,
        zero_temp_t=10, n_search_steps=100):
        super(SelfPlayWorker, self).__init__()
        self.zero_temp_t = zero_temp_t
        self.n_search_steps = n_search_steps
        self.agent_a = MCTSAgent(True, eval_queue, eval_pipe, eval_idx,
            0.5, 1, 3, use_dirichlet_noise=False)
        self.agent_b = MCTSAgent(False, eval_queue, eval_pipe, eval_idx,
            0.5, 1, 3, use_dirichlet_noise=False)
        self.game_queue = game_queue

    def run(self):
        while True:
            board = dotboxes.Board()
            self.agent_a.tau = 1
            self.agent_b.tau = 1
            t = 0
            moves, dists = [], []
            while not board.is_game_over():
                #if t == self.zero_temp_t:
                #    self.agent_a.tau = 0
                #    self.agent_b.tau = 0
                agent = self.agent_a if board.turn else self.agent_b
                for i in range(self.n_search_steps):
                    agent.search_step()
                move = agent.choose_move()
                dist = agent.move_dist()
                moves.append(move)
                dists.append(dist)

                board.push(move)
                self.agent_a.commit_move(move)
                self.agent_b.commit_move(move)
                #print(board)
                t += 1
            self.agent_a.reset()
            self.agent_b.reset()
            self.game_queue.put((moves, dists, np.sign(board.result())))

class LeafEvalWorker(mp.Process):
    # set alpha_pipe to None if not using
    def __init__(self, eval_queue, eval_pipes, alpha_pipe, batch_size=1):
        super(LeafEvalWorker, self).__init__()
        self.eval_queue = eval_queue
        self.eval_pipes = eval_pipes
        self.alpha_pipe = alpha_pipe
        self.batch_size = batch_size
        self.policy = Policy()

    def retrieve_new_alphas(self):
        if self.alpha_pipe is not None:
            while True:
                alpha = self.alpha_pipe.recv()
                self.policy.load_state_dict(alpha)

    def run(self):
        retrieve_alphas_thread = threading.Thread(
            target=self.retrieve_new_alphas, daemon=True)
        retrieve_alphas_thread.start()

        while True:
            boards, idxs = zip(*[self.eval_queue.get() for i in
                range(self.batch_size)])
            boards_t = boards_to_tensor(boards)
            with torch.no_grad():
                actions, values = self.policy(boards_t)
            for i in range(len(boards)):
                self.eval_pipes[idxs[i]].send((actions[i], values[i]))

        retrieve_alphas_thread.join()

class SelfPlayManager(mp.Process):
    def __init__(self, alpha_queue, game_queue):
        super(SelfPlayManager, self).__init__()
        self.alpha_queue = alpha_queue
        self.game_queue = game_queue

    def run(self):
        eval_queue = mp.Queue()
        eval_pipe_recv, eval_pipe_send = mp.Pipe(duplex=False)
        alpha_pipe_recv, alpha_pipe_send = mp.Pipe(duplex=False)
        search_worker = SelfPlayWorker(eval_queue, self.game_queue,
            eval_pipe_recv, 0)
        leaf_eval_worker = LeafEvalWorker(eval_queue, [eval_pipe_send],
            alpha_pipe_recv)
        search_worker.start()
        leaf_eval_worker.start()

        while True:
            alpha = self.alpha_queue.get()
            alpha_pipe_send.send(alpha)

        search_worker.join()
        leaf_eval_worker.join()

if __name__ == "__main__":
    def peek_games(game_queue):
        while True:
            print(game_queue.get())

    eval_queue = mp.Queue()
    game_queue = mp.Queue()
    eval_pipe_recv, eval_pipe_send = mp.Pipe(duplex=False)
    search_worker = SelfPlayWorker(eval_queue, game_queue, eval_pipe_recv, 0)
    eval_worker = LeafEvalWorker(eval_queue, [eval_pipe_send])
    search_worker.start()
    eval_worker.start()
    peek_proc = mp.Process(target=peek_games, args=(game_queue,))
    peek_proc.start()
    peek_proc.join()
    search_worker.join()
    eval_worker.join()
