from common import *
import numpy as np
import threading
import queue

class GameTreeNode:
    def __init__(self):
        self.ns = np.array([])
        self.ws = np.array([])
        self.qs = np.array([])
        self.ps = np.array([])
        self.moves = []
        self.move_to_idx = {}
        self.children = []
        self.lock = threading.Lock()

    def add_edge(self, move, n, w, q, p):
        self.move_to_idx[move] = len(self.moves)
        self.moves.append(move)
        self.ns = np.append(self.ns, n)
        self.ws = np.append(self.ws, w)
        self.qs = np.append(self.qs, q)
        self.ps = np.append(self.ps, p)
        self.children.append(GameTreeNode())

class MCTSAgent:
    def __init__(self, side, policy, c_puct, tau, n_virtual_loss,
        use_dirichlet_noise=False):
        self.side = side
        self.policy = policy
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
            with cur.lock:
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
            with cur.lock:
                cur.ns[idx] += 1 - self.n_virtual_loss
                cur.ws[idx] += leaf_value + self.n_virtual_loss
                cur.qs[idx] = cur.ws[idx] / cur.ns[idx]
                cur = cur.children[idx]

    def search_step(self, thread_idx, eval_in_queue, eval_out_queue):
        # search for leaf
        leaf, leaf_board, path = self.select()
        # eval leaf
        if not leaf_board.is_game_over():
            if eval_in_queue is not None:
                eval_in_queue.put((leaf_board, thread_idx))
                action, value = eval_out_queue.get()
            else:
                action, value = self.eval_board(leaf_board)
            # expand leaf
            action = np.exp(action)
            legal_moves = leaf_board.legal_moves
            if self.use_dirichlet_noise and leaf is self.root:
                eta = np.random.dirichlet([0.03], len(legal_moves))
            with leaf.lock:
                for i, move in enumerate(legal_moves):
                    prior_p = action[move_to_action_idx(move)]
                    if self.use_dirichlet_noise and leaf is self.root:
                        prior_p = 0.75*prior_p + 0.25*eta[i]
                    leaf.add_edge(move, 0, 0, 0, prior_p)
        else:
            value = reward_for_side(leaf_board, self.side)
        # backup
        self.backup(path, value)

    def eval_board(self, board):
        board_t = boards_to_tensor([board])
        with torch.no_grad():
            action, value = self.policy(board_t)
        action = action.squeeze(0).numpy()
        value = value[0].item()
        value *= (1 if self.side == board.turn else -1)
        return action, value

    def eval_boards_async(self, eval_in_queue, eval_out_queues, batch_size=8):
        boards, idxs = [], []
        n_threads = self.n_threads_left
        while self.n_threads_left > 0:
            if not eval_in_queue.empty():
                board, idx = eval_in_queue.get()
                boards.append(board)
                idxs.append(idx)
            if boards and (self.n_threads_left < batch_size or
                len(boards) >= batch_size):
                batch_boards, boards = boards[:batch_size], boards[batch_size:]
                batch_idxs, idxs = idxs[:batch_size], idxs[batch_size:]

                boards_t = boards_to_tensor(batch_boards)
                with torch.no_grad():
                    actions, values = self.policy(boards_t)
                actions = actions.numpy()
                values = values.numpy()
                for i in range(len(actions)):
                    v = values[i][0] * (1 if self.side == batch_boards[i].turn
                        else -1)
                    eval_out_queues[batch_idxs[i]].put((actions[i], v))

    # n_threads=0 for synchronous implementation
    def search(self, n_steps, n_threads=0):
        if n_threads == 0:
            for i in range(n_steps):
                self.search_step(0, None, None)
        else:
            # note: not exact number of steps since might not divide evenly
            self.n_threads_left = n_threads
            n_steps_per_thread = n_steps // n_threads
            def search_n_steps(n, thread_idx, eval_in_queue, eval_out_queue,
                done_lock):
                for i in range(n):
                    self.search_step(thread_idx, eval_in_queue, eval_out_queue)
                with done_lock:
                    self.n_threads_left -= 1
            search_threads = []
            eval_in_queue = queue.Queue()
            eval_out_queues = []
            done_lock = threading.Lock()
            for i in range(n_threads):
                # all leftover steps go to 0th thread
                n_steps_real = n_steps_per_thread if i > 0 else \
                    n_steps - (n_threads-1)*n_steps_per_thread
                eval_out_queue = queue.Queue()
                eval_out_queues.append(eval_out_queue)
                thread = threading.Thread(target=search_n_steps,
                    args=(n_steps_real, i, eval_in_queue, eval_out_queue,
                        done_lock))
                thread.start()
                search_threads.append(thread)
            eval_thread = threading.Thread(target=self.eval_boards_async,
                args=(eval_in_queue, eval_out_queues))
            eval_thread.start()
            eval_thread.join()
            for thread in search_threads:
                thread.join()

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

if __name__ == "__main__":
    agent = MCTSAgent(True, Policy(), 0.5, 1, 3)
    for i in range(10):
        agent.search(100)
        print(agent.choose_move())

