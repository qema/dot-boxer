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

    def eval_board(self, board):
        board_t = boards_to_tensor([board])
        with torch.no_grad():
            action, value = self.policy(board_t)
        action = action.squeeze(0)
        value = value[0].item()
        value *= (1 if self.side == board.turn else -1)
        return action, value

    def search_step(self):
        # search for leaf
        leaf, leaf_board, path = self.select()
        # eval leaf
        if not leaf_board.is_game_over():
            action, value = self.eval_board(leaf_board)
            # expand leaf
            action = torch.exp(action)
            legal_moves = leaf_board.legal_moves
            if self.use_dirichlet_noise and leaf is self.root:
                eta = np.random.dirichlet([0.03], len(legal_moves))
            with leaf.lock:
                for i, move in enumerate(legal_moves):
                    prior_p = action[move_to_action_idx(move)].item()
                    if self.use_dirichlet_noise and leaf is self.root:
                        prior_p = 0.75*prior_p + 0.25*eta[i]
                    leaf.add_edge(move, 0, 0, 0, prior_p)
        else:
            value = reward_for_side(leaf_board, self.side)
        # backup
        self.backup(path, value)

    def eval_boards(self):
        # TODO: implement
        pass

    def search(self, n_steps, n_threads=1):
        # TODO: not exact number of steps
        n_steps_per_thread = n_steps // n_threads
        def search_n_steps(n):
            for i in range(n):
                self.search_step()
        search_threads = []
        for i in range(n_threads):
            thread = threading.Thread(target=search_n_steps,
                args=(n_steps_per_thread,))
            thread.start()
            search_threads.append(thread)
        eval_thread = threading.Thread(target=self.eval_boards)
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
    agent.search(100)

