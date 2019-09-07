from common import *
import numpy as np

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
    def __init__(self, side, eval_inbox, eval_outbox, eval_idx,
        c_puct, tau, n_virtual_loss):
        self.side = side
        self.eval_inbox = eval_inbox
        self.eval_outbox = eval_outbox
        self.eval_idx = eval_idx
        self.board = dotboxes.Board()
        self.root = GameTreeNode()
        self.c_puct = c_puct
        self.tau = tau
        self.n_virtual_loss = n_virtual_loss

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
            self.eval_inbox.put((leaf_board, self.eval_idx))
            action, value = self.eval_outbox.recv()
            # expand leaf
            action = torch.exp(action)
            for move in leaf_board.legal_moves:
                leaf.add_edge(move, 0, 0, 0,
                    action[move_to_action_idx(move)].item())
        else:
            value = reward_for_side(leaf_board, self.side)
        # backup
        self.backup(path, value)

    def best_move(self):
        dist = np.power(self.root.ns, 1 / self.tau)
        dist /= np.sum(dist)
        out = np.random.multinomial(1, dist)
        out = np.argmax(out)
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
    def __init__(self, eval_inbox, eval_outbox, eval_idx):
        super(SelfPlayWorker, self).__init__()
        self.agent = MCTSAgent(True, eval_inbox, eval_outbox, eval_idx,
            0.5, 1, 3)

    def run(self):
        while True:
            board = dotboxes.Board()
            while not board.is_game_over():
                if board.turn:
                    for t in range(100):
                        self.agent.search_step()
                    move = self.agent.best_move()
                else:
                    import random
                    move = random.choice(list(board.legal_moves))

                board.push(move)
                self.agent.commit_move(move)
                print(board)
            self.agent.reset()

class LeafEvalWorker(mp.Process):
    def __init__(self, batch_size, eval_inbox, eval_pipes):
        super(LeafEvalWorker, self).__init__()
        self.eval_inbox = eval_inbox
        self.eval_pipes = eval_pipes
        self.batch_size = batch_size
        self.policy = Policy()

    def run(self):
        while True:
            boards, idxs = zip(*[self.eval_inbox.get() for i in
                range(self.batch_size)])
            boards_t = boards_to_tensor(boards)
            with torch.no_grad():
                actions, values = self.policy(boards_t)
            for i in range(len(boards)):
                self.eval_pipes[idxs[i]].send((actions[i], values[i]))

if __name__ == "__main__":
    eval_inbox = mp.Queue()
    eval_pipe_recv, eval_pipe_send = mp.Pipe(duplex=False)
    search_worker = SelfPlayWorker(eval_inbox, eval_pipe_recv, 0)
    eval_worker = LeafEvalWorker(1, eval_inbox, [eval_pipe_send])
    search_worker.start()
    eval_worker.start()
    search_worker.join()
    eval_worker.join()
