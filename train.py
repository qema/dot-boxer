from common import *
import threading
import random

class TrainWorker(mp.Process):
    def __init__(self, game, minibatch_queue, notify_queue, policy):
        super(TrainWorker, self).__init__()
        self.minibatch_queue = minibatch_queue
        self.notify_queue = notify_queue
        self.policy = policy
        self.game = game

    def run(self):
        opt = optim.SGD(self.policy.parameters(), lr=1e-2, momentum=0.9,
            weight_decay=1e-4)
        #opt = optim.Adam(self.policy.parameters(), lr=1e-2)
        while True:
            boards, dists, rewards = self.minibatch_queue.get()
            boards_t = self.game.boards_to_tensor(boards)
            dists_t = torch.from_numpy(dists).type(torch.float).to(
                get_device())
            rewards_t = torch.from_numpy(rewards).type(torch.float).to(
                get_device())

            # train
            self.policy.zero_grad()
            action, value = self.policy(boards_t)
            value = value.flatten()
            ce_loss = (action * dists_t).sum(dim=1)
            ce_loss = -torch.mean(ce_loss)
            #print(ce_loss.item(), F.mse_loss(value, rewards_t).item())
            loss = F.mse_loss(value, rewards_t) + ce_loss
            loss.backward()
            opt.step()

            self.notify_queue.put(loss.item())

class TrainManager(mp.Process):
    def __init__(self, game, game_queue, policy, n_workers=1,
        start_t=1000, buffer_size=10000, max_queue_size=8,
        minibatch_size=32, save_interval=1000):
        super(TrainManager, self).__init__()
        self.game = game
        self.game_queue = game_queue
        self.policy = policy
        self.game_moves = [None]*buffer_size
        self.game_dists = [None]*buffer_size
        self.game_starts = {}
        self.total_n_moves = 0
        # rewards are relative to player A
        self.game_rewards = np.zeros(buffer_size, dtype=np.int8)
        self.game_next_free_idx = 0
        self.start_t = start_t
        self.buffer_size = buffer_size
        self.n_workers = n_workers
        self.max_queue_size = max_queue_size
        self.minibatch_size = minibatch_size
        self.save_interval = save_interval

    def retrieve_new_games(self, has_games_event):
        while True:
            moves, dists, reward = self.game_queue.get()
            idx = self.game_next_free_idx % self.buffer_size
            if self.game_moves[idx] is not None:
                self.total_n_moves -= len(self.game_moves[idx])
            self.game_rewards[idx] = reward
            self.game_moves[idx] = moves
            self.game_dists[idx] = dists
            self.game_starts[self.total_n_moves] = idx
            self.game_next_free_idx += 1
            self.total_n_moves += len(moves)
            if self.game_next_free_idx >= self.start_t and \
                not has_games_event.is_set():
                print("start training")
                has_games_event.set()

    def get_minibatches(self, has_games_event, minibatch_queue):
        has_games_event.wait()
        while True:
            move_idxs = random.sample(range(self.total_n_moves),
                self.minibatch_size)
            boards, dists, rewards = [], [], []
            for move_idx in move_idxs:
                board = self.game.Board()
                move_stack = []
                cur = move_idx
                # TODO: speed up
                while cur not in self.game_starts:
                    cur -= 1
                idx = self.game_starts[cur]
                offset = move_idx - cur
                dist = self.game_dists[idx][offset]
                for move in self.game_moves[idx][:offset]:
                    board.push(move)
                reward = self.game_rewards[idx]
                reward *= (1 if board.turn else -1)
                boards.append(board)
                dists.append(dist)
                rewards.append(reward)
            dists = np.stack(dists)
            rewards = np.array(rewards)
            minibatch_queue.put((boards, dists, rewards))

    def submit_models(self, notify_queue):
        t = 0
        while True:
            loss = notify_queue.get()
            if t % self.save_interval == 0:
                torch.save(self.policy.state_dict(), "models/alpha.pt")
                print("save {:.6f}".format(loss))
            t += 1

    def run(self):
        # check for new games thread
        has_games_event = threading.Event()
        new_games_thread = threading.Thread(target=self.retrieve_new_games,
            args=(has_games_event,))
        new_games_thread.start()

        # make minibatches thread
        minibatch_queue = mp.Queue(self.max_queue_size)
        get_minibatches_thread = threading.Thread(target=self.get_minibatches,
            args=(has_games_event, minibatch_queue))
        get_minibatches_thread.start()

        # start train workers
        notify_queue = mp.Queue()
        workers = []
        for proc_idx in range(self.n_workers):
            worker = TrainWorker(self.game, minibatch_queue, notify_queue,
                self.policy)
            worker.start()
            workers.append(worker)

        # submit new models thread
        submit_models_thread = threading.Thread(target=self.submit_models,
            args=(notify_queue,), daemon=True)
        submit_models_thread.start()

        # join
        new_games_thread.join()
        get_minibatches_thread.join()
        submit_models_thread.join()
        for worker in workers:
            worker.join()
