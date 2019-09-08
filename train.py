from common import *
import threading
import random

class TrainWorker(mp.Process):
    def __init__(self, minibatch_queue, notify_queue, policy):
        super(TrainWorker, self).__init__()
        self.minibatch_queue = minibatch_queue
        self.notify_queue = notify_queue
        self.policy = policy

    def run(self):
        opt = optim.SGD(self.policy.parameters(), lr=1e-2, momentum=0.9,
            weight_decay=1e-4)
        while True:
            boards, dists, rewards = self.minibatch_queue.get()
            boards_t = boards_to_tensor(boards)
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
    def __init__(self, game_queue, trained_queue, n_workers=1,
        start_t=100, max_n_games=100000, max_queue_size=8,
        minibatch_size=32, submit_interval=1000):
        super(TrainManager, self).__init__()
        self.game_queue = game_queue
        self.trained_queue = trained_queue
        self.game_moves = [None]*max_n_games
        self.game_dists = np.zeros((max_n_games, action_space_size()),
            dtype=np.float)
        self.game_rewards = np.zeros(max_n_games, dtype=np.int8)
        self.game_starts = np.zeros(max_n_games, dtype=np.int8)
        self.game_next_free_idx = 0
        self.start_t = start_t
        self.max_n_games = max_n_games
        self.n_workers = n_workers
        self.max_queue_size = max_queue_size
        self.minibatch_size = minibatch_size
        self.submit_interval = submit_interval

    def retrieve_new_games(self, has_games_event):
        while True:
            moves, dists, reward = self.game_queue.get()
            self.game_rewards[self.game_next_free_idx %
                self.max_n_games] = reward
            for i, (move, dist) in enumerate(zip(moves, dists)):
                idx = self.game_next_free_idx % self.max_n_games
                self.game_moves[idx] = move
                self.game_dists[idx] = dist
                self.game_starts[idx] = True if i == 0 else False
                self.game_next_free_idx += 1
            if self.game_next_free_idx >= self.start_t:
                has_games_event.set()

    def get_minibatches(self, has_games_event, minibatch_queue):
        has_games_event.wait()
        while True:
            top = (self.game_next_free_idx if self.game_next_free_idx <
                self.max_n_games else self.max_n_games)
            idxs = random.sample(range(top), self.minibatch_size)
            boards, dists, rewards = [], [], []
            for idx in idxs:
                board = dotboxes.Board()
                move_stack = []
                cur = idx
                dist = self.game_dists[cur % self.max_n_games]
                while not self.game_starts[cur % self.max_n_games]:
                    cur -= 1
                    move_stack.append(self.game_moves[cur % self.max_n_games])
                reward = self.game_rewards[cur % self.max_n_games]
                for move in reversed(move_stack):
                    board.push(move)
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
            if t % self.submit_interval == 0:
                self.trained_queue.put(self.policy.state_dict())
                print("submit {:.6f}".format(loss))
            t += 1

    def run(self):
        # check for new games thread
        has_games_event = threading.Event()
        new_games_thread = threading.Thread(target=self.retrieve_new_games,
            args=(has_games_event,), daemon=True)
        new_games_thread.start()

        # make minibatches thread
        minibatch_queue = mp.Queue(self.max_queue_size)
        get_minibatches_thread = threading.Thread(target=self.get_minibatches,
            args=(has_games_event, minibatch_queue), daemon=True)
        get_minibatches_thread.start()

        # start train workers
        self.policy = Policy()
        notify_queue = mp.Queue()
        workers = []
        for proc_idx in range(self.n_workers):
            worker = TrainWorker(minibatch_queue, notify_queue, self.policy)
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
