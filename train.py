from common import *
import threading
import random

class TrainWorker(mp.Process):
    def __init__(self, minibatch_queue):
        super(TrainWorker, self).__init__()
        self.minibatch_queue = minibatch_queue

    def run(self):
        while True:
            minibatch = self.minibatch_queue.get()
            # TODO: actually train

class TrainManager(mp.Process):
    def __init__(self, game_queue, n_workers=1, max_n_games=100000,
        max_queue_size=8, minibatch_size=32):
        super(TrainManager, self).__init__()
        self.game_queue = game_queue
        self.games = []
        self.max_n_games = max_n_games
        self.n_workers = n_workers
        self.max_queue_size = max_queue_size
        self.minibatch_size = minibatch_size

    def retrieve_new_games(self, has_games_event):
        while True:
            game = self.game_queue.get()
            self.games.append(game)
            self.games = self.games[-self.max_n_games:]
            has_games_event.set()

    def get_minibatch(self):
        # TODO
        return [self.games[0][0]]

    def run(self):
        has_games_event = threading.Event()
        new_games_thread = threading.Thread(target=self.retrieve_new_games,
            args=(has_games_event,), daemon=True)
        new_games_thread.start()

        minibatch_queue = mp.Queue(self.max_queue_size)
        workers = []
        for proc_idx in range(self.n_workers):
            worker = TrainWorker(minibatch_queue)
            worker.start()
            workers.append(worker)

        has_games_event.wait()
        while True:
            minibatch = self.get_minibatch()
            minibatch_queue.put(minibatch)

        new_games_thread.join()
        for worker in workers:
            worker.join()
