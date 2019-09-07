import dotboxes
from common import *
from mcts import *
from train import *

if __name__ == "__main__":
    mp.set_start_method("spawn")

    eval_queue = mp.Queue()
    game_queue = mp.Queue()
    eval_pipe_recv_a, eval_pipe_send_a = mp.Pipe(duplex=False)
    eval_pipe_recv_b, eval_pipe_send_b = mp.Pipe(duplex=False)
    search_worker = SelfPlayWorker(eval_queue, game_queue, eval_pipe_recv_a, 0,
        eval_pipe_recv_b, 1)
    eval_worker = LeafEvalWorker(1, eval_queue,
        [eval_pipe_send_a, eval_pipe_send_b])
    search_worker.start()
    eval_worker.start()

    train_manager = TrainManager(game_queue)
    train_manager.start()

    search_worker.join()
    eval_worker.join()
    train_manager.join()

