import dotboxes
from common import *
from mcts import *
from train import *
from evaluation import *

if __name__ == "__main__":
    mp.set_start_method("spawn")

    game_queue = mp.Queue()
    trained_queue = mp.Queue()
    alpha_queue = mp.Queue()

    self_play_manager = SelfPlayManager(alpha_queue, game_queue)
    self_play_manager.start()

    train_manager = TrainManager(game_queue, trained_queue)
    train_manager.start()

    model_eval_manager = ModelEvalManager(trained_queue, alpha_queue)
    model_eval_manager.start()

    self_play_manager.join()
    train_manager.join()
    model_eval_manager.join()

