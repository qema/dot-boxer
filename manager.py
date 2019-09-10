from common import *
from selfplay import *
from train import *
from evaluation import *
from games import *

if __name__ == "__main__":
    mp.set_start_method("spawn")

    game = DotBoxesGame(4, 4)
    #game = ChessGame()

    game_queue = mp.Queue()
    trained_queue = mp.Queue()
    alpha_queue = mp.Queue()

    self_play_manager = SelfPlayManager(game, alpha_queue, game_queue)
    self_play_manager.start()

    train_manager = TrainManager(game, game_queue, trained_queue)
    train_manager.start()

    model_eval_manager = ModelEvalManager(game, trained_queue, alpha_queue)
    model_eval_manager.start()

    self_play_manager.join()
    train_manager.join()
    model_eval_manager.join()

