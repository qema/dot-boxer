from common import *
from selfplay import *
from train import *
from games import *

if __name__ == "__main__":
    mp.set_start_method("spawn")

    game = DotBoxesGame(5, 5)
    #game = ChessGame()

    policy = game.Policy()
    policy.share_memory()

    game_queue = mp.Queue()

    self_play_manager = SelfPlayManager(game, game_queue, policy)
    self_play_manager.start()

    train_manager = TrainManager(game, game_queue, policy)
    train_manager.start()

    self_play_manager.join()
    train_manager.join()
