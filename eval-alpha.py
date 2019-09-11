from mcts import *
from games import *
import dotboxes
import numpy as np
from scipy.stats import ttest_1samp

game = DotBoxesGame(2, 3)
good_policy = game.Policy()
good_policy.load_state_dict(torch.load("models/alpha.pt",
    map_location=get_device()))

rewards = []
for game_num in range(100):
    a_side = game_num % 2 == 0
    # TODO: hyperparams?
    agent_a = MCTSAgent(game, a_side, good_policy, 0, c_puct=0)
    agent_b = MCTSAgent(game, not a_side, game.Policy(), 0, c_puct=0)

    board = game.Board()
    t = 0
    moves, dists = [], []
    while not board.is_game_over():
        agent = agent_a if board.turn == a_side else agent_b
        agent.tau = 0
        agent.search(100)
        move = agent.choose_move()
        #if board.turn == a_side:
        #    move = agent.best_move()
        #else:
        #    import random
        #    move = random.choice(list(board.legal_moves))

        board.push(move)
        agent_a.commit_move(move)
        agent_b.commit_move(move)
        #print(a_side)
        #print(board)
        t += 1
    reward = game.reward_for_side(board, a_side)
    rewards.append(reward)
    print("Game {}: Reward {}".format(game_num, reward))

print("Average reward for alpha: {:.4f}".format(np.mean(rewards)))
print("Different from zero with p={:.4f}".format(ttest_1samp(rewards, 0)[1]))
