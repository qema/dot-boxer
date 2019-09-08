from mcts import *
from common import *
import dotboxes
import numpy as np
from scipy.stats import ttest_1samp

eval_queue_a, eval_queue_b = mp.Queue(), mp.Queue()
eval_pipe_recv_a, eval_pipe_send_a = mp.Pipe(duplex=False)
eval_pipe_recv_b, eval_pipe_send_b = mp.Pipe(duplex=False)

leaf_eval_worker_a = LeafEvalWorker(eval_queue_a, [eval_pipe_send_a], None)
leaf_eval_worker_a.start()
leaf_eval_worker_a.policy.load_state_dict(torch.load("models/alpha.pt",
    map_location=get_device()))

leaf_eval_worker_b = LeafEvalWorker(eval_queue_b, [eval_pipe_send_b], None)
leaf_eval_worker_b.start()

rewards = []
for game_num in range(100):
    a_side = game_num % 2 == 0
    # TODO: hyperparams?
    agent_a = MCTSAgent(a_side, eval_queue_a, eval_pipe_recv_a, 0, 0.0, 0, 3)
    agent_b = MCTSAgent(not a_side, eval_queue_b, eval_pipe_recv_b, 0, 0.0, 0,
        3)

    board = dotboxes.Board()
    t = 0
    moves, dists = [], []
    while not board.is_game_over():
        agent = agent_a if board.turn == a_side else agent_b
        agent.tau = 1
        for i in range(30):
            agent.search_step()
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
    reward = reward_for_side(board, a_side)
    rewards.append(reward)
    print("Game {}: Reward {}".format(game_num, reward))

print("Average reward for alpha: {:.4f}".format(np.mean(rewards)))
print("Different from zero with p={:.4f}".format(ttest_1samp(rewards, 0)[1]))

leaf_eval_worker_a.join()
leaf_eval_worker_b.join()
