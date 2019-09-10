from common import *
#from mcts import *

#class ModelEvalWorker(mp.Process):
#    def __init__(self, game, cur_alpha, proposed_alpha, eval_queue,
#        results_queue, n_search_steps=100):
#        super(ModelEvalWorker, self).__init__()
#        self.game = game
#        self.cur_alpha = cur_alpha
#        self.proposed_alpha = proposed_alpha
#        self.eval_queue = eval_queue
#        self.results_queue = results_queue
#        self.n_search_steps = n_search_steps
#
#    def run(self):
#        while True:
#            side = self.eval_queue.get()
#            print("playing game", side)
#            agent_a = MCTSAgent(self.game, side, self.proposed_alpha, 0,
#                use_dirichlet_noise=False)
#            agent_b = MCTSAgent(self.game, not side, self.cur_alpha, 0,
#                use_dirichlet_noise=False)
#            board = self.game.Board()
#            while not board.is_game_over():
#                agent = agent_a if board.turn == side else agent_b
#                agent.search(self.n_search_steps)
#                move = agent.choose_move()
#
#                board.push(move)
#                agent_a.commit_move(move)
#                agent_b.commit_move(move)
#            reward = self.game.reward_for_side(board, side)
#            self.results_queue.put(reward)

class ModelEvalManager(mp.Process):
    def __init__(self, game, trained_queue, alpha_queue):#, n_evals=2):
        super(ModelEvalManager, self).__init__()
        self.trained_queue = trained_queue
        self.alpha_queue = alpha_queue
        self.game = game
        #self.cur_alpha = game.Policy()
        #self.cur_alpha.share_memory()
        #self.proposed_alpha = game.Policy()
        #self.proposed_alpha.share_memory()
        #self.n_evals = n_evals

    def run(self):
        while True:
            alpha = self.trained_queue.get()
            self.alpha_queue.put(alpha)
            torch.save(alpha, "models/alpha.pt")

#    def run(self):
#        eval_queue, results_queue = mp.Queue(), mp.Queue()
#
#        eval_worker = ModelEvalWorker(self.game, self.cur_alpha,
#            self.proposed_alpha, eval_queue, results_queue)
#        eval_worker.start()
#
#        while True:
#            proposed_alpha_state_dict = self.trained_queue.get()
#            self.proposed_alpha.load_state_dict(proposed_alpha_state_dict)
#
#            for i in range(self.n_evals):
#                eval_queue.put(i % 2 == 0)
#            rewards = []
#            for i in range(self.n_evals):
#                rewards.append(results_queue.get())
#
#            print(np.mean(rewards))
#            if np.mean(rewards) > 0: # TODO: criterion
#                self.cur_alpha.load_state_dict(proposed_alpha_state_dict)
#                self.alpha_queue.put(self.cur_alpha)
#                torch.save(self.cur_alpha, "models/alpha.pt")
#
#        eval_worker.join()
