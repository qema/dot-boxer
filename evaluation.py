from common import *

class ModelEvalWorker(mp.Process):
    def __init__(self):
        super(ModelEvalWorker, self).__init__()

    def run(self):
        pass

class ModelEvalManager(mp.Process):
    def __init__(self, game, trained_queue, alpha_queue):
        super(ModelEvalManager, self).__init__()
        self.trained_queue = trained_queue
        self.alpha_queue = alpha_queue
        self.game = game

    def run(self):
        # TODO
        while True:
            alpha = self.trained_queue.get()
            self.alpha_queue.put(alpha)
            torch.save(alpha, "models/alpha.pt")
