from common import *
import dotboxes

class DotBoxesGame:
    def __init__(self, n_rows, n_cols):
        self.n_rows = n_rows
        self.n_cols = n_cols

    def Board(self):
        return dotboxes.Board(self.n_rows, self.n_cols)

    def Policy(self):
        return dotboxes.Policy(self.n_rows, self.n_cols)

    def boards_to_tensor(self, boards):
        boards_t = []
        for board in boards:
            boards_t.append(torch.from_numpy(board.edges).type(
                torch.float).to(get_device()))
        return torch.stack(boards_t)

    def action_space_size(self):
        return 2 * self.n_rows * self.n_cols

    def action_idx_to_move(self, action_idx):
        action_idx = action_idx.item()
        board_size = self.n_rows * self.n_cols
        hv = action_idx // board_size
        row = (action_idx%board_size) // self.n_cols
        col = (action_idx%board_size) % self.n_cols
        return hv, row, col

    def move_to_action_idx(self, move):
        hv, row, col = move
        board_size = self.n_rows * self.n_cols
        return hv*board_size + row*self.n_cols + col

    def reward_for_side(self, board, side):
        return np.sign(board.result()) * (1 if side else -1)
