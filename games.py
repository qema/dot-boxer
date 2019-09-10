from common import *
import dotboxes
import chess

class DotBoxesPolicy(nn.Module):
    def __init__(self, n_rows, n_cols):
        super(DotBoxesPolicy, self).__init__()
        self.conv1 = nn.Conv2d(2, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 2, 1)
        self.fc1 = nn.Linear(2*n_rows*n_cols, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, board):
        out = F.relu(self.conv1(board))
        out = F.relu(self.conv2(out))
        out = self.conv3(out).view(out.shape[0], -1)
        action = F.log_softmax(out, dim=1)
        value = torch.tanh(self.fc2(F.relu(self.fc1(out))))
        return action, value

class DotBoxesGame:
    def __init__(self, n_rows, n_cols):
        self.n_rows = n_rows
        self.n_cols = n_cols

    def Board(self):
        return dotboxes.Board(self.n_rows, self.n_cols)

    def Policy(self):
        return DotBoxesPolicy(self.n_rows, self.n_cols)

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

class ChessPolicy(nn.Module):
    def __init__(self):
        super(ChessPolicy, self).__init__()
        self.conv1 = nn.Conv2d(12, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 2, 3, padding=1)
        self.fc1_action = nn.Linear(2*8*8, 64*64)
        self.fc1_value = nn.Linear(2*8*8, 128)
        self.fc2_value = nn.Linear(128, 1)

    def forward(self, board):
        out = F.relu(self.conv1(board))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = out.view(out.shape[0], -1)
        action = F.log_softmax(self.fc1_action(out), dim=1)
        value = torch.tanh(self.fc2_value(F.relu(self.fc1_value(out))))
        return action, value

class ChessGame:
    def __init__(self):
        self.reward_dict = {"1-0": 1, "0-1": -1, "1/2-1/2": 0, "*": 0}

    def Board(self):
        return chess.Board()

    def Policy(self):
        return ChessPolicy()

    def board_to_tensor(self, state):
        board = state
        side = board.turn
        piece_map = board.piece_map()

        pieces_t = torch.zeros(12, 8, 8, device=get_device())
        for pos, piece in piece_map.items():
            col, row = chess.square_file(pos), chess.square_rank(pos)
            idx = int(piece.color != side)*6 + (piece.piece_type-1)
            pieces_t[idx][row][col] = 1

        board_t = pieces_t
        return board_t

    # input: list of fens
    def boards_to_tensor(self, states):
        boards_t = [self.board_to_tensor(state) for state in states]
        boards_t = torch.stack(boards_t)
        return boards_t

    def action_space_size(self):
        return 64*64

    def action_idx_to_move(self, idx):
        return chess.Move(idx // 64, idx % 64)

    def move_to_action_idx(self, move):
        return move.from_square*64 + move.to_square

    # precond: game is over
    def reward_for_side(self, board, side):
        result = board.result()
        reward = self.reward_dict[result]
        if not side:
            reward *= -1
        return reward
