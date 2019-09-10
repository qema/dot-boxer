from common import *
import dotboxes
import chess

class DotBoxesPolicy(nn.Module):
    def __init__(self, n_rows, n_cols):
        super(DotBoxesPolicy, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 2, 1)
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

    def bin_feat_plane(self, v):
        return (torch.ones(1, self.n_rows, self.n_cols, device=get_device())
            if v else torch.zeros(1, self.n_rows, self.n_cols,
                device=get_device()))

    def boards_to_tensor(self, boards):
        boards_t = []
        for board in boards:
            board_t = torch.from_numpy(board.edges).type(
                torch.float).to(get_device())
            board_t = torch.cat((board_t, self.bin_feat_plane(board.turn)))
            boards_t.append(board_t)
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
    def __init__(self, move_hist_len):
        super(ChessPolicy, self).__init__()
        self.conv1 = nn.Conv2d(12*move_hist_len + 5, 16, 3, padding=1)
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
    def __init__(self, move_hist_len=6):
        self.reward_dict = {"1-0": 1, "0-1": -1, "1/2-1/2": 0, "*": 0}
        self.move_hist_len = move_hist_len

    def Board(self):
        return chess.Board()

    def Policy(self):
        return ChessPolicy(self.move_hist_len)

    def bin_feat_plane(self, v):
        return (torch.ones(1, 8, 8, device=get_device()) if v else
            torch.zeros(1, 8, 8, device=get_device()))

    def board_to_tensor(self, state):
        board = state
        side = board.turn
        piece_map = board.piece_map()

        move_hist = []
        pieces_t = torch.zeros(12*self.move_hist_len, 8, 8,
            device=get_device())
        for hist_idx in range(self.move_hist_len):
            if len(board.move_stack) > 0:
                for pos, piece in piece_map.items():
                    col, row = chess.square_file(pos), chess.square_rank(pos)
                    idx = int(piece.color != side)*6 + (piece.piece_type-1)
                    pieces_t[12*hist_idx + idx][row][col] = 1
                move_hist.append(board.pop())
        for move in reversed(move_hist):
            board.push(move)

        color_t = self.bin_feat_plane(side)
        kw_t = self.bin_feat_plane(board.has_kingside_castling_rights(
            chess.WHITE))
        qw_t = self.bin_feat_plane(board.has_queenside_castling_rights(
            chess.WHITE))
        kb_t = self.bin_feat_plane(board.has_kingside_castling_rights(
            chess.BLACK))
        qb_t = self.bin_feat_plane(board.has_queenside_castling_rights(
            chess.BLACK))
        board_t = torch.cat((pieces_t, color_t, kw_t, qw_t, kb_t, qb_t))
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
