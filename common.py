import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
import dotboxes
import numpy as np

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.conv1 = nn.Conv2d(2, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 2, 1)
        self.fc1 = nn.Linear(2*dotboxes.n_rows*dotboxes.n_cols, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, board):
        out = F.relu(self.conv1(board))
        out = F.relu(self.conv2(out))
        out = self.conv3(out).view(out.shape[0], -1)
        action = F.log_softmax(out, dim=1)
        value = torch.tanh(self.fc2(F.relu(self.fc1(out))))
        return action, value

device_cache = None
def get_device():
    global device_cache
    if not device_cache:
        device_cache = torch.device("cuda" if torch.cuda.is_available() else
            "cpu")
    return device_cache

def boards_to_tensor(boards):
    boards_t = []
    for board in boards:
        boards_t.append(torch.from_numpy(board.edges).type(
            torch.float).to(get_device()))
    return torch.stack(boards_t)

def action_idx_to_move(action_idx):
    action_idx = action_idx.item()
    board_size = dotboxes.n_rows * dotboxes.n_cols
    hv = action_idx // board_size
    row = (action_idx%board_size) // dotboxes.n_cols
    col = (action_idx%board_size) % dotboxes.n_cols
    return hv, row, col

def move_to_action_idx(move):
    hv, row, col = move
    board_size = dotboxes.n_rows * dotboxes.n_cols
    return hv*board_size + row*dotboxes.n_cols + col

def reward_for_side(board, side):
    return (board.result()) * (1 if side else -1)

if __name__ == "__main__":
    policy = Policy()
    board = dotboxes.Board()
    board_t = boards_to_tensor([board])
    action, value = policy(board_t)
    print(action_idx_to_move(action.argmax()))
    print(action, value)
