import numpy as np

n_rows = 4
n_cols = 4
class Board:
    def __init__(self):
        self.n_rows = n_rows
        self.n_cols = n_cols
        # state
        # first dim = 0 is vertical lines; first dim = 1 is horizontal lines
        self.edges = np.zeros((2, self.n_rows, self.n_cols), dtype=np.uint8)
        self.turn = True
        self.move_stack = []
        self.legal_moves = self.init_legal_moves()
        self.ownership = np.zeros((self.n_rows - 1, self.n_cols - 1),
            dtype=np.int8)

    def clone(self):
        board = Board()
        board.edges = np.copy(self.edges)
        board.turn = self.turn
        board.move_stack = list(self.move_stack)
        board.legal_moves = set(self.legal_moves)
        board.ownership = np.copy(self.ownership)
        return board

    def init_legal_moves(self):
        legals = set()
        for hv in range(2):
            for row in range(self.n_rows):
                for col in range(self.n_cols):
                    if not ((hv == 0 and row == self.n_rows - 1) or
                        (hv == 1 and col == self.n_cols - 1)):
                        legals.add((hv, row, col))
        return legals

    def is_game_over(self):
        return len(self.legal_moves) == 0

    def result(self):
        return np.sum(self.ownership)

    def push(self, move):
        self.edges[move] = 1
        hv, row, col = move
        own_a, own_b = None, None
        if hv == 0:  # vertical
            # left
            if col > 0 and self.edges[0][row][col-1] and \
                self.edges[1][row][col-1] and self.edges[1][row+1][col-1] and \
                self.ownership[row][col-1] == 0:
                    own_a = row, col - 1
            # right
            if col < self.n_cols - 1 and self.edges[0][row][col+1] and \
                self.edges[1][row][col] and self.edges[1][row+1][col] and \
                self.ownership[row][col] == 0:
                    own_b = row, col
        else:        # horizontal
            # up
            if row > 0 and self.edges[1][row-1][col] and \
                self.edges[0][row-1][col] and self.edges[0][row-1][col+1] and \
                self.ownership[row-1][col] == 0:
                    own_a = row - 1, col
            # down
            if row < self.n_rows - 1 and self.edges[1][row+1][col] and \
                self.edges[0][row][col] and self.edges[0][row][col+1] and \
                self.ownership[row][col] == 0:
                    own_b = row, col

        if own_a is None and own_b is None:
            self.turn = not self.turn
        if own_a is not None:
            self.ownership[own_a] = 1 if self.turn else -1
        if own_b is not None:
            self.ownership[own_b] = 1 if self.turn else -1
        self.move_stack.append((move, own_a, own_b))
        self.legal_moves.remove(move)

    def pop(self):
        move, own_a, own_b = self.move_stack.pop()
        if own_a is None and own_b is None:
            self.turn = not self.turn
        if own_a is not None:
            self.ownership[own_a] = 0
        if own_b is not None:
            self.ownership[own_b] = 0
        self.edges[move] = 0
        self.legal_moves.add(move)
        return move

    def __str__(self):
        ownership_chars = {-1: "B", 0: " ", 1: "A"}
        s = ""
        for row in range(self.n_rows):
            for col in range(self.n_cols):
                s += ".{}".format("-" if self.edges[1][row][col] else " ")
            s += "\n"
            for col in range(self.n_cols):
                o = ownership_chars[self.ownership[row][col]] if \
                    row < self.n_rows - 1 and col < self.n_cols - 1 else ""
                s += "{}{}".format("|" if self.edges[0][row][col] else " ", o)
            s += "\n"
        return s

if __name__ == "__main__":
    import random
    board = Board()
    while not board.is_game_over():
        move = random.choice(list(board.legal_moves))
        board.push(move)
        board1, own1 = np.array(board.edges), np.array(board.ownership)
        turn1 = board.turn
        legal1 = tuple(sorted(board.legal_moves))
        movestack1 = tuple(board.move_stack)

        board.pop()
        board.push(move)

        board2, own2 = np.array(board.edges), np.array(board.ownership)
        turn2 = board.turn
        legal2 = tuple(sorted(board.legal_moves))
        movestack2 = tuple(board.move_stack)
        assert(np.array_equal(board1, board2))
        assert(np.array_equal(own1, own2))
        assert(turn1 == turn2)
        assert(legal1 == legal2)
        assert(movestack1 == movestack2)
        print(board)
    print(board.result())
