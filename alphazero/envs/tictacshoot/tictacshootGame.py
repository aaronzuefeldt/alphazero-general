# tictactoe/tictacshootGame.py

from __future__ import print_function
import sys
sys.path.append('..')
from Game import Game
from .tictacshootLogic import Board
import numpy as np

class tictacshootGame(Game):
    """
    Game class implementation for the custom C++ Tic-Tac-Toe variant.
    It wraps the Board logic into the API required by the AlphaZero framework.
    """
    def __init__(self, n=3):
        self.n = n
        # 8 rotation options per square + 3 special actions
        self.action_size = n*n*8 + 3
        self.ACTION_SPIN = n*n*8
        self.ACTION_SHOOT = n*n*8 + 1
        self.ACTION_END_TURN = n*n*8 + 2

        # Arrow symbols matching the C++ source (Game.h)
        # C++ piece '1' (O) -> Python player '1'
        # C++ piece '2' (X) -> Python player '-1'
        self.symbols = {
            0: "⬜", # Empty
            1: ["\u21E8", "\u2B02", "\u21E9", "\u2B03", "\u21E6", "\u2B01", "\u21E7", "\u2B00"], # Player O
           -1: ["\u2192", "\u2198", "\u2193", "\u2199", "\u2190", "\u2196", "\u2191", "\u2197"]  # Player X
        }

    def getInitBoard(self):
        b = Board(self.n)
        return self._encode_board(b)

    def getBoardSize(self):
        # Board is encoded into planes; the framework typically uses (n, n).
        return (self.n, self.n)

    def getActionSize(self):
        return self.action_size

    def getNextState(self, board, player, action):
        b = self._decode_board(board)
        b.execute_move(action, player)
        next_player = -player if action == self.ACTION_END_TURN else player
        # self.display(board)
        return self._encode_board(b), next_player

    def getValidMoves(self, board, player):
        valids = [0] * self.getActionSize()
        b = self._decode_board(board)
        legal_moves = b.get_legal_moves(player)

        for move_idx in legal_moves:
            if 0 <= move_idx < self.ACTION_SPIN:
                valids[move_idx] = 1
            elif move_idx == self.ACTION_SPIN:
                valids[self.ACTION_SPIN] = 1
            elif move_idx == self.ACTION_SHOOT:
                valids[self.ACTION_SHOOT] = 1
            elif move_idx == self.ACTION_END_TURN:
                valids[self.ACTION_END_TURN] = 1

        return np.array(valids)

    def getGameEnded(self, board, player):
        b = self._decode_board(board)
        win_status = b.check_win()

        if win_status != 0:
            return win_status * player # 1 if current player won, -1 if they lost

        if b.turn_number > 500: 
            return 1e-4

        return 0 # Game not over

    def getCanonicalForm(self, board, player):
        canonical_board = np.copy(board)
        # Flip the piece plane so the current player is always '1'
        canonical_board[0, :, :] *= player 
        return canonical_board

    def getSymmetries(self, board, pi):

        if board[6,0,0]==1:
            return [(board, pi)]

        n = self.n
        ACTION_SIZE = 8 * n * n + 3
        SPECIAL_BASE = 8 * n * n  # spin, shoot, end turn at +0,+1,+2

        def to_board_state(x):
            # Already an ndarray with expected shape?
            if isinstance(x, np.ndarray):
                if x.ndim == 3 and x.shape[0] >= 7:
                    return x.copy()
                # Sometimes a numpy scalar wrapping a Board object
                if x.dtype == object and x.shape == ():
                    x = x.item()
            # A Board instance: encode it
            if isinstance(x, Board):
                return self._encode_board(x)
            raise TypeError(f"Expected (7,n,n) ndarray or Board; got {type(x)}")

        def rotate_coords_cw(r, c, k, n_):
            for _ in range(k):
                r, c = c, n_ - 1 - r
            return r, c

        board_state = to_board_state(board).astype(np.float32, copy=False)
        pi_arr = np.asarray(pi).reshape(-1)
        has_full_pi = (pi_arr.size == ACTION_SIZE)

        syms = []
        for k in range(4):  # 0, 90, 180, 270 CW
            if k == 0:
                b_rot = board_state.copy()
            else:
                b_rot = np.stack([np.rot90(board_state[p], -k) for p in range(board_state.shape[0])], axis=0)

            # Fix rotations plane (idx 1) only where a piece exists (plane 0 != 0)
            piece_mask = b_rot[0] != 0
            rot_plane = b_rot[1].astype(np.int64)
            rot_plane[piece_mask] = (rot_plane[piece_mask] + 2 * k) % 8
            rot_plane[~piece_mask] = 0
            b_rot[1] = rot_plane.astype(board_state.dtype)

            # Policy remap (only if pi is the full vector)
            if has_full_pi:
                pi_rot = np.zeros_like(pi_arr)
                # placement actions: index = p*(n*n) + r*n + c
                for p_ori in range(8):
                    for r in range(n):
                        for c in range(n):
                            idx = p_ori * (n * n) + (r * n + c)
                            rr, cc = rotate_coords_cw(r, c, k, n)
                            p_new = (p_ori + 2 * k) % 8
                            idx_new = p_new * (n * n) + (rr * n + cc)
                            pi_rot[idx_new] = pi_arr[idx]
                # specials unchanged
                pi_rot[SPECIAL_BASE:SPECIAL_BASE + 3] = pi_arr[SPECIAL_BASE:SPECIAL_BASE + 3]
            else:
                # Keep whatever 'pi' you passed through unchanged (safe for quick tests)
                pi_rot = pi_arr.copy()

            syms.append((b_rot, pi_rot))

        return syms

    def stringRepresentation(self, board):
        return board.tobytes()

    def display(self, board):
        b = self._decode_board(board)
        n = b.n

        player_char = "O" if b.turn_number % 2 == 0 else "X"
        print("-" * (6 * n))
        print(f"Turn: {b.turn_number} | Player: {player_char} | Actions Left: {b.actions_left} | Placed: {b.has_placed}")

        for r in range(n):
            if r > 0: print("-" * (6 * n))
            print(" | ", end="")
            for c in range(n):
                piece = b.pieces[r, c]

                if b.token_active and r == 2 and c == 1:
                     # Special display for the active C++ 'token'
                    symbol = " x "
                elif piece != 0:
                    rot = b.rotations[r, c]
                    symbol = self.symbols[piece][rot]
                    if b.has_shield_states[r,c]==1:
                        symbol="("+symbol+")"
                else:
                    symbol = self.symbols[0]

                print(f"{symbol:^3} | ", end="")
            print()
        print("-" * (6 * n))

    # --- Helper methods for encoding/decoding the board state ---

    def _encode_board(self, b):
        """ Encodes the Board object into a NumPy array for the NN. """
        # 7 planes: pieces, rotations, shields, actions_left, last_placed, turn_number, token_active
        board_state = np.zeros((7, self.n, self.n), dtype=np.float32)
        board_state[0] = b.pieces
        board_state[1] = b.rotations
        board_state[2] = b.has_shield_states
        board_state[3].fill(b.actions_left)
        if b.last_placed is not None:
            r, c = b.last_placed
            board_state[4, r, c] = 1.0
        board_state[5].fill(b.turn_number)
        board_state[6].fill(1 if b.token_active else 0)
        return board_state

    def _decode_board(self, board_state):
        """ Decodes the NumPy array back into a Board object. """
        b = Board(self.n)
        b.pieces = np.array(board_state[0], dtype=int)
        b.rotations = np.array(board_state[1], dtype=int)
        b.has_shield_states = np.array(board_state[2], dtype=int)
        b.actions_left = int(board_state[3, 0, 0])

        ys, xs = np.where(board_state[4] == 1)
        b.last_placed = (int(ys[0]), int(xs[0])) if len(ys) else None
        b.has_placed = b.last_placed is not None

        b.turn_number = int(board_state[5, 0, 0])
        b.token_active = bool(board_state[6, 0, 0])
        return b
