
from typing import List, Tuple, Any

import numpy as np

from alphazero.Game import GameState

# Import the user's custom board logic (same directory as this file when used together)
from CustomTicTacToeLogic import Board as _CustomBoard


def _encode_board(b: _CustomBoard) -> np.ndarray:
    """
    Encodes the custom Board into planes for the NN/observation API.
    Planes (C=7): [pieces, rotations, shields, actions_left, last_placed, turn_number, token_active]
    """
    n = b.n
    board_state = np.zeros((7, n, n), dtype=np.float32)
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


def _decode_last_placed(plane: np.ndarray) -> Tuple[int, int] | None:
    ys, xs = np.where(plane == 1)
    return (int(ys[0]), int(xs[0])) if len(ys) else None


def _clone_board(b: _CustomBoard) -> _CustomBoard:
    nb = _CustomBoard(b.n)
    nb.pieces = np.array(b.pieces, dtype=int, copy=True)
    nb.rotations = np.array(b.rotations, dtype=int, copy=True)
    nb.has_shield_states = np.array(b.has_shield_states, dtype=int, copy=True)
    nb.actions_left = int(b.actions_left)
    nb.has_placed = bool(b.has_placed)
    nb.last_placed = None if b.last_placed is None else (int(b.last_placed[0]), int(b.last_placed[1]))
    nb.turn_number = int(b.turn_number)
    nb.token_active = bool(b.token_active)
    nb.token_row = int(b.token_row)
    nb.token_column = int(b.token_column)
    return nb


# ---------- Game constants mirroring the example API ----------
NUM_PLAYERS = 2

# This custom game exposes 7 planes as observation
NUM_CHANNELS = 7

# Default board size; can be overridden by passing n to Game()
BOARD_SIZE = 3

# Action space: 8 rotations * n*n placements + 3 specials (SPIN, SHOOT, END_TURN)
def _action_size(n: int) -> int:
    return n*n*8 + 3


def _special_base(n: int) -> int:
    return 8 * n * n


class Game(GameState):
    """
    A GameState-compatible wrapper around the user's custom Tic-Tac-Toe with
    spin/shoot/slide mechanics.

    Matches the API of alphazero.envs.tictactoe.tictactoe.Game
    (num_players, action_size, observation_size, valid_moves, play_action, win_state, observation, symmetries).
    """

    def __init__(self, _board: _CustomBoard | None = None, n: int = BOARD_SIZE):
        # Use provided board or create new with size n
        self._n = int(n if _board is None else _board.n)
        super().__init__(_board or _CustomBoard(self._n))

    # ----- Static API shape -----
    @staticmethod
    def num_players() -> int:
        return NUM_PLAYERS

    @staticmethod
    def action_size(n: int = BOARD_SIZE) -> int:
        return _action_size(n)

    @staticmethod
    def observation_size(n: int = BOARD_SIZE) -> Tuple[int, int, int]:
        return (NUM_CHANNELS, n, n)

    # ----- Helpers -----
    def _player_val(self) -> int:
        # Map internal player index (0/1) -> piece value (1 / -1)
        return (1, -1)[self.player]

    # ----- Core required methods -----
    def valid_moves(self) -> np.ndarray:
        """Return a fixed-size binary vector over the *full* action space for size n."""
        n = self._board.n
        valids = np.zeros(self.action_size(n), dtype=np.uint8)
        legal = self._board.get_legal_moves(self._player_val())
        for a in legal:
            if 0 <= a < len(valids):
                valids[a] = 1
        return valids

    def play_action(self, action: int) -> None:
        """Apply the action to the underlying board, then advance turn like the example API."""
        self._board.execute_move(int(action), self._player_val())
        self._update_turn()

    def win_state(self) -> np.ndarray:
        """
        Returns a boolean vector of length NUM_PLAYERS + 1:
          [p0_wins, p1_wins, draw]
        """
        result = [False] * (NUM_PLAYERS + 1)
        pv = self._player_val()
        w = int(self._board.check_win())  # 1, -1, or 0

        if w == pv:
            result[self.player] = True
        elif w == -pv:
            result[self._next_player(self.player)] = True
        else:
            # Hard draw condition for this variant to prevent endless games
            # (mirrors the safeguard in the original implementation).
            if self._board.turn_number > 500:
                result[-1] = True

        return np.array(result, dtype=np.uint8)

    def observation(self) -> np.ndarray:
        """Return CxHxW planes representing the full custom state."""
        return _encode_board(self._board)

    # ----- Symmetries (rotations + flips) -----
    def symmetries(self, pi: np.ndarray) -> List[Tuple[Any, np.ndarray]]:
        """
        Returns symmetry-equivalent states and remapped policy vectors.
        We support 4 rotations (0,90,180,270) and optional horizontal flips,
        similar to the example env, but with action-index remapping for:
          - placement actions with rotations (8*n*n)
          - 3 specials that stay in place (SPIN, SHOOT, END_TURN).
        """
        n = self._board.n
        ACTION_SIZE = _action_size(n)
        SPECIAL_BASE = _special_base(n)

        pi_arr = np.asarray(pi).reshape(-1)
        assert pi_arr.size == ACTION_SIZE, "pi must match full action size for this game."

        def rotate_coords_cw(r: int, c: int, k: int, n_: int) -> tuple[int, int]:
            for _ in range(k):
                r, c = c, n_ - 1 - r
            return r, c

        out: list[tuple[Game, np.ndarray]] = []

        # 4 rotations; for each, also include a horizontal flip variant for augmentation
        for k in range(4):  # 0,1,2,3 -> 0,90,180,270 CW
            # --- rotate an encoded view of the board to copy into a clone ---
            enc = _encode_board(self._board)
            if k > 0:
                enc_rot = np.stack([np.rot90(enc[p], -k) for p in range(enc.shape[0])], axis=0)
            else:
                enc_rot = enc.copy()

            # Fix rotations plane (index 1) where a piece exists
            piece_mask = enc_rot[0] != 0
            rot_plane = enc_rot[1].astype(np.int64)
            rot_plane[piece_mask] = (rot_plane[piece_mask] + 2 * k) % 8  # 90° -> +2 steps
            rot_plane[~piece_mask] = 0
            enc_rot[1] = rot_plane.astype(enc.dtype)

            # ---- Policy remap for this rotation ----
            pi_rot = np.zeros_like(pi_arr)

            # placement actions: index = p*(n*n) + (r*n + c)
            for p_ori in range(8):
                for r in range(n):
                    for c in range(n):
                        idx = p_ori * (n * n) + (r * n + c)
                        rr, cc = rotate_coords_cw(r, c, k, n)
                        p_new = (p_ori + 2 * k) % 8
                        idx_new = p_new * (n * n) + (rr * n + cc)
                        pi_rot[idx_new] = pi_arr[idx]

            # specials unchanged
            pi_rot[SPECIAL_BASE : SPECIAL_BASE + 3] = pi_arr[SPECIAL_BASE : SPECIAL_BASE + 3]

            # ---- Emit rotated state ----
            gs = self.clone()
            # decode enc_rot back into a Board
            b = _CustomBoard(n)
            b.pieces = np.array(enc_rot[0], dtype=int)
            b.rotations = np.array(enc_rot[1], dtype=int)
            b.has_shield_states = np.array(enc_rot[2], dtype=int)
            b.actions_left = int(enc_rot[3, 0, 0])
            last_placed = _decode_last_placed(enc_rot[4])
            b.last_placed = last_placed
            b.has_placed = last_placed is not None
            b.turn_number = int(enc_rot[5, 0, 0])
            b.token_active = bool(int(enc_rot[6, 0, 0]))
            gs._board = b

            out.append((gs, pi_rot))

        return out

    # ----- Object protocol -----
    def __eq__(self, other: 'Game') -> bool:
        return (
            np.array_equal(self._board.pieces, other._board.pieces)
            and np.array_equal(self._board.rotations, other._board.rotations)
            and np.array_equal(self._board.has_shield_states, other._board.has_shield_states)
            and self._board.n == other._board.n
            and self._player == other._player
            and self.turns == other.turns
        )

    def clone(self) -> 'Game':
        g = Game(n=self._board.n)
        g._board = _clone_board(self._board)
        g._player = self._player
        g._turns = self.turns
        return g
