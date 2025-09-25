# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: overflowcheck=False
# cython: initializedcheck=False
# cython: cdivision=True
# cython: auto_pickle=True

import numpy as np
cimport numpy as np
import pyximport

pyximport.install(setup_args={'include_dirs': np.get_include()})

from alphazero.Game import GameState
from typing import List, Tuple, Any

# Type definitions for numpy arrays
ctypedef np.int_t DTYPE_INT_t
ctypedef np.float32_t DTYPE_FLOAT_t
ctypedef np.uint8_t DTYPE_UINT8_t

# Game Constants
cdef int NUM_PLAYERS = 2
cdef int NUM_CHANNELS = 7
cdef int BOARD_SIZE = 3

cdef tuple DIRECTIONS = (
    (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)
)

# -----------------------------------------------------------------------------
# C-level Board implementation for performance
# -----------------------------------------------------------------------------
cdef class Board:
    """
    Cython implementation of the TicTacShoot board logic.
    """
    cdef public int n
    cdef public np.ndarray pieces, has_shield_states, rotations
    cdef public int turn_number, actions_left
    cdef public bint has_placed, token_active
    cdef public object last_placed
    cdef public int token_row, token_column

    def __cinit__(self, int n=3):
        self.n = n
        self.pieces = np.zeros((n, n), dtype=np.intp)
        self.has_shield_states = np.zeros((n, n), dtype=np.intp)
        self.rotations = np.zeros((n, n), dtype=np.intp)
        cdef int token_index = 7
        self.token_row, self.token_column = divmod(token_index, n)
        self.pieces[self.token_row, self.token_column] = -1
        self.turn_number = 0
        self.actions_left = 2
        self.has_placed = False
        self.last_placed = None
        self.token_active = True

    def __reduce__(self):
        return (Board, (self.n,), self.__getstate__())

    def __getstate__(self):
        return (self.n, self.pieces, self.has_shield_states, self.rotations,
                self.turn_number, self.actions_left, self.has_placed,
                self.last_placed, self.token_active, self.token_row, self.token_column)

    def __setstate__(self, state):
        (self.n, self.pieces, self.has_shield_states, self.rotations,
         self.turn_number, self.actions_left, self.has_placed,
         self.last_placed, self.token_active, self.token_row, self.token_column) = state

    cpdef Board clone(self):
        cdef Board new_board = Board(self.n)
        new_board.pieces = np.copy(self.pieces)
        new_board.has_shield_states = np.copy(self.has_shield_states)
        new_board.rotations = np.copy(self.rotations)
        new_board.turn_number = self.turn_number
        new_board.actions_left = self.actions_left
        new_board.has_placed = self.has_placed
        new_board.last_placed = self.last_placed
        new_board.token_active = self.token_active
        new_board.token_row = self.token_row
        new_board.token_column = self.token_column
        return new_board

    # ... other Board methods like get_legal_moves, check_win, execute_move ...
    cpdef list get_legal_moves(self, int player):
        cdef list moves = []
        cdef int p, r, c
        cdef int SPECIAL_BASE = 8 * self.n * self.n
        if not self.has_placed:
            for p in range(8):
                for r in range(self.n):
                    for c in range(self.n):
                        if self.pieces[r, c] == 0:
                            moves.append(p * (self.n * self.n) + (r * self.n + c))
        if self.actions_left > 0:
            if np.count_nonzero(self.pieces) > 1 or (np.count_nonzero(self.pieces) == 1 and not self.token_active):
                moves.append(SPECIAL_BASE)
            if self._has_valid_targets(player):
                moves.append(SPECIAL_BASE + 1)
        if self.has_placed or np.count_nonzero(self.pieces) == self.n * self.n:
            moves.append(SPECIAL_BASE + 2)
        return moves
    cdef bint _has_valid_targets(self, int player): return False # dummy
    cpdef int check_win(self, int win_len=3): return 0 # dummy
    cpdef void execute_move(self, int move_idx, int player): pass # dummy
    cdef void shoot(self, int player): pass # dummy
    cdef bint _in_bounds(self, int r, int c): return 0 <= r < self.n and 0 <= c < self.n


# -----------------------------------------------------------------------------
# Python-level Game class for API compatibility
# -----------------------------------------------------------------------------
class Game(GameState):
    """
    A regular Python class that acts as a GameState-compatible wrapper.
    It holds an instance of the fast Cython `Board` class.
    """
    def __init__(self, _board: Board | None = None, n: int = BOARD_SIZE):
        self._n = int(n if _board is None else _board.n)
        super().__init__(_board or Board(self._n))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Game): return NotImplemented
        if self.player != other.player: return False
        b1, b2 = self._board, other._board
        return (b1.turn_number == b2.turn_number and
                b1.actions_left == b2.actions_left and
                b1.has_placed == b2.has_placed and
                np.array_equal(b1.pieces, b2.pieces) and
                np.array_equal(b1.rotations, b2.rotations))

    @staticmethod
    def num_players() -> int: return NUM_PLAYERS
    @staticmethod
    def action_size(n: int = BOARD_SIZE) -> int: return n * n * 8 + 3
    @staticmethod
    def observation_size(n: int = BOARD_SIZE) -> Tuple[int, int, int]: return (NUM_CHANNELS, n, n)

    def _player_val(self) -> int: return 1 if self.player == 0 else -1
    def valid_moves(self) -> np.ndarray:
        valids = np.zeros(self.action_size(self._n), dtype=np.uint8)
        for move in self._board.get_legal_moves(self._player_val()): valids[move] = 1
        return valids
    def play_action(self, action: int) -> None:
        self._board.execute_move(action, self._player_val())
        self._update_turn()
    def win_state(self) -> np.ndarray:
        result = np.zeros(NUM_PLAYERS + 1, dtype=np.uint8)
        winner = self._board.check_win()
        if winner != 0:
            if winner == self._player_val(): result[self.player] = 1
            else: result[self._next_player(self.player)] = 1
        elif self._board.turn_number > 500: result[NUM_PLAYERS] = 1
        return result
    def observation(self) -> np.ndarray: return _encode_board(self._board)
    def clone(self) -> 'Game':
        cloned_game = Game(n=self._n)
        cloned_game._board = self._board.clone()
        cloned_game._player = self.player
        cloned_game._turns = self.turns
        return cloned_game

    # FIX: Implement the required symmetries method
    def symmetries(self, pi: np.ndarray) -> List[Tuple['Game', np.ndarray]]:
        """
        Generates symmetric game states and policies via board rotations.
        """
        cdef int n = self._n
        cdef int ACTION_SIZE = self.action_size(n)
        cdef int SPECIAL_BASE = 8 * n * n
        cdef int k, p_ori, r, c, rr, cc, p_new, idx, idx_new

        # Get current state and policy as numpy arrays
        board_state = self.observation()
        pi_arr = np.asarray(pi)

        syms = []
        for k in range(4):  # 0, 90, 180, 270 degree clockwise rotations
            # 1. Rotate the board state
            if k == 0:
                b_rot_state = np.copy(board_state)
            else:
                # np.rot90(m, k) rotates k times counter-clockwise. We want clockwise.
                # So we use -k.
                b_rot_state = np.stack([np.rot90(plane, -k) for plane in board_state], axis=0)

            # 2. Correct the piece rotations in the new rotated state
            # A 90-degree board rotation means each piece's arrow also rotates 90 degrees.
            # 8 directions / 4 rotations = 2 steps per 90-degree turn.
            piece_mask = b_rot_state[0] != 0
            rot_plane = b_rot_state[1].astype(np.int64)
            rot_plane[piece_mask] = (rot_plane[piece_mask] + 2 * k) % 8
            b_rot_state[1] = rot_plane.astype(board_state.dtype)

            # 3. Remap the policy vector to match the new board state
            pi_rot = np.zeros_like(pi_arr)
            # a. Remap placement actions
            for p_ori in range(8):
                for r in range(n):
                    for c in range(n):
                        # Original action index
                        idx = p_ori * (n*n) + r*n + c
                        
                        # New coordinates after board rotation
                        rr, cc = r, c
                        if k == 1: rr, cc = c, n - 1 - r      # 90 deg
                        elif k == 2: rr, cc = n - 1 - r, n - 1 - c # 180 deg
                        elif k == 3: rr, cc = n - 1 - c, r      # 270 deg

                        # New piece orientation after board rotation
                        p_new = (p_ori + 2*k) % 8
                        
                        # New action index
                        idx_new = p_new * (n*n) + rr*n + cc
                        pi_rot[idx_new] = pi_arr[idx]
            
            # b. Special actions are not position-dependent
            pi_rot[SPECIAL_BASE:ACTION_SIZE] = pi_arr[SPECIAL_BASE:ACTION_SIZE]

            # 4. Create the new Game object from the rotated state
            new_game = Game(n=n)
            new_game._board = _decode_board(b_rot_state, n)
            new_game._player = self.player # Turn does not change with symmetry
            new_game._turns = self.turns
            
            syms.append((new_game, pi_rot))
            
        return syms

# -----------------------------------------------------------------------------
# Cython Helper Functions
# -----------------------------------------------------------------------------
cpdef np.ndarray _encode_board(Board b):
    cdef int n = b.n
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=3] state = np.zeros((NUM_CHANNELS, n, n), dtype=np.float32)
    state[0] = b.pieces
    state[1] = b.rotations
    state[2] = b.has_shield_states
    state[3].fill(b.actions_left)
    if b.last_placed is not None: r, c = b.last_placed; state[4, r, c] = 1.0
    state[5].fill(b.turn_number)
    state[6].fill(1 if b.token_active else 0)
    return state

# Helper to create a Board object from a numpy state array
cdef Board _decode_board(np.ndarray board_state, int n):
    """Decodes the NumPy array back into a Board object."""
    cdef Board b = Board(n)
    b.pieces = np.array(board_state[0], dtype=np.intp)
    b.rotations = np.array(board_state[1], dtype=np.intp)
    b.has_shield_states = np.array(board_state[2], dtype=np.intp)
    b.actions_left = int(board_state[3, 0, 0])
    
    ys, xs = np.where(board_state[4] == 1)
    if len(ys) > 0:
        b.last_placed = (int(ys[0]), int(xs[0]))
        b.has_placed = True
    else:
        b.last_placed = None
        b.has_placed = False

    b.turn_number = int(board_state[5, 0, 0])
    b.token_active = bool(board_state[6, 0, 0])
    return b