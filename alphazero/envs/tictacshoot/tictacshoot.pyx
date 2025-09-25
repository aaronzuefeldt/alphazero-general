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

# -----------------------------------------------------------------------------
# C-level Board implementation for performance
# -----------------------------------------------------------------------------


# Use a module-level constant (Python object) for directions
DIRECTIONS = (
    (0, 1),    # 0: Right (→)
    (1, 1),    # 1: Down-Right (↘)
    (1, 0),    # 2: Down (↓)
    (1, -1),   # 3: Down-Left (↙)
    (0, -1),   # 4: Left (←)
    (-1, -1),  # 5: Up-Left (↖)
    (-1, 0),   # 6: Up (↑)
    (-1, 1),   # 7: Up-Right (↗)
)

cdef class Board:
    """
    Custom Tic-Tac-Toe (with spin/shoot/slide) board logic.

    Action space (for n=3): 72 placement actions (8 rotations × 9 squares)
    plus 3 specials: 72=SPIN, 73=SHOOT, 74=END_TURN.
    """
    cdef public int n
    cdef object pieces           # np.ndarray[int32]
    cdef object rotations        # np.ndarray[int32]
    cdef object has_shield_states# np.ndarray[int32]
    cdef public int token_row, token_column
    cdef public bint token_active
    cdef public int turn_number, actions_left
    cdef public bint has_placed
    cdef object last_placed      # (r,c) or None

    def __cinit__(self, int n=3):
        self.n = n
        self.pieces = np.zeros((n, n), dtype=np.int32)          # 1, -1, or 0
        self.rotations = np.zeros((n, n), dtype=np.int32)       # 0..7 where piece exists
        self.has_shield_states = np.zeros((n, n), dtype=np.int32)  # 1 has shield, 0 none

        # Token starts at (2,1) as -1 and is active
        cdef int token_index = 7
        self.token_row, self.token_column = divmod(token_index, n)
        self.pieces[self.token_row, self.token_column] = -1
        self.token_active = True

        # Turn / action state
        self.turn_number = 0
        self.actions_left = 2
        self.has_placed = False
        self.last_placed = None

    # ---------------------- Rules / API ----------------------
    def get_legal_moves(self, int player):
        cdef int n = self.n
        cdef int SPECIAL_BASE = 8 * n * n
        moves = []

        if not self.has_placed:
            for p in range(8):
                base = p * (n * n)
                for r in range(n):
                    rn = r * n
                    for c in range(n):
                        if self.pieces[r, c] == 0:
                            moves.append(base + rn + c)

        if self.actions_left > 0:
            if np.count_nonzero(self.pieces) > 1 or (np.count_nonzero(self.pieces) == 1 and not self.token_active):
                moves.append(SPECIAL_BASE)      # SPIN
            if self._has_valid_targets(player):
                moves.append(SPECIAL_BASE + 1)  # SHOOT

        if self.has_placed or np.count_nonzero(self.pieces) == n * n:
            moves.append(SPECIAL_BASE + 2)      # END_TURN

        return moves

    def _has_valid_targets(self, int player):
        cdef int n = self.n
        cdef int dir_idx, r, c, r0, c0, dr, dc
        for r0 in range(n):
            for c0 in range(n):
                if self.pieces[r0, c0] == player and self.last_placed != (r0, c0):
                    if self.token_active and r0 == self.token_row and c0 == self.token_column:
                        continue
                    dir_idx = int(self.rotations[r0, c0])
                    dr, dc = DIRECTIONS[dir_idx]
                    r, c = r0 + dr, c0 + dc
                    while self._in_bounds(r, c):
                        if self.pieces[r, c] != 0:
                            return True
                        r += dr; c += dc
        return False

    def check_win(self, int win_len=3):
        board = self.pieces
        m, n = board.shape
        k = int(win_len)
        if k <= 0:
            raise ValueError("win_len must be positive")
        if k > max(m, n):
            return 0

        def has_k(player):
            # Horizontal
            for r in range(m):
                cnt = 0
                for c in range(n):
                    cnt = cnt + 1 if board[r, c] == player else 0
                    if cnt >= k: return True

            # Vertical
            for c in range(n):
                cnt = 0
                for r in range(m):
                    cnt = cnt + 1 if board[r, c] == player else 0
                    if cnt >= k: return True

            # Diagonal down-right (↘)
            for r0 in range(m):
                r, c = r0, 0
                cnt = 0
                while r < m and c < n:
                    cnt = cnt + 1 if board[r, c] == player else 0
                    if cnt >= k: return True
                    r += 1; c += 1
            for c0 in range(1, n):
                r, c = 0, c0
                cnt = 0
                while r < m and c < n:
                    cnt = cnt + 1 if board[r, c] == player else 0
                    if cnt >= k: return True
                    r += 1; c += 1

            # Diagonal down-left (↙)
            for r0 in range(m):
                r, c = r0, n - 1
                cnt = 0
                while r < m and c >= 0:
                    cnt = cnt + 1 if board[r, c] == player else 0
                    if cnt >= k: return True
                    r += 1; c -= 1
            for c0 in range(n - 2, -1, -1):
                r, c = 0, c0
                cnt = 0
                while r < m and c >= 0:
                    cnt = cnt + 1 if board[r, c] == player else 0
                    if cnt >= k: return True
                    r += 1; c -= 1

            return False

        for p in (1, -1):
            if has_k(p): return p
        return 0

    def execute_move(self, int move_idx, int player):
        cdef int n = self.n
        cdef int SPECIAL_BASE = 8 * n * n
        if 0 <= move_idx < SPECIAL_BASE:
            p, mod = divmod(move_idx, n * n)
            r, c = divmod(mod, n)
            if self.pieces[r, c] != 0 or self.has_placed:
                raise AssertionError("Illegal placement")
            self.pieces[r, c] = player
            self.has_shield_states[r, c] = 1
            self.rotations[r, c] = p
            self.has_placed = True
            self.last_placed = (r, c)
            return

        if move_idx == SPECIAL_BASE:
            if self.actions_left <= 0:
                raise AssertionError("No actions left for SPIN")
            self.rotations = (self.rotations + 1) % 8
            self.actions_left -= 1
            return

        if move_idx == SPECIAL_BASE + 1:
            if self.actions_left <= 0:
                raise AssertionError("No actions left for SHOOT")
            self.shoot(player)
            self.actions_left -= 1
            return

        # END_TURN
        self.turn_number += 1
        self.actions_left = 2
        self.has_placed = False
        self.last_placed = None

    def shoot(self, int player):
        if self.actions_left <= 0:
            return
        cdef int n = self.n
        hits = {}  # (r,c) -> dir_idx
        cdef int r0, c0, dir_idx, dr, dc, r, c

        for r0 in range(n):
            for c0 in range(n):
                if self.pieces[r0, c0] == player and self.last_placed != (r0, c0):
                    if self.token_active and r0 == self.token_row and c0 == self.token_column:
                        continue
                    dir_idx = int(self.rotations[r0, c0])
                    dr, dc = DIRECTIONS[dir_idx]
                    r, c = r0 + dr, c0 + dc
                    while self._in_bounds(r, c):
                        if self.pieces[r, c] != 0:
                            hits.setdefault((r, c), dir_idx)
                            break
                        r += dr; c += dc

        if not hits:
            return

        will_die = set()
        will_slide = {}
        for (r, c), dir_idx in hits.items():
            if self.has_shield_states[r, c] == 0:
                will_die.add((r, c))
            else:
                will_slide[(r, c)] = dir_idx

        if (self.token_row, self.token_column) in will_die:
            self.token_active = False

        slide_targets = {}

        def rc_to_idx(r, c): return r * n + c
        def idx_to_rc(idx): return divmod(idx, n)

        def plan_slide_from(rc, hit_dir_idx):
            r0, c0 = rc
            origin_idx = rc_to_idx(r0, c0)
            dr, dc = DIRECTIONS[hit_dir_idx]
            slide_vec = [dc, -dr]

            for attempt in range(3):
                if attempt == 1:
                    slide_vec[:] = [slide_vec[1], -slide_vec[0]]
                elif attempt == 2:
                    slide_vec[:] = [-slide_vec[0], -slide_vec[1]]

                j = 1
                while j < n:
                    rr = r0 + (-slide_vec[1]) * j
                    cc = c0 + slide_vec[0] * j
                    if (not self._in_bounds(rr, cc) or
                        (self.pieces[rr, cc] != 0 and (rr, cc) not in will_die)):
                        if j > 1:
                            rr2 = r0 + (-slide_vec[1]) * (j - 1)
                            cc2 = c0 + slide_vec[0] * (j - 1)
                            rr3 = r0 + (-slide_vec[1]) * (j - 2)
                            cc3 = c0 + slide_vec[0] * (j - 2)
                            dest_idx = rc_to_idx(int(rr2), int(cc2))
                            prev_idx = rc_to_idx(int(rr3), int(cc3))
                            slide_targets.setdefault(dest_idx, []).append([origin_idx, prev_idx, j])
                            return dest_idx
                        else:
                            break
                    else:
                        j += 1

            slide_targets.setdefault(origin_idx, []).append([origin_idx, origin_idx, 0])
            return origin_idx

        for rc, d in will_slide.items():
            plan_slide_from(rc, d)

        max_iters = n * n
        it = 0
        while it < max_iters:
            it += 1
            overlaps = 0
            for dest_idx, contenders in list(slide_targets.items()):
                if len(contenders) > 1:
                    overlaps += 1
                    contenders.sort(key=lambda x: x[2])
                    if len(contenders) >= 2 and contenders[0][2] == contenders[1][2]:
                        for origin_idx, prev_idx, dist in contenders:
                            if dist > 0:
                                new_dest = prev_idx
                                step = dest_idx - prev_idx
                                new_prev = prev_idx - step
                                slide_targets.setdefault(new_dest, []).append([origin_idx, new_prev, dist - 1])
                            else:
                                slide_targets.setdefault(origin_idx, []).append([origin_idx, origin_idx, 0])
                        slide_targets[dest_idx] = []
                    else:
                        winner = contenders[0]
                        losers = contenders[1:]
                        slide_targets[dest_idx] = [winner]
                        for origin_idx, prev_idx, dist in losers:
                            if dist > 0:
                                new_dest = prev_idx
                                step = dest_idx - prev_idx
                                new_prev = prev_idx - step
                                slide_targets.setdefault(new_dest, []).append([origin_idx, new_prev, dist - 1])
                            else:
                                slide_targets.setdefault(origin_idx, []).append([origin_idx, origin_idx, 0])
            if overlaps == 0:
                break

        for (r, c) in will_die:
            self.pieces[r, c] = 0
            self.rotations[r, c] = 0
            self.has_shield_states[r, c] = 0

        for dest_idx, contenders in slide_targets.items():
            if len(contenders) != 1:
                continue
            origin_idx, prev_idx, dist = contenders[0]
            if dest_idx == origin_idx:
                r0, c0 = idx_to_rc(origin_idx)
                self.has_shield_states[r0, c0] = 0
                continue

            orr, occ = idx_to_rc(origin_idx)
            drr, dcc = idx_to_rc(dest_idx)
            self.pieces[drr, dcc] = self.pieces[orr, occ]
            self.rotations[drr, dcc] = self.rotations[orr, occ]
            self.has_shield_states[drr, dcc] = 0
            self.pieces[orr, occ] = 0
            self.rotations[orr, occ] = 0
            self.has_shield_states[orr, occ] = 0

    # ---------------------- Pickling support ----------------------
    def __getstate__(self):
        return {
            "n": int(self.n),
            "pieces": self.pieces,
            "rotations": self.rotations,
            "has_shield_states": self.has_shield_states,
            "token_row": int(self.token_row),
            "token_column": int(self.token_column),
            "token_active": bool(self.token_active),
            "turn_number": int(self.turn_number),
            "actions_left": int(self.actions_left),
            "has_placed": bool(self.has_placed),
            "last_placed": self.last_placed,
        }

    def __setstate__(self, state):
        import numpy as _np
        self.n = int(state["n"])
        self.pieces = _np.array(state["pieces"], dtype=_np.int32, copy=True)
        self.rotations = _np.array(state["rotations"], dtype=_np.int32, copy=True)
        self.has_shield_states = _np.array(state["has_shield_states"], dtype=_np.int32, copy=True)
        self.token_row = int(state["token_row"])
        self.token_column = int(state["token_column"])
        self.token_active = bool(state["token_active"])
        self.turn_number = int(state["turn_number"])
        self.actions_left = int(state["actions_left"])
        self.has_placed = bool(state["has_placed"])
        self.last_placed = tuple(state["last_placed"]) if state["last_placed"] is not None else None

    def __reduce__(self):
        return (Board, (int(self.n),), self.__getstate__())

    # ---------------------- Helpers ----------------------
    cdef bint _in_bounds(self, int r, int c):
        return 0 <= r < self.n and 0 <= c < self.n

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