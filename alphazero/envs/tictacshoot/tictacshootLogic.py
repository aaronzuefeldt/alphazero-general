# tictactoe/tictacshootLogic_sliding.py

import numpy as np

class Board():
    """
    Manages the custom Tic-Tac-Toe with shooting + sliding.
    Action space (for n=3): 72 placement actions (8 rotations × 9 squares)
    plus 3 special actions: 72=SPIN, 73=SHOOT, 74=END_TURN.
    """
    # Maps rotation index to a (row, col) direction vector
    DIRECTIONS = [
        (0, 1),   # 0: Right (→)
        (1, 1),   # 1: Down-Right (↘)
        (1, 0),   # 2: Down (↓)
        (1, -1),  # 3: Down-Left (↙)
        (0, -1),  # 4: Left (←)
        (-1, -1), # 5: Up-Left (↖)
        (-1, 0),  # 6: Up (↑)
        (-1, 1)   # 7: Up-Right (↗)
    ]

    def __init__(self, n=3):
        "Set up initial board configuration."
        self.n = n
        # 1 for Player O, -1 for Player X, 0 for empty
        self.pieces = np.zeros((self.n, self.n), dtype=int)

        # 1 = has shield, 0 = no shield
        self.has_shield_states = np.zeros((self.n, self.n), dtype=int)

        # Player -1 ('X') starts with the special token at (2,1) with NO shield
        token_index=7
        self.token_row, self.token_column = divmod(token_index, n)
        self.pieces[self.token_row, self.token_column] = -1


        # Rotation index (0-7) for each piece
        self.rotations = np.zeros((self.n, self.n), dtype=int)

        self.turn_number = 0
        self.actions_left = 2
        self.has_placed = False
        self.last_placed = None
        self.token_active = True


    # ---------------------- Rules / API ----------------------
    def get_legal_moves(self, player):
        """
        Returns a list of all legal moves.
        A move is an integer:
        - 0..(8*n*n - 1): Place a piece at square (r,c) with rotation p in [0..7].
                           Encoding: move = p*(n*n) + r*n + c
        - 8*n*n: SPIN
        - 8*n*n + 1: SHOOT
        - 8*n*n + 2: END_TURN
        """
        moves = []

        PLACEMENT_BASE = 0
        SPECIAL_BASE = 8 * self.n * self.n

        # 1. PLACE
        if not self.has_placed:
            for p in range(8):
                for r in range(self.n):
                    for c in range(self.n):
                        if self.pieces[r, c] == 0:
                            moves.append(PLACEMENT_BASE + p*(self.n*self.n) + (r * self.n + c))

        # 2. Actions that cost an action point
        if self.actions_left > 0:
            # SPIN possible if there's at least one piece (and token can't be the only piece that blocks this)
            if np.count_nonzero(self.pieces) > 1 or (np.count_nonzero(self.pieces) == 1 and not self.token_active):
                 moves.append(SPECIAL_BASE) # SPIN

            # SHOOT possible if there are valid targets
            if self._has_valid_targets(player):
                moves.append(SPECIAL_BASE + 1) # SHOOT

        # 3. END_TURN
        if self.has_placed or np.count_nonzero(self.pieces) == self.n * self.n:
            moves.append(SPECIAL_BASE + 2) # END TURN

        return moves

    def _has_valid_targets(self, player):
        """Checks if a shoot action would have any effect."""
        for r_start in range(self.n):
            for c_start in range(self.n):
                if self.pieces[r_start, c_start] == player and self.last_placed != (r_start, c_start):
                    # Special token at (2,1) cannot shoot while active
                    if self.token_active and r_start == self.token_row and c_start == self.token_column:
                        continue

                    rot_idx = self.rotations[r_start, c_start]
                    dr, dc = self.DIRECTIONS[rot_idx]
                    r, c = r_start + dr, c_start + dc

                    while self._in_bounds(r, c):
                        if self.pieces[r, c] != 0:
                            return True
                        r, c = r + dr, c + dc
        return False

    def check_win(self, win_len=3):
        """
        Check whether a player has won with `win_len` in a row on an m x n board.
        Returns 1 if Player O (1) wins, -1 if Player X (-1) wins, 0 for no win.
        """
        board = self.pieces
        m, n = board.shape
        k = int(win_len)

        if k <= 0:
            raise ValueError("win_len must be a positive integer.")
        if k > max(m, n):
            return 0  # impossible to have k in a row

        def has_k(player):
            # Horizontal
            for r in range(m):
                count = 0
                for c in range(n):
                    count = count + 1 if board[r, c] == player else 0
                    if count >= k:
                        return True

            # Vertical
            for c in range(n):
                count = 0
                for r in range(m):
                    count = count + 1 if board[r, c] == player else 0
                    if count >= k:
                        return True

            # Diagonal down-right (↘)
            # start along left edge
            for r0 in range(m):
                r, c = r0, 0
                count = 0
                while r < m and c < n:
                    count = count + 1 if board[r, c] == player else 0
                    if count >= k:
                        return True
                    r += 1; c += 1
            # start along top edge (excluding 0,0 to avoid double-count)
            for c0 in range(1, n):
                r, c = 0, c0
                count = 0
                while r < m and c < n:
                    count = count + 1 if board[r, c] == player else 0
                    if count >= k:
                        return True
                    r += 1; c += 1

            # Diagonal down-left (↙)
            # start along right edge
            for r0 in range(m):
                r, c = r0, n - 1
                count = 0
                while r < m and c >= 0:
                    count = count + 1 if board[r, c] == player else 0
                    if count >= k:
                        return True
                    r += 1; c -= 1
            # start along top edge (excluding top-right corner)
            for c0 in range(n - 2, -1, -1):
                r, c = 0, c0
                count = 0
                while r < m and c >= 0:
                    count = count + 1 if board[r, c] == player else 0
                    if count >= k:
                        return True
                    r += 1; c -= 1

            return False

        for player in (1, -1):
            if has_k(player):
                return player
        return 0


    def execute_move(self, move_idx, player):
        """Perform the given move on the board."""

        SPECIAL_BASE = 8 * self.n * self.n

        if 0 <= move_idx < SPECIAL_BASE: # PLACE
            p, mod = divmod(move_idx, self.n * self.n)
            r, c = divmod(mod, self.n)
            assert self.pieces[r, c] == 0 and not self.has_placed
            self.pieces[r, c] = player
            self.has_shield_states[r, c] = 1  # new piece gets a shield
            self.rotations[r, c] = p  # 0..7
            self.has_placed = True
            self.last_placed = (r, c)

        elif move_idx == SPECIAL_BASE: # SPIN
            assert self.actions_left > 0
            self.rotations = (self.rotations + 1) % 8
            self.actions_left -= 1

        elif move_idx == SPECIAL_BASE + 1: # SHOOT
            assert self.actions_left > 0
            self.shoot(player)
            self.actions_left -= 1


        else: # END_TURN
            self.turn_number += 1
            self.actions_left = 2
            self.has_placed = False
            self.last_placed = None
        
    
    
    def shoot(self,player):
        """Fixed version that matches C# logic more closely"""
        if self.actions_left <= 0:
            return
            
        # 1) Gather all hits (first target in each ray)
        hits = {}  # (r,c) -> dir_idx that hit it (first one encountered is enough)
        for r_start in range(self.n):
            for c_start in range(self.n):
                if self.pieces[r_start, c_start] == player and self.last_placed != (r_start, c_start):
                    # Special token cannot shoot while active
                    if self.token_active and r_start == self.token_row and c_start == self.token_column:
                        continue
                    dir_idx = int(self.rotations[r_start, c_start])
                    dr, dc = self.DIRECTIONS[dir_idx]
                    r, c = r_start + dr, c_start + dc
                    while self._in_bounds(r, c):
                        if self.pieces[r, c] != 0:
                            hits.setdefault((r, c), dir_idx)  # keep first shooter direction
                            break
                        r, c = r + dr, c + dc
    
        if not hits:
            return
    
        # 2) Partition into removals vs slides based on shields
        will_die = set()
        will_slide = {}  # (r,c) -> dir_idx
        for (r, c), dir_idx in hits.items():
            if self.has_shield_states[r, c] == 0:
                will_die.add((r, c))
            else:
                will_slide[(r, c)] = dir_idx
    
        # Token deactivation if it gets hit and dies
        if (self.token_row, self.token_column) in will_die:
            self.token_active = False
    
        # 3) Plan sliding destinations - FIXED LOGIC
        slide_targets = {}
        
        def plan_slide_from(rc, hit_dir_idx):
            r0, c0 = rc
            origin_idx = self._rc_to_idx(r0, c0)
            
            # Convert hit direction to Vector2-like representation for rotation
            dr, dc = self.DIRECTIONS[hit_dir_idx]
            slide_direction = [dc, -dr]  # Note: Y is flipped to match C# coordinate system
            
            # Try up to 3 rotations like C# (original, +90°, +180° cumulative)
            for rotation_attempt in range(3):
                if rotation_attempt == 1:
                    # Rotate +90° (clockwise in screen coordinates)
                    slide_direction = [slide_direction[1], -slide_direction[0]]
                elif rotation_attempt == 2:
                    # Rotate +180° from the +90° position (so +270° total from original)
                    slide_direction = [-slide_direction[0], -slide_direction[1]]
                
                j = 1
                while j < self.n:
                    rr = r0 + (-slide_direction[1]) * j  # Y component (flipped back)
                    cc = c0 + slide_direction[0] * j      # X component
                    
                    # Check bounds and obstacles (including dying pieces as obstacles)
                    if (not self._in_bounds(rr, cc) or 
                        (self.pieces[rr, cc] != 0 and (rr, cc) not in will_die)):
                        
                        if j > 1:
                            # Found valid slide destination
                            rr2 = r0 + (-slide_direction[1]) * (j - 1)
                            cc2 = c0 + slide_direction[0] * (j - 1)
                            rr3 = r0 + (-slide_direction[1]) * (j - 2)
                            cc3 = c0 + slide_direction[0] * (j - 2)
                            
                            dest_idx = self._rc_to_idx(int(rr2), int(cc2))
                            prev_idx = self._rc_to_idx(int(rr3), int(cc3))
                            
                            slide_targets.setdefault(dest_idx, []).append([origin_idx, prev_idx, j])
                            return dest_idx
                        else:
                            # Immediate block, try next rotation
                            break
                    else:
                        j += 1
            
            # No valid slide found: stay in place (shield consumed)
            slide_targets.setdefault(origin_idx, []).append([origin_idx, origin_idx, 0])
            return origin_idx
    
        for rc, dir_idx in will_slide.items():
            plan_slide_from(rc, dir_idx)
    
        # 4) Resolve overlapping slide destinations - FIXED BACKOFF CALCULATION
        max_iterations = self.n * self.n  # Prevent infinite loops
        iteration = 0
        while iteration < max_iterations:
            iteration += 1
            overlaps = 0
            
            for dest_idx, contenders in list(slide_targets.items()):
                if len(contenders) > 1:
                    overlaps += 1
                    # Sort by distance (ascending)
                    contenders.sort(key=lambda x: x[2])
                    
                    if len(contenders) >= 2 and contenders[0][2] == contenders[1][2]:
                        # Tie: push everyone back one cell
                        for origin_idx, prev_idx, dist in contenders:
                            if dist > 0:  # Only backoff if we moved
                                new_dest = prev_idx
                                # Calculate step more carefully to match C# logic
                                step = dest_idx - prev_idx
                                new_prev = prev_idx - step
                                slide_targets.setdefault(new_dest, []).append([origin_idx, new_prev, dist - 1])
                            else:
                                # Already at origin, just stay there
                                slide_targets.setdefault(origin_idx, []).append([origin_idx, origin_idx, 0])
                        slide_targets[dest_idx] = []
                    else:
                        # Winner stays; everyone else backs off one
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
                break  # conflict-free
    
        # 5) Apply removals
        for (r, c) in will_die:
            self.pieces[r, c] = 0
            self.rotations[r, c] = 0
            self.has_shield_states[r, c] = 0
    
        # 6) Apply slides
        for dest_idx, contenders in slide_targets.items():
            if len(contenders) == 1:
                origin_idx, prev_idx, dist = contenders[0]
                if dest_idx != origin_idx:
                    orr, occ = self._idx_to_rc(origin_idx)
                    drr, dcc = self._idx_to_rc(dest_idx)
                    # Move the piece
                    self.pieces[drr, dcc] = self.pieces[orr, occ]
                    self.rotations[drr, dcc] = self.rotations[orr, occ]
                    self.has_shield_states[drr, dcc] = 0  # shield consumed
                    # Clear origin
                    self.pieces[orr, occ] = 0
                    self.rotations[orr, occ] = 0
                    self.has_shield_states[orr, occ] = 0
                else:
                    # No move, but shield still consumed
                    r0, c0 = self._idx_to_rc(origin_idx)
                    self.has_shield_states[r0, c0] = 0
    
    def _in_bounds(self, r, c):
        return 0 <= r < self.n and 0 <= c < self.n
    
    def _rc_to_idx(self, r, c):
        return r * self.n + c
    
    def _idx_to_rc(self, idx):
        return divmod(idx, self.n)
