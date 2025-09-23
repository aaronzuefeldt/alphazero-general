# tictactoe/CustomTicTacToePlayers.py

import numpy as np

class RandomPlayer:
    """A player that chooses a legal move at random."""
    def __init__(self, game):
        self.game = game

    def play(self, board):
        # Get a list of all valid moves
        valids = self.game.getValidMoves(board, 1)
        # Convert the binary vector to a list of action indices
        valid_moves = np.where(valids == 1)[0]
        # Choose one of the valid moves at random
        return np.random.choice(valid_moves)


class HumanPlayer:
    """
    Human-controlled player with intuitive text commands.

    Updated to match new action space and input style:
      - Place actions: 0..71 (72 total) = 8 rotations × 9 board cells
      - Spin: 72
      - Shoot: 73
      - End Turn: 74

    Input options (case-insensitive):
      • "shoot"
      • "spin"
      • "end" / "end turn" / "pass"
      • "place r c rot"  (e.g., "place 1 0 7")
      • "place r,c,rot"  (e.g., "place 1,0,7")
      • "place (r,c,rot)" (e.g., "place (1,0,7)")

    You can still enter the raw numeric action index as before.
    """
    def __init__(self, game):
        self.game = game
        self.action_size = game.getActionSize()

        # 8 rotations × n*n placement slots
        self.n = game.n
        self.place_space = 8 * (self.n * self.n)
        self.ACTION_SPIN = self.place_space           # 72 when n=3
        self.ACTION_SHOOT = self.place_space + 1      # 73
        self.ACTION_END_TURN = self.place_space + 2   # 74

        # For nicer prompts when showing rotation
        self._dir_arrows = ["→", "↘", "↓", "↙", "←", "↖", "↑", "↗"]

    # ----------------------------
    # Helpers for parsing commands
    # ----------------------------
    def _parse_place_triplet(self, s):
        """Return (r, c, rot) if s contains three ints; else None.
        Accepts formats like: "1 0 7", "1,0,7", "(1,0,7)".
        """
        if not isinstance(s, str):
            return None
        t = s.strip().lower()
        for ch in "(),;":
            t = t.replace(ch, " ")
        t = " ".join(t.split())  # collapse whitespace
        parts = [p for p in t.split(" ") if p]
        nums = []
        for p in parts:
            if p.lstrip("-+").isdigit():
                nums.append(int(p))
        if len(nums) >= 3:
            r, c, rot = nums[0], nums[1], nums[2]
            return r, c, rot
        return None

    def _parse_human_input(self, text, valids):
        """Translate a human-friendly command into an action index or None.
        Ensures returned action is currently valid per valids.
        """
        if not isinstance(text, str):
            return None
        s = text.strip().lower()
        if not s:
            return None

        # Direct keywords
        if s in {"shoot", "fire", "shoot!"}:
            a = self.ACTION_SHOOT
            return a if a < len(valids) and valids[a] else None
        if s in {"spin", "rotate", "rot"}:
            a = self.ACTION_SPIN
            return a if a < len(valids) and valids[a] else None
        if s in {"end", "end turn", "endturn", "pass", "done", "finish"}:
            a = self.ACTION_END_TURN
            return a if a < len(valids) and valids[a] else None

        # PLACE commands
        if s.startswith("place") or s.startswith("p ") or s == "p":
            rest = s[5:] if s.startswith("place") else s[1:]
            triplet = self._parse_place_triplet(rest)
            if triplet is None:
                return None
            r, c, rot = triplet
            # bounds
            if not (0 <= r < self.n and 0 <= c < self.n and 0 <= rot < 8):
                return None
            pos = r * self.n + c
            a = rot * (self.n * self.n) + pos
            return a if a < len(valids) and valids[a] else None

        # Numeric fallback
        try:
            a = int(s)
            return a if 0 <= a < len(valids) and valids[a] else None
        except ValueError:
            return None

    def play(self, board):
        # Get a list of all valid moves
        valids = self.game.getValidMoves(board, 1)

        print("--- Available Actions ---")
        # Display a compact summary of special actions
        if self.ACTION_SPIN < len(valids) and valids[self.ACTION_SPIN]:
            print(f"[spin]  Spin all pieces (idx {self.ACTION_SPIN})")
        if self.ACTION_SHOOT < len(valids) and valids[self.ACTION_SHOOT]:
            print(f"[shoot] Shoot (idx {self.ACTION_SHOOT})")
        if self.ACTION_END_TURN < len(valids) and valids[self.ACTION_END_TURN]:
            print(f"[end]   End Turn (idx {self.ACTION_END_TURN})")

        # Show a few example PLACE commands
        print("Examples: place 1 0 7   |   place 1,0,7   |   place (1,0,7)")

        # Input loop
        while True:
            user_in = input("Enter action (e.g., 'shoot', 'spin', 'end', or 'place r c rot'): ")
            a = self._parse_human_input(user_in, valids)
            if a is not None:
                return a
            print("Couldn't understand or action not currently legal. Try again.")
