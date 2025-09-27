"""
players.py (TicTacToe env)

Drop this file at:  ENVS_DIR/<your_tictactoe_env>/players.py
The GUI will import it via PLAYERS_MODULE automatically.

Implements two players that match the "regular" API:
  - class HumanTicTacToePlayer(BasePlayer)
  - class RandomTicTacToePlayer(BasePlayer)

Key notes:
  • No call to BasePlayer.__init__ (avoids spinning up MCTS).
  • Provides GUI-friendly attrs: name, wins, games, winrate.
  • Implements a no-op reset(self) to satisfy code paths that expect it.
  • play(self, state) returns a single valid integer action index from state.valid_moves().
"""

from typing import Optional, Tuple
import math
import random

from alphazero.GenericPlayers import BasePlayer  # keep the base type for compatibility
from alphazero.Game import GameState

__all__ = [
    "HumanTicTacToePlayer",
    "RandomTicTacToePlayer",
    # Prefer these distinct names in GUI dropdowns to avoid clashes with generic classes
    "TTTHumanPlayer",
    "TTTRandomPlayer",
]


class RandomTicTacToePlayer(BasePlayer):
    """Chooses a currently valid action uniformly at random."""

    def __init__(self, *args, **kwargs):
        # Do NOT call BasePlayer.__init__ (it tries to spin up MCTS).
        self.name = kwargs.get("name", self.__class__.__name__)
        self.wins = 0
        self.games = 0
        self.winrate = 0.0
        self._raw_init_args = args
        self._raw_init_kwargs = kwargs
        self.symbols = {
            0: "⬜", # Empty
            1: ["\u21E8", "\u2B02", "\u21E9", "\u2B03", "\u21E6", "\u2B01", "\u21E7", "\u2B00"], # Player O
           -1: ["\u2192", "\u2198", "\u2193", "\u2199", "\u2190", "\u2196", "\u2191", "\u2197"]  # Player X
        }

    def reset(self):  # optional hook some GUIs call
        return


    # ----------------------------
    # Display helper
    # ----------------------------
    def display(self, state: GameState):
        """Pretty-print the current board, turn info, and shields."""
        b = getattr(state, "_board", None)
        if b is None:
            return
        n = int(getattr(b, "n", 3))

        player_char = "O" if b.turn_number % 2 == 0 else "X"
        print("-" * (6 * n))
        print(f"Turn: {b.turn_number} | Player: {player_char} | Actions Left: {b.actions_left} | Placed: {b.has_placed}")

        for r in range(n):
            if r > 0:
                print("-" * (6 * n))
            print(" | ", end="")
            for c in range(n):
                piece = int(b.pieces[r, c])
                # Special display for the active C++ 'token'
                if getattr(b, "token_active", False) and r == int(b.token_row) and c == int(b.token_column):
                    symbol = " x "
                elif piece != 0:
                    rot = int(b.rotations[r, c]) % 8
                    try:
                        symbol = self.symbols[piece][rot]
                    except Exception:
                        symbol = " ? "
                    if int(b.has_shield_states[r, c]) == 1:
                        symbol = f"({symbol})"
                else:
                    symbol = self.symbols[0]
                print(f"{symbol:^3} | ", end="")
            print()
        print("-" * (6 * n))

    def play(self, state: GameState) -> int:
        self.display(state)
        valids = list(state.valid_moves())
        choices = [i for i, ok in enumerate(valids) if ok]
        if not choices:
            return 0
        return random.choice(choices)


class HumanTicTacToePlayer(BasePlayer):
    """Human-controlled player with intuitive text commands.

    Supported inputs (case-insensitive):
      • "shoot" (if present in the action space)
      • "spin"  (if present)
      • "end" / "end turn" / "pass" (if present)
      • "place r c rot"   (e.g., "place 1 0 7")
      • "place r,c,rot"   (e.g., "place 1,0,7")
      • "place (r,c,rot)" (e.g., "place (1,0,7)")
      • A raw numeric action index
    """

    def __init__(self, *args, **kwargs):
        # Do NOT call BasePlayer.__init__ (it tries to spin up MCTS).
        self.name = kwargs.get("name", self.__class__.__name__)
        self.wins = 0
        self.games = 0
        self.winrate = 0.0

        # Derived per-state in play()
        self.n: Optional[int] = None
        self.place_space: Optional[int] = None
        self.ACTION_SPIN: Optional[int] = None
        self.ACTION_SHOOT: Optional[int] = None
        self.ACTION_END_TURN: Optional[int] = None

    def reset(self):  # optional hook some GUIs call
        return

    # ----------------------------
    # Helpers for parsing commands
    # ----------------------------
    def _parse_place_triplet(self, s: str) -> Optional[Tuple[int, int, int]]:
        """Return (r, c, rot) if s contains three ints; else None."""
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

    def _parse_human_input(self, text: str, valids, action_size: int) -> Optional[int]:
        """Translate a human-friendly command into a valid action index or None."""
        if not isinstance(text, str):
            return None
        s = text.strip().lower()
        if not s:
            return None

        # Direct keywords (only if present and legal)
        for key, idx in (
            ("shoot", self.ACTION_SHOOT),
            ("spin", self.ACTION_SPIN),
            ("end", self.ACTION_END_TURN),
            ("end turn", self.ACTION_END_TURN),
            ("endturn", self.ACTION_END_TURN),
            ("pass", self.ACTION_END_TURN),
            ("done", self.ACTION_END_TURN),
            ("finish", self.ACTION_END_TURN),
        ):
            if key in s and idx is not None and 0 <= idx < action_size and valids[idx]:
                return idx

        # PLACE commands
        if s.startswith("place") or s.startswith("p ") or s == "p":
            rest = s[5:] if s.startswith("place") else s[1:]
            triplet = self._parse_place_triplet(rest)
            if triplet is None:
                return None
            if self.n is None:
                return None
            r, c, rot = triplet
            if not (0 <= r < self.n and 0 <= c < self.n):
                return None
            if rot is None:
                rot = 0
            if not (0 <= rot < 8):
                return None
            pos = r * self.n + c
            a = rot * (self.n * self.n) + pos
            if 0 <= a < action_size and valids[a]:
                return a
            return None

        # Numeric fallback
        try:
            a = int(s)
            return a if 0 <= a < action_size and valids[a] else None
        except ValueError:
            return None

    # ----------------------------
    # Inference of action layout
    # ----------------------------
    def _infer_layout(self, state: GameState, action_size: int) -> None:
        # Try to get board size n from the state if available
        n = None
        if hasattr(state, "_board") and hasattr(state._board, "n"):
            try:
                n = int(state._board.n)
            except Exception:
                n = None

        # If not available, guess n assuming 8 rotations × n*n placements + <= 3 specials
        if n is None:
            approx = max(1, int(round(math.sqrt(max(1, action_size / 8)))))
            candidates = {approx - 1, approx, approx + 1}
            candidates = [c for c in candidates if c > 0]
            for c in sorted(candidates, key=lambda x: abs(x - approx)):
                if 8 * c * c <= action_size:
                    n = c
                    break
            if n is None:
                # fallback for classic TicTacToe (no rotations): try a perfect square
                root = int(round(math.sqrt(action_size)))
                n = root if root * root <= action_size else max(1, root - 1)

        self.n = n

        # Define special action indices only if they fit inside the space
        self.place_space = 8 * (n * n)
        self.ACTION_SPIN = self.place_space if self.place_space < action_size else None
        self.ACTION_SHOOT = (
            self.place_space + 1 if (self.place_space + 1) < action_size else None
        )
        self.ACTION_END_TURN = (
            self.place_space + 2 if (self.place_space + 2) < action_size else None
        )

    # ----------------------------
    # Main API
    # ----------------------------
    def play(self, state: GameState) -> int:
        self.display(state)
        # Ask the environment for the current valid action mask
        valids = list(state.valid_moves())
        action_size = len(valids)

        # Infer layout and helpful indices
        self._infer_layout(state, action_size)

        # Compact summary of special actions that are currently legal
        print("--- Available Actions ---")
        if self.ACTION_SPIN is not None and self.ACTION_SPIN < action_size and valids[self.ACTION_SPIN]:
            print(f"[spin]  Spin all pieces (idx {self.ACTION_SPIN})")
        if self.ACTION_SHOOT is not None and self.ACTION_SHOOT < action_size and valids[self.ACTION_SHOOT]:
            print(f"[shoot] Shoot (idx {self.ACTION_SHOOT})")
        if self.ACTION_END_TURN is not None and self.ACTION_END_TURN < action_size and valids[self.ACTION_END_TURN]:
            print(f"[end]   End Turn (idx {self.ACTION_END_TURN})")
        if self.place_space is not None and self.place_space <= action_size and self.n:
            print("Examples: place 1 0 7   |   place 1,0,7   |   place (1,0,7)")

        # Input loop: keep asking until we get a currently-legal action index
        while True:
            try:
                user_in = input("Enter action (e.g., 'shoot', 'spin', 'end', or 'place r c rot' or a number): ")
            except EOFError:
                # Non-interactive fallback
                choices = [i for i, ok in enumerate(valids) if ok]
                return random.choice(choices) if choices else 0

            a = self._parse_human_input(user_in, valids, action_size)
            if a is not None:
                return a
            print("Couldn't understand or action not currently legal. Try again.")

# Distinct aliases to avoid colliding with generic RandomPlayer/HumanPlayer
class TTTHumanPlayer(HumanTicTacToePlayer):
    pass

class TTTRandomPlayer(RandomTicTacToePlayer):
    pass
