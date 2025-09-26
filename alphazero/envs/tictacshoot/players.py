# Updated players that match the "regular" TicTacToe API
# (BasePlayer with play(self, state: GameState) -> int and state.valid_moves())
#
# This adapts your custom interface to work where the standard game expects
# alphazero.GenericPlayers.BasePlayer subclasses that consume a GameState.
#
# It preserves the nicer human input parsing you wrote (place r c rot / spin / shoot / end),
# but falls back gracefully if those actions aren't part of the current game's action space.

from typing import Optional, Tuple
import math
import random

try:
    from alphazero.GenericPlayers import BasePlayer
    from alphazero.Game import GameState
except Exception:  # soft fallback for type checkers or alt envs
    class BasePlayer:  # type: ignore
        pass
    class GameState:  # type: ignore
        pass


class RandomTicTacToePlayer(BasePlayer):
    """A player that chooses any currently valid action at random.

    API: play(self, state: GameState) -> int
    """
    def __init__(self, *args, **kwargs):
        try:
            super().__init__(*args, **kwargs)
        except Exception:
            pass

    def play(self, state: GameState) -> int:
        valids = list(state.valid_moves())
        choices = [i for i, ok in enumerate(valids) if ok]
        if not choices:
            # If no valid actions (should be rare), return a pass-like action if present,
            # otherwise 0.
            return len(valids) - 1 if valids else 0
        return random.choice(choices)


class HumanTicTacToePlayer(BasePlayer):
    """Human-controlled player with intuitive text commands.

    Works with the standard API (GameState + valid_moves()). If your game uses
    an action space with special actions (e.g., spin / shoot / end turn) and
    8 rotational placements, those are detected heuristically from the action size.

    Supported inputs (case-insensitive):
      • "shoot"
      • "spin"
      • "end" / "end turn" / "pass"
      • "place r c rot"   (e.g., "place 1 0 7")
      • "place r,c,rot"   (e.g., "place 1,0,7")
      • "place (r,c,rot)" (e.g., "place (1,0,7)")
      • A raw numeric action index
    """

    def __init__(self, *args, **kwargs):
        try:
            super().__init__(*args, **kwargs)
        except Exception:
            pass
        # Will be derived per-state when play() is called
        self.n: Optional[int] = None
        self.place_space: Optional[int] = None
        self.ACTION_SPIN: Optional[int] = None
        self.ACTION_SHOOT: Optional[int] = None
        self.ACTION_END_TURN: Optional[int] = None
        self._dir_arrows = ["→", "↘", "↓", "↙", "←", "↖", "↑", "↗"]

    # ----------------------------
    # Helpers for parsing commands
    # ----------------------------
    def _parse_place_triplet(self, s: str) -> Optional[Tuple[int, int, int]]:
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

    def _parse_human_input(self, text: str, valids, action_size: int) -> Optional[int]:
        """Translate a human-friendly command into a valid action index or None."""
        if not isinstance(text, str):
            return None
        s = text.strip().lower()
        if not s:
            return None

        # Direct keywords (only if these indices exist and are valid)
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
            # Probe for an n where 8*n*n <= action_size
            # Prefer perfect squares close to action_size/8
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

        # Show a few example PLACE commands if the layout suggests rotations
        if self.place_space is not None and self.place_space <= action_size and self.n:
            print("Examples: place 1 0 7   |   place 1,0,7   |   place (1,0,7)")

        # Input loop
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
