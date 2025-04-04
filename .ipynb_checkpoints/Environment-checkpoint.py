import pygame
import random
import numpy as np
from copy import deepcopy
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = ''


def is_move_vulnerable(move, state, env, player):
    """
    Simule le mouvement 'move' à partir de 'state' pour le joueur 'player'
    et renvoie True si, après ce coup, le pion se trouve dans une position
    où un adversaire peut le capturer.
    Pour un pion blanc, on regarde les cases diagonales vers le haut.
    """
    new_state, _, _ = env.apply_action_to_environment(state, move, player)
    (_, new_pos) = move
    r, c = new_pos
    for dr, dc in [(-1, -1), (-1, 1)]:
        enemy_r, enemy_c = r + dr, c + dc
        landing_r, landing_c = r + 2 * dr, c + 2 * dc
        if env._is_on_board(enemy_r, enemy_c) and env._is_on_board(landing_r, landing_c):
            if new_state[enemy_r][enemy_c] in (env.BLACK_PAWN, env.BLACK_KING) and new_state[landing_r][landing_c] == env.EMPTY:
                return True
    return False

# --- Bonus de centralisation pour les white kings ---
def centrality_bonus(state, env):
    """
    Pour chaque white king dans state, on calcule la distance euclidienne au centre du plateau.
    Plus cette distance est faible, plus le bonus est important.
    """
    bonus = 0.0
    center_row = (env.BOARD_SIZE - 1) / 2.0
    center_col = (env.BOARD_SIZE - 1) / 2.0
    # Distance maximale possible depuis le centre (coin)
    max_distance = ((center_row)**2 + (center_col)**2)**0.5
    for r in range(env.BOARD_SIZE):
        for c in range(env.BOARD_SIZE):
            if state[r][c] == env.WHITE_KING:
                d = ((r - center_row)**2 + (c - center_col)**2)**0.5
                # Bonus proportionnel à (max_distance - d)
                bonus += (max_distance - d) * 2.0  # coefficient ajustable
    return bonus

# --- ENVIRONNEMENT CHECKERS ---
class CheckersEnv:
    EMPTY = 0
    WHITE_PAWN, WHITE_KING = 1, 2
    BLACK_PAWN, BLACK_KING = 3, 4

    BOARD_SIZE = 8
    TILE_SIZE = 60
    WIDTH, HEIGHT = BOARD_SIZE * TILE_SIZE, BOARD_SIZE * TILE_SIZE

    COLORS = {
        "light": (255, 205, 160),
        "dark": (210, 140, 70),
        "white": (255, 255, 255),
        "black": (0, 0, 0),
        "highlight": (0, 255, 0)
    }

    def __init__(self, stalemate_threshold: int = 30):
        self.stalemate_threshold = stalemate_threshold
        self.non_capture_moves = 0
        self.done = False
        self.current_player = self.WHITE_PAWN  # le blanc commence
        self.board = None
        self.current_state = None
        self.screen = None
        # Propriété action_space pour éviter une éventuelle erreur (non utilisée ici)
        self.action_space = None
        self.reset()

    def reset(self) -> tuple[list, int]:
        # Initialiser un plateau vide
        self.board = [[self.EMPTY for _ in range(self.BOARD_SIZE)] for _ in range(self.BOARD_SIZE)]
        # Placement des pions noirs (en haut)
        for row in range(3):
            for col in range(self.BOARD_SIZE):
                if (row + col) % 2 == 1:
                    self.board[row][col] = self.BLACK_PAWN
        # Placement des pions blancs (en bas)
        for row in range(self.BOARD_SIZE - 3, self.BOARD_SIZE):
            for col in range(self.BOARD_SIZE):
                if (row + col) % 2 == 1:
                    self.board[row][col] = self.WHITE_PAWN

        self.current_state = deepcopy(self.board)
        self.current_player = self.WHITE_PAWN
        self.non_capture_moves = 0
        self.done = False
        return self.current_state, self.current_player

    def get_available_moves(self, state: list, player: int) -> list:
        """
        Retourne toutes les actions possibles pour le joueur donné.
        Chaque action est un tuple: ((row, col), (new_row, new_col))
        Si une capture est possible, seules les captures sont retournées.
        """
        moves = []
        capture_moves = []
        for row in range(self.BOARD_SIZE):
            for col in range(self.BOARD_SIZE):
                piece = state[row][col]
                if piece in (player, player + 1):  # pion ou roi
                    piece_moves = self._get_moves_for_piece(state, row, col)
                    for m in piece_moves:
                        if abs(m[1][0] - m[0][0]) == 2:
                            capture_moves.append(m)
                        else:
                            moves.append(m)
        return capture_moves if capture_moves else moves

    def _get_moves_for_piece(self, state: list, row: int, col: int) -> list:
        moves = []
        piece = state[row][col]
        if piece == self.WHITE_PAWN:
            directions = [(-1, -1), (-1, 1)]
        elif piece == self.BLACK_PAWN:
            directions = [(1, -1), (1, 1)]
        else:
            directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if self._is_on_board(new_row, new_col) and state[new_row][new_col] == self.EMPTY:
                moves.append(((row, col), (new_row, new_col)))
        for dr, dc in directions:
            mid_row, mid_col = row + dr, col + dc
            jump_row, jump_col = row + 2 * dr, col + 2 * dc
            if self._is_on_board(mid_row, mid_col) and self._is_on_board(jump_row, jump_col):
                opponent = self.BLACK_PAWN if piece in (self.WHITE_PAWN, self.WHITE_KING) else self.WHITE_PAWN
                if state[mid_row][mid_col] in (opponent, opponent + 1) and state[jump_row][jump_col] == self.EMPTY:
                    moves.append(((row, col), (jump_row, jump_col)))
        return moves

    def _is_on_board(self, row: int, col: int) -> bool:
        return 0 <= row < self.BOARD_SIZE and 0 <= col < self.BOARD_SIZE

    def is_game_finished(self, state: list) -> tuple[bool, int]:
        white_exists = any(piece in (self.WHITE_PAWN, self.WHITE_KING) for row in state for piece in row)
        black_exists = any(piece in (self.BLACK_PAWN, self.BLACK_KING) for row in state for piece in row)
        if not white_exists:
            return True, self.BLACK_PAWN
        if not black_exists:
            return True, self.WHITE_PAWN
        moves = self.get_available_moves(state, self.current_player)
        if not moves:
            return True, self._opponent(self.current_player)
        # Match nul (draw) si 30 coups consécutifs sans capture
        if self.non_capture_moves >= self.stalemate_threshold:
            return True, None
        return False, None

    def _opponent(self, player: int) -> int:
        return self.BLACK_PAWN if player == self.WHITE_PAWN else self.WHITE_PAWN

    def apply_action_to_environment(self, state: list, action: tuple, player: int) -> tuple[list, int, bool]:
        new_state = deepcopy(state)
        (row, col), (new_row, new_col) = action
        piece = new_state[row][col]
        new_state[new_row][new_col] = piece
        new_state[row][col] = self.EMPTY
        capture_occurred = False
        if abs(new_row - row) == 2:
            cap_row = (row + new_row) // 2
            cap_col = (col + new_col) // 2
            new_state[cap_row][cap_col] = self.EMPTY
            self.non_capture_moves = 0
            capture_occurred = True
        else:
            self.non_capture_moves += 1
        if piece == self.WHITE_PAWN and new_row == 0:
            new_state[new_row][new_col] = self.WHITE_KING
        if piece == self.BLACK_PAWN and new_row == self.BOARD_SIZE - 1:
            new_state[new_row][new_col] = self.BLACK_KING
        new_player = self._opponent(player)
        return new_state, new_player, capture_occurred

    def step(self, action: tuple) -> tuple[list, float, bool, int]:
        prev_state = deepcopy(self.current_state)
        new_state, new_player, capture_occurred = self.apply_action_to_environment(self.current_state, action, self.current_player)
        self.current_state = new_state
        self.current_player = new_player
        done, winner = self.is_game_finished(self.current_state)
        self.done = done

        reward = 1.0
        if done:
            if winner is None:
                reward = 0.0
            elif winner == self.WHITE_PAWN:
                reward = 250.0
            else:
                reward = -250.0
        enriched = compute_enriched_reward(prev_state, self.current_state, action, self, player=self.WHITE_PAWN)
        reward += enriched
        return self.current_state, reward, self.done, self.current_player

    def render(self, state: list = None):
        if state is None:
            state = self.current_state
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            pygame.display.set_caption("Checkers Environment")
        try:
            self.screen.fill((0, 0, 0))
            for row in range(self.BOARD_SIZE):
                for col in range(self.BOARD_SIZE):
                    color = self.COLORS["dark"] if (row + col) % 2 else self.COLORS["light"]
                    pygame.draw.rect(self.screen, color,
                                    (col * self.TILE_SIZE, row * self.TILE_SIZE,
                                    self.TILE_SIZE, self.TILE_SIZE))
            for row in range(self.BOARD_SIZE):
                for col in range(self.BOARD_SIZE):
                    piece = state[row][col]
                    if piece != self.EMPTY:
                        piece_color = self.COLORS["white"] if piece in (self.WHITE_PAWN, self.WHITE_KING) else self.COLORS["black"]
                        center = (col * self.TILE_SIZE + self.TILE_SIZE // 2, row * self.TILE_SIZE + self.TILE_SIZE // 2)
                        pygame.draw.circle(self.screen, piece_color, center, self.TILE_SIZE // 3)
                        if piece in (self.WHITE_KING, self.BLACK_KING):
                            pygame.draw.circle(self.screen, self.COLORS["highlight"], center, self.TILE_SIZE // 4, 3)
            pygame.display.flip()
        except:
            None

    def close(self):
        if self.screen:
            pygame.quit()

# --- FONCTIONS POUR LA RÉCOMPENSE ENRICHIE ---
def count_pieces(state, piece_types):
    return sum(tile in piece_types for row in state for tile in row)

def count_center_pieces(state, piece_types):
    count = 0
    for r in range(3, 5):
        for c in range(3, 5):
            if state[r][c] in piece_types:
                count += 1
    return count

def count_battalions(state, piece_types):
    count = 0
    board_size = len(state)
    directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    for r in range(board_size):
        for c in range(board_size):
            if state[r][c] in piece_types:
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < board_size and 0 <= nc < board_size and state[nr][nc] in piece_types:
                        count += 1
    return count / 2  # chaque paire est comptée deux fois

def compute_enriched_reward(prev_state, new_state, action, env, player=CheckersEnv.WHITE_PAWN):
    """
    Calcule une récompense enrichie en ajoutant :
      - Un bonus si le pion blanc avance (pousse ses pièces).
      - Une pénalité pour chaque pièce exposée au viseur adverse (sans protection derrière).
      - Les autres critères existants (captures, promotions, contrôle du centre, formations).
      - Un bonus de centralisation pour encourager le white king à rester au centre,
        afin de mieux exploiter ses capacités d'attaque sur les pièces adverses non-roi.
    """
    reward = 0.0
    prev_black = count_pieces(prev_state, [env.BLACK_PAWN, env.BLACK_KING])
    new_black = count_pieces(new_state, [env.BLACK_PAWN, env.BLACK_KING])
    if new_black < prev_black:
        reward += 10 * (prev_black - new_black)
    prev_white = count_pieces(prev_state, [env.WHITE_PAWN, env.WHITE_KING])
    new_white = count_pieces(new_state, [env.WHITE_PAWN, env.WHITE_KING])
    if new_white < prev_white:
        reward -= 10 * (prev_white - new_white)
    prev_white_kings = count_pieces(prev_state, [env.WHITE_KING])
    new_white_kings = count_pieces(new_state, [env.WHITE_KING])
    if new_white_kings > prev_white_kings:
        reward += 15 * (new_white_kings - prev_white_kings)
    prev_black_kings = count_pieces(prev_state, [env.BLACK_KING])
    new_black_kings = count_pieces(new_state, [env.BLACK_KING])
    if new_black_kings > prev_black_kings:
        reward -= 15 * (new_black_kings - prev_black_kings)
    center_white = count_center_pieces(new_state, [env.WHITE_PAWN, env.WHITE_KING])
    reward += 2 * center_white
    reward += 5 * (new_white - new_black)
    battalions = count_battalions(new_state, [env.WHITE_PAWN, env.WHITE_KING])
    reward += 3 * battalions

    orig_pos, new_pos = action
    orig_row, orig_col = orig_pos
    new_row, new_col = new_pos
    moved_piece = prev_state[orig_row][orig_col]
    if player == env.WHITE_PAWN and moved_piece == env.WHITE_PAWN and new_row < orig_row:
        reward += 2.0

    opponent = env.BLACK_PAWN if player == env.WHITE_PAWN else env.WHITE_PAWN
    enemy_moves = env.get_available_moves(new_state, opponent)
    exposure_penalty = 0.0
    for m in enemy_moves:
        if abs(m[1][0] - m[0][0]) == 2:
            captured_pos = ((m[0][0] + m[1][0]) // 2, (m[0][1] + m[1][1]) // 2)
            captured_piece = new_state[captured_pos[0]][captured_pos[1]]
            if captured_piece in ([env.WHITE_PAWN, env.WHITE_KING] if player == env.WHITE_PAWN else [env.BLACK_PAWN, env.BLACK_KING]):
                dr = m[1][0] - m[0][0]
                dc = m[1][1] - m[0][1]
                behind = (captured_pos[0] - dr, captured_pos[1] - dc)
                defended = False
                if 0 <= behind[0] < env.BOARD_SIZE and 0 <= behind[1] < env.BOARD_SIZE:
                    behind_piece = new_state[behind[0]][behind[1]]
                    if behind_piece in ([env.WHITE_PAWN, env.WHITE_KING] if player == env.WHITE_PAWN else [env.BLACK_PAWN, env.BLACK_KING]):
                        defended = True
                if not defended:
                    exposure_penalty += 20.0
                    if captured_piece == (env.WHITE_KING if player == env.WHITE_PAWN else env.BLACK_KING):
                        exposure_penalty += 10.0
    reward -= exposure_penalty

    bonus_centrality = centrality_bonus(new_state, env)
    reward += bonus_centrality

    return reward


