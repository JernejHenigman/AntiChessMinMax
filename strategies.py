"""
Some example strategies for people who want to create a custom, homemade bot.
And some handy classes to extend
"""

import chess
from chess.engine import PlayResult
import random
from engine_wrapper import EngineWrapper
import AlphaZeroAgent
import numpy as np
import chess.variant
import time
import random
import chess.engine
import itertools
import random
import chess.polyglot
import pandas as pd
from collections import defaultdict


piece_values = {1 : 1, 2 : 2, 3 : 3, 4 : 6, 5 : 4, 6 : 5}

class FillerEngine:
    """
    Not meant to be an actual engine.

    This is only used to provide the property "self.engine"
    in "MinimalEngine" which extends "EngineWrapper"
    """
    def __init__(self, main_engine, name=None):
        self.id = {
            "name": name
        }
        self.name = name
        self.main_engine = main_engine

    def __getattr__(self, method_name):
        main_engine = self.main_engine

        def method(*args, **kwargs):
            nonlocal main_engine
            nonlocal method_name
            return main_engine.notify(method_name, *args, **kwargs)

        return method


class MinimalEngine(EngineWrapper):
    """
    Subclass this to prevent a few random errors

    Even though MinimalEngine extends EngineWrapper,
    you don't have to actually wrap an engine.

    At minimum, just implement `search`,
    however you can also change other methods like
    `notify`, `first_search`, `get_time_control`, etc.
    """
    def __init__(self, *args, name=None):
        super().__init__(*args)

        self.engine_name = self.__class__.__name__ if name is None else name

        self.last_move_info = []
        self.engine = FillerEngine(self, name=self.name)
        self.engine.id = {
            "name": self.engine_name
        }

    def search_with_ponder(self, board, wtime, btime, winc, binc, ponder, draw_offered):
        timeleft = 0
        if board.turn:
            timeleft = wtime
        else:
            timeleft = btime
        return self.search(board, timeleft, ponder, draw_offered)

    def search(self, board, timeleft, ponder, draw_offered):
        """
        The method to be implemented in your homemade engine

        NOTE: This method must return an instance of "chess.engine.PlayResult"
        """
        raise NotImplementedError("The search method is not implemented")

    def notify(self, method_name, *args, **kwargs):
        """
        The EngineWrapper class sometimes calls methods on "self.engine".
        "self.engine" is a filler property that notifies <self> 
        whenever an attribute is called.

        Nothing happens unless the main engine does something.

        Simply put, the following code is equivalent
        self.engine.<method_name>(<*args>, <**kwargs>)
        self.notify(<method_name>, <*args>, <**kwargs>)
        """
        pass


class ExampleEngine(MinimalEngine):
    pass


# Strategy names and ideas from tom7's excellent eloWorld video

class RandomMove(ExampleEngine):
    def search(self, board, *args):
        return PlayResult(random.choice(list(board.legal_moves)), None)


class Node():
    def __init__(self, parent=None, board=None, move=None, h=None):
        self.parent = parent
        self.board = board
        self.move = move
        self.children = []
        self.children_sorted = []
        self.heuristic = h

    def eval_material(self):
        piece_list = [(piece.color, piece.piece_type) for piece in list(self.board.piece_map().values())]
        black_piece_values = []
        white_piece_values = []
        for i in piece_list:
            if i[0]:
                white_piece_values.append(piece_values[i[1]])
            else:
                black_piece_values.append(piece_values[i[1]])

        black_sum = sum(black_piece_values)
        white_sum = sum(white_piece_values)

        return black_sum - white_sum

    def num_captures(self):
        pieces_indicies = []
        num_possible_caputres = []
        for piece in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]:
            pieces_indicies.append(list(self.board.pieces(piece, not self.board.turn)))

        pieces_indicies = list(filter(None, pieces_indicies))
        pieces_indicies = list(itertools.chain(*pieces_indicies))

        for piece in pieces_indicies:
            num_possible_caputres.append(list(self.board.attackers(self.board.turn, piece)))

        num_possible_caputres = list(filter(None, num_possible_caputres))
        num_possible_caputres = list(itertools.chain(*num_possible_caputres))

        return len(num_possible_caputres)

    def piece_mobility(self):

        old_fen = self.board.fen()
        fen0 = self.board.fen().split(" ")[0]
        fen_c = self.board.fen().split(" ")[1]
        fen_end = self.board.fen().split(" ")[2:]
        white_mobility, black_mobility = 1, 1

        new_fen = fen0 + ' w ' + ' '.join(fen_end) if fen_c == 'b' else fen0 + ' b ' + ' '.join(fen_end)

        if self.board.turn:
            white_mobility = len(list(self.board.pseudo_legal_moves))
        else:
            black_mobility = len(list(self.board.pseudo_legal_moves))

        self.board.set_fen(new_fen)

        if self.board.turn:
            white_mobility = len(list(self.board.pseudo_legal_moves))
        else:
            black_mobility = len(list(self.board.pseudo_legal_moves))

        self.board.set_fen(old_fen)

        mobility = 0

        try:
            white_mobility / black_mobility
        except:
            mobility = 0

        return mobility

    def eval_pos(self):
        global transposition
        index = chess.polyglot.zobrist_hash(self.board)
        eval_score = 0.0

        if index in transposition:
            eval_score = transposition.get(index)[0]
            return eval_score

        else:

            if self.board.result() == '*':
                if self.heuristic == 1:
                    eval_score = self.eval_material()
                elif self.heuristic == 2:
                    eval_score = self.num_captures()
                elif self.heuristic == 3:
                    eval_score = self.piece_mobility()
                elif self.heuristic == 12:
                    eval_score = self.eval_material() + self.num_captures()
                elif self.heuristic == 13:
                    eval_score = self.eval_material() + self.piece_mobility()
                elif self.heuristic == 23:
                    eval_score = self.num_captures() + self.piece_mobility()
                elif self.heuristic == 123:
                    eval_score = self.eval_material() + self.num_captures() + self.piece_mobility()

                transposition.update({index: [eval_score, self.board.ply()]})
                return eval_score
            else:
                if int(self.board.result().split("-")[0]) == 0:
                    return -np.inf
                elif int(self.board.result().split("-")[0]) == 1:
                    return np.inf
                else:
                    print("draw")
                    return 0

    def stockfish_eval(self):
        global engine
        info = engine.analyse(self.board, chess.engine.Limit(depth=10))
        return info["score"].pov(self.board.turn).score(mate_score=1000000)

    def sort_children(self):
        self.children = sorted(self.children, key=lambda x: x[1], reverse=True)
        self.children = [x[0] for x in self.children]

def iterativeDeepening(node, global_depth, time_s):
    startTime = time.time()
    time_limit = time_s

    def minimax(node, depth, player, alpha, beta):

        if depth == 0 or node.board.is_game_over() or time.time() - startTime > time_limit:
            return node.eval_pos(), 0

        for move in node.board.legal_moves:
            node.board.push_uci(str(move))
            new_board = node.board.copy()
            node.board.pop()
            child_node = Node(node, new_board, move, node.heuristic)
            node.children.append(child_node)

        if player:
            value = -np.inf
            which_child = None
            for child in node.children:

                value_child, _ = minimax(child, depth - 1, not player, alpha, beta)

                if value_child > value:
                    value = value_child
                    which_child = child
                if value >= beta:
                    break
                alpha = max(alpha, value)

            return value, which_child

        else:
            value = np.inf
            which_child = None
            for child in node.children:

                value_child, _ = minimax(child, depth - 1, not player, alpha, beta)

                if value_child < value:
                    value = value_child
                    which_child = child
                if value <= alpha:
                    break
                beta = min(beta, value)
            return value, which_child

    val = -np.inf
    for depth in range(1, global_depth):
        if time.time() - startTime > time_limit: break

        value_child, which_child = minimax(node, depth, node.board.turn, -np.inf, np.inf)
        if value_child > val:
            val, which_child = value_child, which_child

    return which_child

class MinMax(ExampleEngine):

    def __init__(self, *args):
        super().__init__(*args)
        global transposition
        transposition = {}

    def search(self, board, *args):

        bot_move = ""
        if board.turn:
            root = Node(None, board, None, h=123)
            best_child = iterativeDeepening(root, 10, 1)

            if best_child is not None:

                bot_move = best_child.move

            else:
                print("random move white to play")
                legal_moves = list(board.legal_moves)
                bot_move = random.choice(legal_moves)

        else:
            print("random_move black to play")
            legal_moves = list(board.legal_moves)
            bot_move = random.choice(legal_moves)

        return PlayResult(bot_move, None)




class AlphaZero(ExampleEngine):

    def __init__(self, *args):
        super().__init__(*args)


        self.model = "checkpoint_1500.pth.tar"
        self.num_mcts = 75

        self.cpuct = 2
        self.temp = 1.0
        self.game = AlphaZeroAgent.AntiChessGame(8)

        self.n1 = AlphaZeroAgent.NNetWrapper(self.game)
        self.n1.load_checkpoint('C:/Users/jerne/PycharmProjects/lichess-bot/models/',self.model)

        self.args1 = AlphaZeroAgent.dotdict({'numMCTSSims': self.num_mcts, 'cpuct': self.cpuct})
        self.mcts1 = AlphaZeroAgent.MCTS(self.game, self.n1, self.args1)
        self.agent1 = lambda x: np.argmax(self.mcts1.getActionProb(x, temp=self.temp))
        self.board = self.game.getInitBoard()
        self.curPlayer = 1


    def search(self, board, *args):

        copy_board = board.copy()
        move_stack = copy_board.move_stack

        lichess_move = ""
        if len(move_stack) != 0:
            lichess_move = copy_board.pop()

            lichess_action = AlphaZeroAgent.get_action_from_move(str(lichess_move))
            print("move stack != 0:", str(lichess_move),lichess_action)

            self.board, self.curPlayer = self.game.getNextState(self.board, self.curPlayer, lichess_action)

        action = self.agent1(self.game.getCanonicalForm(self.board, self.curPlayer))
        bot_move = AlphaZeroAgent.get_move_from_action(action)
        print("alphaze move:",bot_move)
        print("lichess move:",str(lichess_move))

        print("lichess fen:", board.fen())
        print("alphaze fen:", self.board.board.fen())
        self.board, self.curPlayer = self.game.getNextState(self.board, self.curPlayer, action)

        return PlayResult(bot_move, None)


class Alphabetical(ExampleEngine):
    def search(self, board, *args):
        moves = list(board.legal_moves)
        moves.sort(key=board.san)
        return PlayResult(moves[0], None)


class FirstMove(ExampleEngine):
    """Gets the first move when sorted by uci representation"""
    def search(self, board, *args):
        moves = list(board.legal_moves)
        moves.sort(key=str)
        return PlayResult(moves[0], None)
