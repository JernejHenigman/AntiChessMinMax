import torch.nn as nn
import torch.nn.functional as F
import os
from tqdm import tqdm
import torch
import torch.optim as optim
import logging
import math
import time
import pandas as pd
import chess.variant
import chess.pgn
import string
digs = string.digits + string.ascii_letters


EPS = 1e-8

log = logging.getLogger(__name__)

class AntiChessNNet(nn.Module):
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        super(AntiChessNNet, self).__init__()
        self.conv1 = nn.Conv2d(1, args.num_channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1)
        self.conv4 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1)

        self.bn1 = nn.BatchNorm2d(args.num_channels)
        self.bn2 = nn.BatchNorm2d(args.num_channels)
        self.bn3 = nn.BatchNorm2d(args.num_channels)
        self.bn4 = nn.BatchNorm2d(args.num_channels)

        self.fc1 = nn.Linear(args.num_channels*(self.board_x-4)*(self.board_y-4), 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, self.action_size)

        self.fc4 = nn.Linear(512, 1)

    def forward(self, s):
        #                                                           s: batch_size x board_x x board_y
        s = s.view(-1, 1, self.board_x, self.board_y)                # batch_size x 1 x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))                          # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn2(self.conv2(s)))                          # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn3(self.conv3(s)))                          # batch_size x num_channels x (board_x-2) x (board_y-2)
        s = F.relu(self.bn4(self.conv4(s)))                          # batch_size x num_channels x (board_x-4) x (board_y-4)
        s = s.view(-1, self.args.num_channels*(self.board_x-4)*(self.board_y-4))

        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=self.args.dropout, training=self.training)  # batch_size x 1024
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=self.args.dropout, training=self.training)  # batch_size x 512

        pi = self.fc3(s)                                                                         # batch_size x action_size
        v = self.fc4(s)                                                                          # batch_size x 1

        return F.log_softmax(pi, dim=1), torch.tanh(v)

class NeuralNet():
    """
    This class specifies the base NeuralNet class. To define your own neural
    network, subclass this class and implement the functions below. The neural
    network does not consider the current player, and instead only deals with
    the canonical form of the board.

    See othello/NNet.py for an example implementation.
    """

    def __init__(self, game):
        pass

    def train(self, examples):
        """
        This function trains the neural network with examples obtained from
        self-play.

        Input:
            examples: a list of training examples, where each example is of form
                      (board, pi, v). pi is the MCTS informed policy vector for
                      the given board, and v is its value. The examples has
                      board in its canonical form.
        """
        pass

    def predict(self, board):
        """
        Input:
            board: current board in its canonical form.

        Returns:
            pi: a policy vector for the current board- a numpy array of length
                game.getActionSize
            v: a float in [-1,1] that gives the value of the current board
        """
        pass

    def save_checkpoint(self, folder, filename):
        """
        Saves the current neural network (with its parameters) in
        folder/filename
        """
        pass

    def load_checkpoint(self, folder, filename):
        """
        Loads parameters of the neural network from folder/filename
        """
        pass


class AverageMeter(object):
    """From https://github.com/pytorch/examples/blob/master/imagenet/main.py"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0



    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class dotdict(dict):
    def __getattr__(self, name):
        return self[name]


args = dotdict({
    'lr': 0.2,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
    'num_channels': 512,
})


class NNetWrapper(NeuralNet):
    def __init__(self, game):
        self.nnet = AntiChessNNet(game, args)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

        if args.cuda:
            self.nnet.cuda()

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        optimizer = optim.Adam(self.nnet.parameters())

        for epoch in range(args.epochs):
            print('EPOCH ::: ' + str(epoch + 1))
            self.nnet.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()

            batch_count = int(len(examples) / args.batch_size)

            t = tqdm(range(batch_count), desc='Training Net')
            for _ in t:
                sample_ids = np.random.randint(len(examples), size=args.batch_size)
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                boards = [board.pieces for board in boards]
                boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                # predict
                if args.cuda:
                    boards, target_pis, target_vs = boards.contiguous().cuda(), target_pis.contiguous().cuda(), target_vs.contiguous().cuda()

                # compute output
                out_pi, out_v = self.nnet(boards)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                # record loss
                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))
                t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)

                # compute gradient and do SGD step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

    def predict(self, board):
        """
        board: np array with board
        """
        # timing

        # preparing input
        board = torch.FloatTensor(board.pieces.astype(np.float64))
        if args.cuda: board = board.contiguous().cuda()
        board = board.view(1, self.board_x, self.board_y)
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(board)

        # print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict': self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)

        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        map_location = None if args.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])


def int2base(x, base, length):
    if x < 0:
        sign = -1
    elif x == 0:
        return digs[0]
    else:
        sign = 1

    x *= sign
    digits = []

    while x:
        digits.append(digs[int(x % base)])
        x = int(x / base)

    if sign < 0:
        digits.append('-')

    while len(digits) < length: digits.extend(["0"])

    return list(map(lambda x: int(x), digits))


FILE_MAP = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
RANK_MAP = {'1': 7, '2': 6, '3': 5, '4': 4, '5': 3, '6': 2, '7': 1, '8': 0}

RESIDUALS_MAP = {0.875: 7, 0.75: 6, 0.625: 5, 0.5: 4, 0.375: 3, 0.25: 2, 0.125: 1, 0.0: 0}
PIECE_MAP = {"r": -4, "n": -2, "b": -3, "q": -5, "k": -6, "p": -1, "R": 4, "N": 2, "B": 3, "Q": 5, "K": 6, "P": 1}

FILE_MAP_REVERSE = {v: k for k, v in FILE_MAP.items()}
RANK_MAP_REVERSE = {v: k for k, v in RANK_MAP.items()}

PIECE_MAP = {"p": 1, "P": -1, "R": -4, "r": 4, "N": -2, "n": 2, "B": -3, "b": 3, "Q": -5, "q": 5, "K": -6, "k": 6}

PROMOTIONS = {'a2a1q': 0,
              'a2a1r': 11,
              'a2a1k': 12,
              'a2a1b': 13,
              'a2a1n': 14,
              'a2b1q': 15,
              'a2b1r': 19,
              'a2b1k': 20,
              'a2b1b': 21,
              'a2b1n': 22,
              'a7a8q': 23,
              'a7a8r': 25,
              'a7a8k': 26,
              'a7a8b': 28,
              'a7a8n': 29,
              'a7b8q': 30,
              'a7b8r': 31,
              'a7b8k': 33,
              'a7b8b': 34,
              'a7b8n': 35,
              'b2a1q': 37,
              'b2a1r': 38,
              'b2a1k': 39,
              'b2a1b': 41,
              'b2a1n': 42,
              'b2b1q': 43,
              'b2b1r': 44,
              'b2b1k': 46,
              'b2b1b': 47,
              'b2b1n': 49,
              'b2c1q': 50,
              'b2c1r': 51,
              'b2c1k': 52,
              'b2c1b': 53,
              'b2c1n': 55,
              'b7a8q': 57,
              'b7a8r': 58,
              'b7a8k': 59,
              'b7a8b': 60,
              'b7a8n': 61,
              'b7b8q': 62,
              'b7b8r': 65,
              'b7b8k': 76,
              'b7b8b': 77,
              'b7b8n': 78,
              'b7c8q': 79,
              'b7c8r': 84,
              'b7c8k': 85,
              'b7c8b': 86,
              'b7c8n': 87,
              'c2b1q': 88,
              'c2b1r': 90,
              'c2b1k': 91,
              'c2b1b': 93,
              'c2b1n': 94,
              'c2c1q': 95,
              'c2c1r': 96,
              'c2c1k': 98,
              'c2c1b': 99,
              'c2c1n': 100,
              'c2d1q': 102,
              'c2d1r': 103,
              'c2d1k': 104,
              'c2d1b': 106,
              'c2d1n': 107,
              'c7b8q': 108,
              'c7b8r': 109,
              'c7b8k': 111,
              'c7b8b': 112,
              'c7b8n': 114,
              'c7c8q': 115,
              'c7c8r': 116,
              'c7c8k': 117,
              'c7c8b': 118,
              'c7c8n': 120,
              'c7d8q': 122,
              'c7d8r': 123,
              'c7d8k': 124,
              'c7d8b': 125,
              'c7d8n': 126,
              'd2c1q': 127,
              'd2c1r': 130,
              'd2c1k': 141,
              'd2c1b': 142,
              'd2c1n': 143,
              'd2d1q': 149,
              'd2d1r': 150,
              'd2d1k': 151,
              'd2d1b': 152,
              'd2d1n': 153,
              'd2e1q': 155,
              'd2e1r': 156,
              'd2e1k': 158,
              'd2e1b': 159,
              'd2e1n': 160,
              'd7c8q': 161,
              'd7c8r': 163,
              'd7c8k': 164,
              'd7c8b': 165,
              'd7c8n': 167,
              'd7d8q': 168,
              'd7d8r': 169,
              'd7d8k': 171,
              'd7d8b': 172,
              'd7d8n': 173,
              'd7e8q': 174,
              'd7e8r': 176,
              'd7e8k': 177,
              'd7e8b': 179,
              'd7e8n': 180,
              'e2d1q': 181,
              'e2d1r': 182,
              'e2d1k': 183,
              'e2d1b': 184,
              'e2d1n': 185,
              'e2e1q': 187,
              'e2e1r': 188,
              'e2e1k': 189,
              'e2e1b': 190,
              'e2e1n': 191,
              'e2f1q': 195,
              'e2f1r': 200,
              'e2f1k': 206,
              'e2f1b': 207,
              'e2f1n': 208,
              'e7d8q': 214,
              'e7d8r': 215,
              'e7d8k': 217,
              'e7d8b': 218,
              'e7d8n': 220,
              'e7e8q': 221,
              'e7e8r': 223,
              'e7e8k': 224,
              'e7e8b': 225,
              'e7e8n': 226,
              'e7f8q': 228,
              'e7f8r': 229,
              'e7f8k': 230,
              'e7f8b': 232,
              'e7f8n': 233,
              'f2e1q': 234,
              'f2e1r': 236,
              'f2e1k': 237,
              'f2e1b': 238,
              'f2e1n': 239,
              'f2f1q': 240,
              'f2f1r': 241,
              'f2f1k': 242,
              'f2f1b': 244,
              'f2f1n': 245,
              'f2g1q': 246,
              'f2g1r': 247,
              'f2g1k': 248,
              'f2g1b': 249,
              'f2g1n': 250,
              'f7e8q': 252,
              'f7e8r': 253,
              'f7e8k': 254,
              'f7e8b': 255,
              'f7e8n': 260,
              'f7f8q': 264,
              'f7f8r': 265,
              'f7f8k': 271,
              'f7f8b': 272,
              'f7f8n': 273,
              'f7g8q': 279,
              'f7g8r': 280,
              'f7g8k': 282,
              'f7g8b': 283,
              'f7g8n': 285,
              'g2f1q': 286,
              'g2f1r': 289,
              'g2f1k': 290,
              'g2f1b': 291,
              'g2f1n': 293,
              'g2g1q': 294,
              'g2g1r': 295,
              'g2g1k': 296,
              'g2g1b': 297,
              'g2g1n': 298,
              'g2h1q': 299,
              'g2h1r': 301,
              'g2h1k': 302,
              'g2h1b': 303,
              'g2h1n': 304,
              'g7f8q': 305,
              'g7f8r': 306,
              'g7f8k': 307,
              'g7f8b': 309,
              'g7f8n': 310,
              'g7g8q': 311,
              'g7g8r': 312,
              'g7g8k': 313,
              'g7g8b': 314,
              'g7g8n': 315,
              'g7h8q': 317,
              'g7h8r': 318,
              'g7h8k': 319,
              'g7h8b': 325,
              'g7h8n': 328,
              'h2g1q': 329,
              'h2g1r': 330,
              'h2g1k': 336,
              'h2g1b': 337,
              'h2g1n': 338,
              'h2h1q': 344,
              'h2h1r': 345,
              'h2h1k': 347,
              'h2h1b': 348,
              'h2h1n': 350,
              'h7g8q': 351,
              'h7g8r': 352,
              'h7g8k': 354,
              'h7g8b': 355,
              'h7g8n': 356,
              'h7h8q': 358,
              'h7h8r': 359,
              'h7h8k': 361,
              'h7h8b': 362,
              'h7h8n': 363}

REVERSE_PROMOTIONS = {v: k for k, v in PROMOTIONS.items()}



def get_move_from_action(action):
    if action in REVERSE_PROMOTIONS:
        move = REVERSE_PROMOTIONS[action]

    else:
        move = int2base(action, 8, 4)
        move = FILE_MAP_REVERSE[move[0]] + "" + RANK_MAP_REVERSE[move[1]] + "" + FILE_MAP_REVERSE[move[2]] + "" + \
               RANK_MAP_REVERSE[move[3]]

    return move


def get_action_from_move(move):
    action = -1
    if len(move) > 4:
        action = PROMOTIONS[move]
    else:
        x1, y1, x2, y2 = FILE_MAP[move[0]], RANK_MAP[move[1]], FILE_MAP[move[2]], RANK_MAP[move[3]]
        action = x1 + y1 * 8 + x2 * 8 ** 2 + y2 * 8 ** 3

    return action

def get_move_evaluation(engine, board, time_per_move=0.01):
    list_moves = []
    for el in board.legal_moves:
        info = engine.analyse(board, chess.engine.Limit(time=time_per_move), root_moves=[el])
        t = str(info["score"])
        list_moves.append((str(el), info["score"].pov(board.turn).score(mate_score=1000000)))

    sorted_list_moves = sorted(list_moves, key=lambda x: x[1], reverse=True)

    return dict(sorted_list_moves)


from sys import platform

class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, args, dirichlet_noise=False):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.dirichlet_noise = dirichlet_noise
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s

        self.time_sims = 0

        self.time_nnet = 0
        self.time_vali = 0
        self.time_next = 0
        self.time_diri = 0
        self.time_pick = 0

    def getActionProb(self, canonicalBoard, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.
        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """

        probs = np.zeros(4096)
        if canonicalBoard.board.fullmove_number < 2:
            try:
                if canonicalBoard.board.fullmove_number == 1 and canonicalBoard.board.turn:
                    probs = self.opening_book(canonicalBoard.board, book1)

                elif canonicalBoard.board.fullmove_number == 1 and not canonicalBoard.board.turn:
                    copy_board = canonicalBoard.board.copy()
                    previous = copy_board.pop()
                    move_chosen_previously = copy_board.san(chess.Move.from_uci(str(previous)))
                    probs = self.opening_book(canonicalBoard.board, book2[move_chosen_previously])


                elif canonicalBoard.board.fullmove_number == 2 and canonicalBoard.board.turn:
                    copy_board = canonicalBoard.board.copy()
                    previous = copy_board.pop()
                    move_chosen_previously = copy_board.san(chess.Move.from_uci(str(previous)))
                    previous_previous = copy_board.pop()
                    move_chosen_previously_previously = copy_board.san(chess.Move.from_uci(str(previous_previous)))
                    move_chosen_previously1 = move_chosen_previously_previously + move_chosen_previously
                    probs = self.opening_book(canonicalBoard.board, book3[move_chosen_previously1])


                else:
                    copy_board = canonicalBoard.board.copy()
                    previous = copy_board.pop()
                    move_chosen_previously = copy_board.san(chess.Move.from_uci(str(previous)))
                    previous_previous = copy_board.pop()
                    move_chosen_previously_previously = copy_board.san(chess.Move.from_uci(str(previous_previous)))
                    previous_previous_previous = copy_board.pop()
                    move_chosen_previously_previously_previously = copy_board.san(
                        chess.Move.from_uci(str(previous_previous_previous)))
                    move_chosen_previously1 = move_chosen_previously_previously_previously + move_chosen_previously_previously + move_chosen_previously
                    probs = self.opening_book(canonicalBoard.board, book4[move_chosen_previously1])

            except Exception as x:
                print(x)
                for i in range(self.args.numMCTSSims):
                    dir_noise = (i == 0 and self.dirichlet_noise)
                    self.search(canonicalBoard, dirichlet_noise=dir_noise)

                    s = self.game.stringRepresentation(canonicalBoard)
                    counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.getActionSize())]

                if temp == 0:
                    bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
                    bestA = np.random.choice(bestAs)
                    probs = [0] * len(counts)
                    probs[bestA] = 1
                    return probs

                counts = [x ** (1. / temp) for x in counts]
                counts_sum = float(sum(counts))
                probs = [x / counts_sum for x in counts]



        else:

            for i in range(self.args.numMCTSSims):
                dir_noise = (i == 0 and self.dirichlet_noise)
                self.search(canonicalBoard, dirichlet_noise=dir_noise)

            s = self.game.stringRepresentation(canonicalBoard)
            counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.getActionSize())]

            if temp == 0:
                bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
                bestA = np.random.choice(bestAs)
                probs = [0] * len(counts)
                probs[bestA] = 1
                return probs

            counts = [x ** (1. / temp) for x in counts]
            counts_sum = float(sum(counts))
            probs = [x / counts_sum for x in counts]

        return probs

    def search(self, canonicalBoard, dirichlet_noise=False):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.
        Once a leaf node is found, the neural network is called to return an
        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.
        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.
        Returns:
            v: the negative of the value of the current canonicalBoard
        """


        s = self.game.stringRepresentation(canonicalBoard)

        if canonicalBoard.board.is_fifty_moves() or canonicalBoard.board.is_fivefold_repetition():
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)
            if self.Es[s] != 0:
                # terminal node
                return -self.Es[s]

        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)
        if self.Es[s] != 0:
            # terminal node
            return -self.Es[s]

        if s not in self.Ps:
            # leaf node

            start = time.time()
            self.Ps[s], v = self.nnet.predict(canonicalBoard)
            end = time.time()
            self.time_nnet += (end - start)

            start = time.time()
            valids = self.game.getValidMoves(canonicalBoard, 1)

            end = time.time()
            self.time_vali += (end - start)
            self.Ps[s] = self.Ps[s] * valids  # masking invalid moves

            start = time.time()
            if self.dirichlet_noise:
                self.applyDirNoise(s, valids)

            end = time.time()
            self.time_diri += (end - start)
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s  # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.
                log.error("All valid moves were masked, doing a workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v

        valids = self.Vs[s]
        start = time.time()
        if self.dirichlet_noise:
            self.applyDirNoise(s, valids)
            sum_Ps_s = np.sum(self.Ps[s])
            self.Ps[s] /= sum_Ps_s  # renormalize
        end = time.time()
        self.time_diri += (end - start)
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        start = time.time()
        # for a in range(self.game.getActionSize()):
        #   if valids[a]:
        for a in np.argwhere(valids):
            a = a[0]
            if valids[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (
                            1 + self.Nsa[(s, a)])

                else:
                    u = self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = a

        end = time.time()
        self.time_pick += (end - start)

        a = best_act

        start = time.time()
        next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
        end = time.time()
        self.time_next += (end - start)
        next_s = self.game.getCanonicalForm(next_s, next_player)
        v = self.search(next_s)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1

        return -v

    def applyDirNoise(self, s, valids):
        dir_values = np.random.dirichlet([self.args.dirichletAlpha] * np.count_nonzero(valids))
        dir_idx = 0
        s_policy = self.Ps[s]
        s_policy = np.argwhere(s_policy)  # optimization
        # for idx in range(len(self.Ps[s])):
        for idx in s_policy:
            idx = idx[0]
            if self.Ps[s][idx]:
                self.Ps[s][idx] = (0.75 * self.Ps[s][idx]) + (0.25 * dir_values[dir_idx])
                dir_idx += 1


'''
Board class for the game of AntiChessGame.

pieces[0][0] is the top left square,
pieces[7][0] is the bottom left square,

'''

class Board():

    def __init__(self):
        self.board = chess.variant.AntichessBoard()
        self.board.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1")
        # self.board.set_fen("r3k2r/p2p3p/8/q7/Q7/8/P2P3P/R3K2R w - - 0 1")
        self.pieces = np.zeros((8, 8)).astype(int)
        self.n = 8
        self.from_piece_map_to_pieces()

    def __str__(self):
        return str(self.get_player_to_move()) + ''.join(str(r) for v in self.pieces for r in v)

    def getCopy(self):
        b = Board()
        b.board = self.board.copy()
        b.pieces = np.copy(np.array(self.pieces))
        return b

    def set_fen(self, fen):
        self.board.set_fen(fen)

    def get_player_to_move(self):

        if self.board.turn:
            return "white"
        else:
            return "black"

    def get_legal_moves(self):
        moves = []
        for move in self.board.legal_moves:
            moves.append(list(str(move)))

        return moves

    def execute_move(self, move, color):
        """Perform the given move on the board.
        color gives the color pf the piece to play (1=white,-1=black)
        """

        try:
            self.board.push_uci(move)
            self.from_piece_map_to_pieces()
        except Exception as e:
            print(e)

            print("in exeption:",move)
            action = 0
            if len(move) > 4:  # if promotion move, we map moves differently
                action = PROMOTIONS["".join(move)]
                #print("in expeption move is:",move)
                #print("in expeption move is jon:","".join(move))

            else:
                x1, y1, x2, y2 = FILE_MAP[move[0]], RANK_MAP[move[1]], FILE_MAP[move[2]], RANK_MAP[move[3]]
                action = x1 + y1 * 8 + x2 * 8 ** 2 + y2 * 8 ** 3

            valids = [0] * 4096
            legalMoves = self.get_legal_moves()
            #print(legalMoves)
            #print("action",action)

            # need to optimize
            for move in legalMoves:
                if len(move) > 4:  # if promotion move, we map moves differently
                    valid_index = PROMOTIONS["".join(move)]
                    print(move+" ",valid_index)
                    valids[valid_index] = 1
                else:
                    x1, y1, x2, y2 = FILE_MAP[move[0]], RANK_MAP[move[1]], FILE_MAP[move[2]], RANK_MAP[move[3]]
                    valids[x1 + y1 * self.n + x2 * self.n ** 2 + y2 * self.n ** 3] = 1

            if valids[action] == 0:
                valid_actions = np.where(np.array(valids) == 1)
                action = random.choice(valid_actions[0])

                move = "a1a1"
                if action in REVERSE_PROMOTIONS:
                    move = REVERSE_PROMOTIONS[action]
                    #print("promotion move chosen:", move)

                else:
                    move = int2base(action, 8, 4)
                    move = FILE_MAP_REVERSE[move[0]] + "" + RANK_MAP_REVERSE[move[1]] + "" + FILE_MAP_REVERSE[
                        move[2]] + "" + RANK_MAP_REVERSE[move[3]]

                    #print("move randomly chosen:", move)

            print("played at the end expetionn:",move)
            self.board.push_uci(move)
            self.from_piece_map_to_pieces()

    # need to optimze
    def from_piece_map_to_pieces(self):

        square_at_index = 0
        for i in reversed(range(0, 8)):
            for j in range(0, 8):
                try:
                    color = self.board.piece_at(square=square_at_index).color
                    piece_type = self.board.piece_at(square=square_at_index).piece_type
                    if color == False:
                        piece_type = piece_type * (-1)
                except:
                    piece_type = 0
                self.pieces[i][j] = piece_type
                square_at_index = square_at_index + 1




class Game():
    """
    This class specifies the base Game class. To define your own game, subclass
    this class and implement the functions below. This works when the game is
    two-player, adversarial and turn-based.

    Use 1 for player1 and -1 for player2.

    See othello/OthelloGame.py for an example implementation.
    """

    def __init__(self):
        pass

    def getInitBoard(self):
        """
        Returns:
            startBoard: a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        """
        pass

    def getBoardSize(self):
        """
        Returns:
            (x,y): a tuple of board dimensions
        """
        pass

    def getActionSize(self):
        """
        Returns:
            actionSize: number of all possible actions
        """
        pass

    def getNextState(self, board, player, action):
        """
        Input:
            board: current board
            player: current player (1 or -1)
            action: action taken by current player

        Returns:
            nextBoard: board after applying action
            nextPlayer: player who plays in the next turn (should be -player)
        """
        pass

    def getValidMoves(self, board, player):
        """
        Input:
            board: current board
            player: current player

        Returns:
            validMoves: a binary vector of length self.getActionSize(), 1 for
                        moves that are valid from the current board and player,
                        0 for invalid moves
        """
        pass

    def getGameEnded(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.

        """
        pass

    def getCanonicalForm(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            canonicalBoard: returns canonical form of board. The canonical form
                            should be independent of player. For e.g. in chess,
                            the canonical form can be chosen to be from the pov
                            of white. When the player is white, we can return
                            board as is. When the player is black, we can invert
                            the colors and return the board.
        """
        pass

    def getSymmetries(self, board, pi):
        """
        Input:
            board: current board
            pi: policy vector of size self.getActionSize()

        Returns:
            symmForms: a list of [(board,pi)] where each tuple is a symmetrical
                       form of the board and the corresponding pi vector. This
                       is used when training the neural network from examples.
        """
        pass

    def stringRepresentation(self, board):
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        pass


class AntiChessGame(Game):
    """
    This class specifies the base Game class. To define your own game, subclass
    this class and implement the functions below. This works when the game is
    two-player, adversarial and turn-based.
    Use 1 for player1 and -1 for player2.
    See othello/OthelloGame.py for an example implementation.
    """

    def __init__(self,n):
        self.n = n
        self.getInitBoard()

    def getInitBoard(self):
        """
        Returns:
            startBoard: a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        """
        board = Board()
        return board


    def getBoardSize(self):
        """
        Returns:
            (x,y): a tuple of board dimensions
        """
        return (self.n,self.n)

    def getActionSize(self):
        """
        Returns:
            actionSize: number of all possible actions
        """
        return self.n ** 4

    def getNextState(self, board, player, action):
        """
        Input:
            board: current board
            player: current player (1 or -1)
            action: action taken by current player
        Returns:
            nextBoard: board after applying action
            nextPlayer: player who plays in the next turn (should be -player)
        """


        b = board.getCopy()

        if action in REVERSE_PROMOTIONS:
            move = REVERSE_PROMOTIONS[action]

        else:
            move = int2base(action, self.n, 4)
            move = FILE_MAP_REVERSE[move[0]]+""+RANK_MAP_REVERSE[move[1]]+""+FILE_MAP_REVERSE[move[2]]+""+RANK_MAP_REVERSE[move[3]]

        b.execute_move(move, player)
        return (b, -player)


    def getValidMoves(self, board, player):
        """
        Input:
            board: current board
            player: current player
        Returns:
            validMoves: a binary vector of length self.getActionSize(), 1 for
                        moves that are valid from the current board and player,
                        0 for invalid moves
        """

        valids = [0] * self.getActionSize()
        b = board.getCopy()
        legalMoves = b.get_legal_moves()

        #need to optimize
        for move in legalMoves:
            if len(move) > 4: #if promotion move, we map moves differently
                valid_index = PROMOTIONS["".join(move)]

                valids[valid_index] = 1
            else:
                x1, y1, x2, y2 =  FILE_MAP[move[0]],RANK_MAP[move[1]],FILE_MAP[move[2]],RANK_MAP[move[3]]
                valids[x1 + y1 * b.n + x2 * b.n ** 2 + y2 * b.n ** 3] = 1

        return np.array(valids)

    def getGameEnded(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)
        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.

        """

        b = board.getCopy()

        if b.board.is_fivefold_repetition():
            #print("five fold")
            return 0.0001

        if b.board.is_fifty_moves():
            #print("fifty")
            return 0.0001

        if b.board.is_game_over():

            result = b.board.outcome().result().split("-")
            if result[0] == "1":
                return 1 if b.board.turn else -1
            elif result[0] == "0":
                return -1 if b.board.turn else 1
            else:
                return 0.0001

        else:
            return 0

    def getCanonicalForm(self, board, player):
        b = board.getCopy()
        # rules and objectives are different for the different players, so inverting board results in an invalid state.
        return b

    def getSymmetries(self, board, pi):

        """
        Input:
            board: current board
            pi: policy vector of size self.getActionSize()
        Returns:
            symmForms: a list of [(board,pi)] where each tuple is a symmetrical
                       form of the board and the corresponding pi vector. This
                       is used when training the neural network from examples.
        """
        return [(board, pi)]


    def stringRepresentation(self, board):
        """
        Input:
            board: current board
        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        board_s = board.board.fen().split(" ")
        board_s = board_s[0] + board_s[1] + board_s[3]
        return board_s


import chess.engine
import chess.variant


class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        while valids[a] != 1:
            a = np.random.randint(self.game.getActionSize())

        return a


class HumanAntiChessPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board, a):
        # display(board)
        valid = self.game.getValidMoves(board, 1)
        # while True:
        # a = input()
        # move = [x for x in a.strip().split(' ')]
        # x1, y1, x2, y2 = FILE_MAP[move[0]], RANK_MAP[move[1]], FILE_MAP[move[2]], RANK_MAP[move[3]]
        # a = x1 + y1 * self.game.n + x2 * self.game.n**2 + y2 * self.game.n**3
        # if valid[a]:
        #    break
        # else:
        #    print('Invalid')

        return a


class StockFishPlayer():
    def __init__(self, game, time):
        self.game = game
        self.time = time
        self.engine = chess.engine.SimpleEngine.popen_uci("./fairy-stockfish-largeboard_x86-64.exe")

    def play(self, board):

        result = self.engine.play(board.board, chess.engine.Limit(time=self.time))
        move = str(result.move)

        a = 0
        if len(move) > 4:  # if promotion move, we map moves differently
            a = PROMOTIONS["".join(move)]
        else:
            x1, y1, x2, y2 = FILE_MAP[move[0]], RANK_MAP[move[1]], FILE_MAP[move[2]], RANK_MAP[move[3]]
            a = x1 + y1 * 8 + x2 * 8 ** 2 + y2 * 8 ** 3

        return a

    def close_engine(self):
        self.engine.quit()


import sys
import chess.variant
import numpy as np
import random

FILE_MAP = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
RANK_MAP = {'1': 7, '2': 6, '3': 5, '4': 4, '5': 3, '6': 2, '7': 1, '8': 0}

FILE_MAP_REVERSE = {v: k for k, v in FILE_MAP.items()}
RANK_MAP_REVERSE = {v: k for k, v in RANK_MAP.items()}

book1 = [('Nh3', 0.5712763079410298, 613598),
         ('g4', 0.5593962228771996, 511977),
         ('c4', 0.5575896103276872, 671128),
         ('e3', 0.5400545955905178, 6281826),
         ('Na3', 0.5120225422667501, 143730),
         ('g3', 0.5033715727067002, 2752573),
         ('b3', 0.5030213830888164, 1471677)]

book2 = {'Nh3': [('b6', 0.5233891279252807, 76232), ('e6', 0.4909453202836547, 85315)],
         'Na3': [('b5', 0.522741518942908, 79766), ('e6', 0.43458378859617175, 24711)],
         'g3': [('b6', 0.5142890497314966, 642450),
                ('e6', 0.5079845259703353, 272715),
                ('b5', 0.4958576700310369, 81838),
                ('g5', 0.481261861764264, 326153),
                ('g6', 0.4464190490700737, 735112),
                ('Nh6', 0.4197372307771342, 155726),
                ('c5', 0.41346334331130274, 101223)],
         'g4': [('b6', 0.508440257030104, 48399), ('g6', 0.43230419422730004, 106432)],
         'b3': [('g6', 0.5080923761298566, 295834),
                ('b5', 0.470743616835756, 194966),
                ('Na6', 0.4641160949868074, 41690),
                ('b6', 0.4623230805724678, 354116),
                ('e6', 0.45675518025688905, 255441),
                ('c5', 0.39718369198859216, 61712)],
         'e3': [('b5', 0.5002405564398114, 4287975),
                ('g5', 0.4959632008882544, 63045),
                ('c5', 0.4866542046567225, 314556)],
         'c4': [('g6', 0.4939559529723464, 54351),
                ('c5', 0.4887156775907883, 28225),
                ('e6', 0.48395434612145327, 34433),
                ('c6', 0.47802340060905696, 81109)]}

book3 = {'c4g6': [('a4', 0.6381637028629084, 7929),
                  ('e3', 0.5451866404715128, 6108),
                  ('c5', 0.5006047899362954, 12401)],
         'b3b6': [('c4', 0.6359555079067274, 14924),
                  ('Na3', 0.5247058823529411, 9775),
                  ('e3', 0.5061101332662811, 19885),
                  ('b4', 0.4909090909090909, 37840)],
         'g3e6': [('b4', 0.6316940149021786, 169237)],
         'b3g6': [('e3', 0.6262023789001896, 29005),
                  ('c4', 0.5560526969922943, 8046),
                  ('Na3', 0.5364141084258655, 6124),
                  ('g4', 0.5229966634099582, 54247),
                  ('g3', 0.5040465690597989, 141725)],
         'b3c5': [('b4', 0.3600178826895565, 343434)],  # move added
         'g3b6': [('c4', 0.6169710650523291, 19492),
                  ('Nh3', 0.6117384179949352, 33565),
                  ('b3', 0.5372625389295971, 395907),
                  ('b4', 0.49710701346455416, 73972)],
         'Nh3b6': [('Nf4', 0.6018914658860617, 4441),
                   ('g3', 0.5712875885289679, 34452)],
         'g3b5': [('Nh3', 0.5986410496719775, 4268),
                  ('b3', 0.5488695284116962, 28351)],
         'e3c5': [('Bd3', 0.5886844526218952, 21740),
                  ('Bb5', 0.513766309391662, 54953),
                  ('Bc4', 0.4956475863888156, 7582)],
         'g4b6': [('g5', 0.5592344598513207, 18967),
                  ('c4', 0.515621383938903, 4321),
                  ('b3', 0.5153460147041662, 8569)],
         'g3g6': [('Nh3', 0.5500178826895565, 55920)],
         'g3g5': [('b3', 0.5413560477912909, 149567),
                  ('c4', 0.5176667681998173, 6566)],
         'c4c6': [('Qb3', 0.5360107095046854, 7470),
                  ('g3', 0.49665551839464883, 13156),
                  ('Na3', 0.4920952571542926, 9994)],
         'b3e6': [('e3', 0.5358708189158016, 8670), ('b4', 0.49342499095484, 152015)],
         'g3Nh6': [('d3', 0.5348615090735435, 8376),
                   ('Nh3', 0.49551020408163265, 8575)],
         'b3b5': [('a4', 0.5239007468578899, 77257)],
         'e3g5': [('Ba6', 0.5230626234460323, 51642)],
         'Na3b5': [('Nxb5', 0.5226981373953699, 79566)],
         'Na3e6': [('b4', 0.4626981373953699, 16000)],  # move added
         'g3c5': [('Nh3', 0.5112291716976576, 8282),
                  ('c3', 0.49405964600339447, 12373)],
         'c4c5': [('b4', 0.508867515427809, 17987)],
         'g4g6': [('Nh3', 0.5059642147117296, 11066)],
         'c4e6': [('c5', 0.5035268817204301, 23250)],
         'e3b5': [('Bxb5', 0.5004159487016842, 4281778)],
         'Nh3e6': [('Ng5', 0.49984267164523505, 47671)],
         'b3Na6': [('g3', 0.49800148038490005, 6755)]}

book4 = {'b3b6Na3': [('g6', 0.5496535796766744, 2598),
                     ('b5', 0.5089748703629836, 5014)],
         'b3g6Na3': [('b5', 0.5488435374149659, 3675)],
         'Na3b5Nxb5': [('d6', 0.5484496124031008, 2064),
                       ('a6', 0.5210205316821082, 21479),
                       ('g5', 0.5148188803512623, 911),
                       ('e6', 0.5055884286653517, 4563),
                       ('g6', 0.4986753017368266, 6794),
                       ('c6', 0.49092588926285224, 15153),
                       ('Bb7', 0.4844649021864212, 8690),
                       ('Nh6', 0.4797979797979798, 1386)],
         'g3c5Nh3': [('g5', 0.547707100591716, 2704), ('c4', 0.500438596491228, 2280)],
         'g3g5b3': [('h5', 0.5456081081081081, 1184),
                    ('b5', 0.5452996646880731, 92153),
                    ('b6', 0.5388299851190477, 21504),
                    ('Nh6', 0.5020785219399538, 2165),
                    ('a6', 0.4605263157894737, 836),
                    ('c5', 0.43672537508153947, 9198)],
         'g3e6b4': [('Bxb4', 0.65, 85000)],  # move added
         'c4g6c5': [('d6', 0.5450292397660819, 855),
                    ('c6', 0.5260232440626579, 5937),
                    ('b6', 0.4218248509857863, 4362)],
         'g3b6b3': [('Nh6', 0.5446623093681917, 2754),
                    ('g6', 0.5417104433704559, 233191),
                    ('e5', 0.5404651162790698, 2150),
                    ('g5', 0.5098690735965087, 40328),
                    ('b5', 0.5019659239842726, 6104),
                    ('e6', 0.45127866688665236, 30305),
                    ('c5', 0.43456435998565796, 8367),
                    ('Na6', 0.4312586022327573, 13078),
                    ('a5', 0.41856586992773764, 3598)],
         'e3c5Bd3': [('h5', 0.5419520547945206, 1168),
                     ('a5', 0.49820750738085195, 9484)],
         'e3b5Bxb5': [('a5', 0.5398130841121496, 5350),
                      ('c6', 0.534215136540309, 318514),
                      ('h5', 0.5209603140649772, 198685),
                      ('c5', 0.520682503032912, 25553),
                      ('Ba6', 0.5159343554741411, 398510),
                      ('g5', 0.4774140656493598, 220779),
                      ('Bb7', 0.46449888798869665, 1075079),
                      ('e6', 0.46372888559907355, 1110498),
                      ('e5', 0.42771550770592137, 54244),
                      ('Nh6', 0.4026486996224762, 314947)],
         'b3e6e3': [('Ba3', 0.5394996708360764, 6076),
                    ('b5', 0.4649425287356322, 1740)],
         'c4g6a4': [('c6', 0.538878842676311, 1106)],
         'g4g6Nh3': [('g5', 0.5369484316852738, 3762),
                     ('h5', 0.3994227994227994, 3465)],
         'c4c6Qb3': [('g6', 0.5345528455284553, 984),
                     ('b5', 0.48127880184331795, 3472)],
         'b3g6e3': [('b5', 0.5317415001297691, 19265)],
         'g3b5b3': [('c6', 0.5306345733041575, 914),
                    ('c5', 0.5285839929639402, 2274),
                    ('Na6', 0.41935483870967744, 2511),
                    ('f5', 0.36718115353371245, 3693)],
         'g3b6b4': [('e6', 0.5300638226402419, 2977),
                    ('g5', 0.4925568655472192, 8397),
                    ('g6', 0.45753402528169296, 24761),
                    ('a5', 0.4547487884687073, 24143)],
         'b3g6g3': [('b6', 0.5272677171271826, 79838),
                    ('g5', 0.48231884057971014, 1725),
                    ('b5', 0.4515164835164835, 11375),
                    ('e6', 0.44247891283973756, 8536),
                    ('d6', 0.4046783625730994, 5130),
                    ('c5', 0.33224596313815036, 6131),
                    ('Nh6', 0.3031490741992161, 7399)],
         'b3g6g4': [('b5', 0.5249866619242397, 5623),
                    ('b6', 0.4823086010012938, 17777),
                    ('h5', 0.48104812129202373, 18204),
                    ('c5', 0.3928150765606596, 1698)],
         'b3b5a4': [('bxa4', 0.5239007468578899, 77257)],
         'e3c5Bb5': [('h5', 0.520891364902507, 1077),
                     ('g5', 0.5171171171171172, 1665),
                     ('c4', 0.4932769395871473, 15841),
                     ('Nh6', 0.4837662337662338, 1540),
                     ('Qc7', 0.44179875206196656, 13943),
                     ('a5', 0.39246178176067475, 3794)],
         'g3g5c4': [('b5', 0.5138721351025332, 4145)],
         'g4b6g5': [('g6', 0.5118674798088018, 12134)],
         'Nh3b6Nf4': [('d5', 0.5106970324361628, 1449)],
         'c4c5b4': [('cxb4', 0.508867515427809, 17987)],
         'g3c5c3': [('Nc6', 0.5057899090157155, 2418),
                    ('g5', 0.5051993067590987, 1154),
                    ('Na6', 0.4551219512195122, 2050),
                    ('Qb6', 0.41359447004608296, 868),
                    ('g6', 0.4053058216654385, 1357)],
         'c4g6e3': [('g5', 0.5041518386714117, 843),
                    ('b5', 0.49504950495049505, 3636)],
         'c4e6c5': [('Bxc5', 0.5035268817204301, 23250)],
         'Nh3b6g3': [('e6', 0.5014156285390714, 3532)],
         'Nh3e6Ng5': [('Qxg5', 0.49984267164523505, 47671)],
         'b3b6b4': [('g5', 0.49654491609081935, 1013),
                    ('a5', 0.4599132503148174, 14294),
                    ('g6', 0.43541324575807333, 14616)],
         'g4b6b3': [('g6', 0.4945748170577845, 3963),
                    ('f5', 0.4936092513694461, 1643)],
         'b3b6e3': [('Na6', 0.49361702127659574, 1410),
                    ('b5', 0.4372466417613862, 12581)],
         'b3e6b4': [('Bxb4', 0.49342499095484, 152015)],
         'c4c6g3': [('b5', 0.49159327461969576, 1249),
                    ('g6', 0.48589796764827875, 4822),
                    ('g5', 0.4839071856287425, 2672)],
         'g3g6Nh3': [('Nh6', 0.48918734350102433, 4393),
                     ('d6', 0.4500654164849542, 4586),
                     ('c5', 0.4324580828851629, 3161)],
         'e3g5Ba6': [('Nxa6', 0.4829725313999796, 39172)],
         'g3Nh6d3': [('g5', 0.4784345047923323, 6260)],
         'c4c6Na3': [('b5', 0.47724801987886317, 6439),
                     ('g6', 0.44306418219461696, 966)],
         'g4b6c4': [('b5', 0.4675223648385842, 2571)],
         'g3b6c4': [('b5', 0.5975223648385842, 11111)],  # mvoe added
         'Na3e6b4': [('Bxb4', 0.4675223648385842, 16111)],  # move addded
         'g3b5Nh3': [('g5', 0.6275223648385842, 2000)],  # move addded
         'b3b6c4': [('b5', 0.6075223648385842, 8000)],  # move addded
         'b3c5b4': [('cxb4', 0.3675223648385842, 34000)],  # move addded
         'g3b6Nh3': [('e6', 0.46467817896389324, 3185)],
         'e3c5Bc4': [('d5', 0.4586518415566366, 2878)],
         'g3Nh6Nh3': [('g5', 0.44689626727160237, 4849)],
         'b3g6c4': [('b5', 0.40689626727160237, 1000)],  # move added
         'b3Na6g3': [('b6', 0.3951727401798391, 2113)]}