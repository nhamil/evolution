import chess 
from stockfish import Stockfish 
import numpy as np 

engine = Stockfish("./stockfish/stockfish_4_x64.exe", parameters={
    "Write Debug Log": "false",
    "Contempt": 1,
    "Min Split Depth": 0,
    "Threads": 8,
    "Ponder": "false",
    "Hash": 16,
    "MultiPV": 1,
    "Skill Level": 20,
    "Move Overhead": 1,
    "Minimum Thinking Time": 1,
    "Slow Mover": 1,
    "UCI_Chess960": "false",
})

def game_done(stats): 
        if 'type' in stats and stats['type'] == 'cp': 
            return False 
        if 'value' in stats: 
            if stats['value'] == 0: 
                return True 

        return False 

def game_mate(stats): 
    return 'type' in stats and stats['type'] == 'mate' and stats['value'] == 0 

pieces = {
    'P': 1, 
    'R': 2, 
    'N': 3, 
    'B': 4, 
    'Q': 5, 
    'K': 6, 
    'p': -1, 
    'r': -2, 
    'n': -3, 
    'b': -4, 
    'q': -5, 
    'k': -6, 
    '.': 0 
}

class ChessEnv: 

    def __init__(self): 
        self.reset(None) 

    def reset(self, outfile): 
        self.board = chess.Board() 
        self.moves = [] 
        self.cur = np.zeros((6, 8, 8)) 
        self.last = np.zeros((6, 8, 8)) 
        self.outfile = outfile 
        self.r = outfile is not None  

        return self._get_moves() 

    def step(self, move): 
        f = self.outfile 
        reward = 0

        self.last[:] = self.cur 
        self.moves.append(move) 
        self.board.push_uci(move) 
        engine.set_position(self.moves) 
        self._get_board_rep(self.cur) 
        reward += self._reward() 
        if self.r: 
            f.write("\nWhite: {}\n".format(move))
            f.write("{}\n".format(self.board)) 

        stats = engine.get_evaluation() 
        if game_done(stats): 
            if self.r: 
                f.write("Done!\n")
            if game_mate(stats): 
                reward += 100
                if self.r: 
                    f.write("Won\n")
            else: 
                print('white -> draw?', stats)
                print(self.board)
                if self.r: 
                    f.write("Draw\n")
            return None, None, reward, True

        move = engine.get_best_move() 
        self.last[:] = self.cur 
        self.moves.append(move)
        self.board.push_uci(move) 
        engine.set_position(self.moves) 
        self._get_board_rep(self.cur) 
        reward += self._reward() 
        if self.r: 
            f.write("\nBlack: {}\n".format(move))
            f.write("{}\n".format(self.board)) 

        stats = engine.get_evaluation() 
        if game_done(stats): 
            if self.r: 
                f.write("Done!\n")
            if game_mate(stats): 
                reward -= 100
                if self.r: 
                    f.write("Lost\n")
            else: 
                print('black -> draw?', stats)
                print(self.board)
                if self.r: 
                    f.write("Draw\n")
            return None, None, reward, True

        return *self._get_moves(), reward, False 

    def _reward(self): 
        out = 0 
        out -= 1 * (np.sum(self.last[0]) - np.sum(self.cur[0])) # pawn 
        out -= 5 * (np.sum(self.last[1]) - np.sum(self.cur[1])) # rook 
        out -= 3 * (np.sum(self.last[2]) - np.sum(self.cur[2])) # knight
        out -= 3.5 * (np.sum(self.last[3]) - np.sum(self.cur[3])) # bishop 
        out -= 9 * (np.sum(self.last[4]) - np.sum(self.cur[4])) # queen
        return out 

    def _get_moves(self): 
        mv = [m.uci() for m in self.board.generate_legal_moves()] 
        out = np.zeros((len(mv), 6, 8, 8))

        for i in range(len(mv)): 
            self.board.push_uci(mv[i]) 
            self._get_board_rep(out[i]) 
            self.board.pop() 

        return mv, out 

    def _get_board_rep(self, out): 
        out[:] = 0
        lines = [line[::2] for line in str(self.board).split('\n')] 
        for y in range(8): 
            line = lines[y] 
            for x in range(8): 
                val = pieces[line[x]] 
                if val < 0: 
                    out[-val - 1, y, x] = -1
                elif val > 0: 
                    out[val - 1, y, x] = 1

if __name__ == "__main__":
    board = chess.Board()
    moves = []
    engine.set_position([]) 
    stats = {} 
    turn = 0
    while not game_done(stats): 
        turn += 1 

        # move = input("Move: ") 
        move = engine.get_best_move() 
        print([m.uci() for m in board.generate_legal_moves()]) 
        board.push_uci(move) 
        print(board.result()) 
        print(turn, 'White:', move) 

        moves.append(move) 
        engine.set_position(moves) 
        stats = engine.get_evaluation() 
        print(stats) 
        print(engine.get_board_visual()) 

        if game_mate(stats): 
            print("White wins") 
            break 

        move = engine.get_best_move() 
        print([m.uci() for m in board.generate_legal_moves()]) 
        board.push_uci(move) 
        print(board.result()) 
        print(turn, 'Black:', move) 
        moves.append(move) 
        engine.set_position(moves) 

        stats = engine.get_evaluation() 
        print(stats) 
        print(engine.get_board_visual()) 

        if game_mate(stats): 
            print("Black wins") 
            break 

    if not game_mate(stats): 
        print("Draw") 