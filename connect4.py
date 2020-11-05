import sys 

import numpy as np 

class Connect4: 

    def __init__(self, width, height): 
        self.width = width 
        self.height = height 
        self.board = np.zeros((height, width)) 
        self.turn = 1 

    def get_player(self): 
        return self.turn 

    def get_state(self): 
        return np.copy(self.board) 

    def get_actions(self): 
        actions = [] 
        for i in range(self.width): 
            if self.board[0,i] == 0: 
                actions.append(i) 
        return actions 

    def step(self, action, board=None): 
        real = False 
        if board is None: 
            board = self.board 
            real = True 

        if board[0, action] == 0: 
            for i in range(self.height-1, -1, -1): 
                if board[i, action] == 0: 
                    board[i, action] = self.turn 
                    if real: self.turn *= -1 
                    return board 
        else: 
            raise Exception("Cannot take action {}: move is illegal".format(action)) 

    def get_winner(self, board=None): 
        if board is None: 
            board = self.board 

        # horizontal 
        for y in range(self.height): 
            for x in range(self.width - 3): 
                val = np.sum(board[y,x:x+4])
                if np.abs(val) == 4: 
                    return int(np.sign(val)) 

        # vertical 
        for y in range(self.height - 3): 
            for x in range(self.width): 
                val = np.sum(board[y:y+4,x])
                if np.abs(val) == 4: 
                    return int(np.sign(val)) 

        # + diagonal 
        for y in range(self.height - 3): 
            for x in range(self.width - 3): 
                val = 0 
                for i in range(4): 
                    val += board[y+i,x+i] 
                if np.abs(val) == 4: 
                    return int(np.sign(val)) 

        # - diagonal 
        for y in range(self.height - 3): 
            for x in range(self.width - 3): 
                val = 0 
                for i in range(4): 
                    val += board[3+y-i,x+i] 
                if np.abs(val) == 4: 
                    return int(np.sign(val)) 

        if 0 not in board: 
            return 0
        else: 
            return None  

    def render(self): 
        s = '+' + '-'*(self.width*2-1) + '+\n'  
        for y in range(self.height): 
            s += '|'
            for x in range(self.width): 
                if x != 0: 
                    s += ' '

                if self.board[y,x] == -1: 
                    s += 'O' 
                elif self.board[y,x] == 1: 
                    s += 'X' 
                else: 
                    s += ' '
            s += '|\n' 
        s += '+' + '-'*(self.width*2-1) + '+\n '  
        for x in range(self.width): 
            if x != 0: 
                s += ' '
            s += '{}'.format(x) 
        s += ' \n' 
        print(s) 
    
if __name__ == "__main__": 
    env = Connect4(7, 6)  
    env.render() 

    while True: 
        actions = env.get_actions() 

        if len(actions) > 0: 
            while True: 
                try: 
                    a = input("Player {}'s turn, enter action {}: ".format(env.get_player(), actions))
                    if a == 'exit': 
                        sys.exit() 
                    a = int(a) 
                    if a in actions: 
                        env.step(a) 
                        break 
                except SystemExit as e: 
                    raise e 
                except: 
                    pass 

                print("Invalid action!") 

            env.render() 

            win = env.get_winner()
            if win != 0: 
                print("Winner is {}".format(win)) 
                sys.exit() 
