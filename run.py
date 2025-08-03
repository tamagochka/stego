import sys

from numpy import set_printoptions, inf

from app import App

if __name__ == '__main__':
    
    set_printoptions(threshold=int(inf))
    set_printoptions(linewidth=int(inf))

    app = App()
    app.run()
    
    sys.exit()