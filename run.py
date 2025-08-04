import sys

from numpy import set_printoptions, iinfo, int64

from app import App

if __name__ == '__main__':
    
    set_printoptions(threshold=iinfo(int64).max)
    set_printoptions(linewidth=iinfo(int64).max)

    app = App()
    app.run()
    
    sys.exit()