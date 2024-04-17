#!/usr/bin/env python3
from tkinter import *
import sys
sys.path.insert(1, 'hidden')
from front_end import *

def main():
    root = Tk()
    root.iconbitmap('hidden/favicon.ico')
    root.title('Visual Network Builder')
    resolution = "{}x{}".format(W_PIXELS, H_PIXELS)
    root.geometry(resolution)
    active_window = Window(root)
    active_window.setup_elements()
    root.mainloop()

if __name__ == "__main__":
    main()