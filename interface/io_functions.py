import os, sys
import pickle as pkl

import numpy as np
from pyfiglet import Figlet

def span_window():
    print("-" * os.get_terminal_size().columns)
    return

def clear():
    os.system("clear")
    return

def get_fortune():
    gen = eval(open("files/fortune_file.dat",'r').read())

    return np.random.choice(gen)


def intro():
    clear()
    span_window()

    f = Figlet(font='slant')
    print(f.renderText('SPHINX'))

    print("Stellar Photometric Index Network eXplorer")
    span_window()
    span_window()
    print("Author: Devin D. Whitten")
    print("Institute: University of Notre Dame")
    print("Copyright Creative Commons")
    print("Contact: dwhitten@nd.edu")
    span_window()




## For loading trained networks
def load_network_state(filename):
    print("\t loading network state:  ", filename, '.pkl')
    net = pkl.load(open("net_pkl/" + filename + ".pkl", 'rb'))
    return net
