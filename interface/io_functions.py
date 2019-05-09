import os
from pyfiglet import Figlet

def span_window():
    print("-" * os.get_terminal_size().columns)
    return

def clear():
    os.system("clear")
    return


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
