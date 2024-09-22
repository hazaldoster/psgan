# -*- coding: utf-8 -*-
import sys
import os
from time import time

def create_dir(folder):
    """
    Creates a folder if it doesn't already exist.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)

class TimePrint:
    """
    Simple convenience class to measure and print how long it takes between successive calls.
    
    Usage:
        tp = TimePrint("Start") -- prints "Start"
        <do some stuff here>
        tp.p("End") -- prints "End (took ?s)", where ? is the time passed since the "Start" call.
    """
    
    def __init__(self, text):
        """
        Initializes the TimePrint object, prints the initial message, and stores the current time.
        """
        self.t_last = time()  # Store the time of the initial call
        self.p(text)

    def p(self, text):
        """
        Prints the provided text along with the time taken since the last call.
        """
        t = time()  # Current time
        elapsed_time = t - self.t_last
        print(f"{text}", end='')  # Print text without newline
        if self.t_last is not None:
            print(f" (took {elapsed_time:.2f}s)")
        else:
            print()
        self.t_last = t  # Update the last recorded time
        sys.stdout.flush()  # Ensure the output is flushed immediately

if __name__ == "__main__":
    print("This is just a library.")