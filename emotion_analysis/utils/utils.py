import sys
import os


def blockPrint():
    sys.stdout = open(os.devnull, 'w')

    """
    Disable printing.
        """

def enablePrint():
    sys.stdout = sys.__stdout__

    """
    Enable printing.
        """
