import imp
import os


def load(name):
    pathname = os.path.join(os.path.dirname(__file__), name)
    return imp.load_source("", pathname)
