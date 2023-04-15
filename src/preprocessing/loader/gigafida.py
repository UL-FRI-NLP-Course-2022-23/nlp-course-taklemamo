from .base import Loader

class GIGAFidaLoader(Loader):

    def __load_file__(self, path):
        with open(path, "r") as f:
            # text chunks need to be split into individual lines
            gigatext = f.read()
        return [para.replace("\n", "") for para in gigatext.split("\n\n")]
    