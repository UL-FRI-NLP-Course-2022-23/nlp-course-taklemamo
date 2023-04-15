from .base import Loader
import json

class BibleLoader(Loader):

    def __init__(self, file_queue, lang, batch_size=1, text_list=[]):
        super().__init__(file_queue, batch_size, text_list)
        self.lang = lang

    def __load_file__(self, path):
        with open(path, "r") as f: 
            bible = json.load(f)
        # python moment incoming
        return [para[self.lang] 
                for book in bible.values() 
                for part in book.values() if part is not None 
                for para in part.values()]
 