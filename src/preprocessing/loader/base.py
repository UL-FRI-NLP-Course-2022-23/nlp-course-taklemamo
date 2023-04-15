import json

class Loader():

    @classmethod
    def from_checkpoint(cls, checkpoint_path):
        with open(checkpoint_path, "r") as f:
            state_dict = json.load(f)
        return cls(**state_dict)

    def __init__(self, file_queue, batch_size=1, text_list=[]):
        self.file_queue = file_queue
        self.batch_size = batch_size
        self.text_list = text_list

    def __fetch_single__(self):
        if not self.text_list:
            if not self.file_queue:
                raise StopIteration
            path = self.file_queue.pop(0)
            self.text_list = self.__load_file__(path)
        return self.text_list.pop(0)
    
    def __load_file__(self, path):
        raise NotImplementedError
    
    def __next__(self):
        return [self.__fetch_single__() for _ in range(self.batch_size)]

    def __iter__(self):
        return self
    
    def save_state(self, save_path):
        state_dict = dict(vars(self))
        with open(save_path, "w") as f:
            json.dump(state_dict, f, indent=4)
   