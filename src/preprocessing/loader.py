import json

class Loader():

    @classmethod
    def from_checkpoint(cls, checkpoint_path):
        with open(checkpoint_path, "r") as f:
            state_dict = json.load(f)
        return cls(
            state_dict["queue"],
            state_dict["batch_size"],
            state_dict["text_list"]
            )

    def __init__(self, fpaths, batch_size=1, text_list=[]):
        self.queue = fpaths
        self.batch_size = batch_size
        self.text_list = text_list

    def __fetch_single(self):
        if not self.text_list:
            if not self.queue:
                raise StopIteration
            path = self.queue.pop(0)
            with open(path, "r") as f:
                # text chunks need to be split into individual lines
                self.text_list = f.read().splitlines()
        return self.text_list.pop(0)
    
    def __next__(self):
        return [self.__fetch_single() for _ in range(self.batch_size)]

    def __iter__(self):
        return self
    
    def save_state(self, save_path):
        state_dict = {
            "queue": self.queue,
            "batch_size": self.batch_size,
            "text_list": self.text_list,
        }
        with open(save_path, "w") as f:
            json.dump(state_dict, f, indent=4)
