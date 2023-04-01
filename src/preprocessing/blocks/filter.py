from .base import PipelineBlock

class FilterFunctions():
    
    def max_length(text, max_length):
        return text if len(text) <= max_length else None
    
    def min_length(text, min_length):
        return text if len(text) >= min_length else None
    
    def max_pass(text, max_length):
        return text if len(text) <= max_length else text[:max_length]
    

class FilterBlock(PipelineBlock):

    def __init__(self, filter_fcn, *filter_args, **filter_kwargs):
        super().__init__()
        self.filter = filter_fcn
        self.args = filter_args
        self.kwargs = filter_kwargs

    def __call__(self, text_list):
        aug = [
            self.filter(text, *self.args, **self.kwargs) 
            for text in text_list
        ]
        return aug