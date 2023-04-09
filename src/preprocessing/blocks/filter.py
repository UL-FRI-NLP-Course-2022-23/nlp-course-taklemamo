from .base import PipelineBlock


class FilterBlock(PipelineBlock):
    """
    Block for text filtering
    applies filtering function to input text
    """

    def __init__(self, filter_fcn, *filter_args, **filter_kwargs):
        """
        args:
            filter_fcn: filtering function that takes 
            str as first positional arg and filter_args, filter_kwargs
            see FilterFunctions class for details
        returns:
            None
        """

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
    
    
class FilterFunctions():
    """
    Basic text filtering functions
    args:
        text: str: text string to be filtered
        *args, **kwargs
    returns:
        filtered text: str or None: filtered input
            text or None
    """
    
    def max_chars(text, max_chars):
        return text if len(text) <= max_chars else None
    
    def min_chars(text, min_chars):
        return text if len(text) >= min_chars else None
    
    def max_pass(text, max_chars):
        return text if len(text) <= max_chars else text[:max_chars]
    
    def min_words(text, min_words):
        return text if len(text.split(" ")) >= min_words else None
    
    def max_words(text, max_words):
        return text if len(text.split(" ")) <= max_words else None
