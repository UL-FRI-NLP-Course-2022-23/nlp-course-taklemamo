class PipelineBlock():
    """
    Basic block for the preprocessing pipeline (Identity block)
    Working on principle: List[str] in List[str or None] out
    Both inputs and outputs must be of the same length
    Where some text from the input list is not processable, place None on output
    None handling will be done in pipeline, so in theory no input should be None
    """

    def __init__(self):
        pass

    def __call__(self, text_list):
        """
        Identity mapping
        args:
            text_list: List[str]: List of strings to preprocess
        returns:
            aug: List[str or None]: List of same length as input text_list of processed input texts, outputs can be None
        """
        aug = text_list
        return aug