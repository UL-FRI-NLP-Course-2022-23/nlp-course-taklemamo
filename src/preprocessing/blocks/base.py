class PipelineBlock():

    def __init__(self):
        pass

    def __call__(self, text_list):
        """
        Basic block for the preprocessing pipeline (Identity block)
        args:
            text_list: List[str]: List of strings to preprocess (not None) -> none handling at PipeLine level
        returns:
            aug: List[str or None]: List of same length as input text_list of processed input texts, outputs can be None
        """
        aug = text_list
        return aug