class PipeLine():
    """
    Sequential text processing pipeline
    """

    def __init__(self, block_list):
        """
        args:
            block_list: List[PipelineBlock]
        returns:
            None
        """
        self.blocks = block_list

    def __call__(self, text_list):
        """
        args:
            text_list: List[str or None]
        returns:
            dataset: Dict("inputs": List[str], "outputs:" List[str])
                can be shorter in length than input since
                any text that returns None on path is discarded
        """

        dataset = {
            "inputs": [],
            "outputs": []
        }

        # this is pretty hacky
        for text in text_list:
            aug = [text]
            for block in self.blocks:
                if aug[0] is None:
                    break
                aug = block(aug)

            dataset["inputs"].append(text)
            dataset["outputs"].extend(aug)

        return dataset
    