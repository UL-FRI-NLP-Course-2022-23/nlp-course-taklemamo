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
            dataset: Dict("inputs": List[str], "targets:" List[str])
                can be shorter in length than input since
                any text that returns None on path is discarded
        """

        dataset = {
            "inputs": [],
            "targets": []
        }

        # this is pretty hacky - no need to iterate through text list since loader already loads in batches
        for text in text_list:
            aug = [text]
            for block in self.blocks:
                aug = block(aug)
                if aug[0] is None:
                    # don't add the paragraph to dataset
                    # if some block returned None for it
                    break
            else:
                # if the paragraph went through then it's ok
                dataset["inputs"].append(text)
                dataset["targets"].extend(aug)

        return dataset
    