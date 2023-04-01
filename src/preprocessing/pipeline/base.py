class PipeLine():

    def __init__(self, block_list):
        self.blocks = block_list

    def __call__(self, paragraphs):
        dataset = {
            "in": paragraphs,
            "out": []
        }

        for paragraph in paragraphs:
            aug = paragraph
            for block in self.block_list:
                aug = block([aug])
                if aug is None:
                    # pop the paragraph and continue
                    break
            else:
                # if the paragraph went through then it's ok
                dataset["out"].extend(aug)

        return dataset
    