from .base import PipelineBlock
import deepl

class DeepLTranslateBlock(PipelineBlock):
    """
    Block for translation with DeepL
    """
    
    def __init__(self, api_key, out_lang, in_lang=None):
        """
        args:
            api_key: str: api key from DeepL, see: https://www.deepl.com/account/summary
            out_lang: str: see deepl.Translator.translate_text()
            in_lang: str: see deepl.Translator.translate_text(),
                if None it will be automatically detected by DeepL
        """

        super().__init__()
        self.api_key = api_key
        self.out_lang = out_lang
        self.in_lang = in_lang
        self.translator = deepl.Translator(self.api_key)

    def __call__(self, text_list):
        translated_text_list = [
            self.translator.translate_text(text, target_lang=self.out_lang, source_lang=self.in_lang).text
            for text in text_list
            ]
        return translated_text_list