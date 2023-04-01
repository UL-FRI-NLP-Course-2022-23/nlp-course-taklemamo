from .base import PipelineBlock
from urllib import request
import json

class SLONMTTranslateBlock(PipelineBlock):
    """
    Block for local slovene-english machine translation
    Needs a running instance of the api
    see: https://github.com/clarinsi/Slovene_NMT
    """

    def __init__(self, src_lang, tgt_lang, api_url="http://localhost:4000/api/translate"):
        """
        args:
            src_lang: str: source language, must be in SLONMTRequest.langs
            tgt_lang: str: target language, must be in SLONMTRequest.langs
                must not be the same as src_lang
            api_url: str: url for machine translation api, see: https://github.com/clarinsi/Slovene_NMT
        """
        
        super().__init__()
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.url = api_url

    def __call__(self, text_list):
        aug = [self.__translate(text) for text in text_list]
        return aug

    def __translate(self, text):
        req = SLONMTRequest(self.src_lang, self.tgt_lang, text, self.url)
        response = request.urlopen(req)
        response = json.loads(response.read().decode())
        return response["result"]


class SLONMTRequest(request.Request):
    
    langs = ["sl", "en"]

    def __init__(self, src_lang, tgt_lang, to_translate, api_url):
        assert src_lang in SLONMTRequest.langs and tgt_lang in SLONMTRequest.langs,\
            f"src_lang and tgt_lang should be in {SLONMTRequest.langs}"
        body = {
            "src_language": src_lang,
            "tgt_language": tgt_lang,
            "text": to_translate
        }
        super().__init__(
            url=api_url,
            data=bytes(json.dumps(body), "utf-8"),
            headers={
                "accept": "application/json",
                "Content-Type": "application/json"
            },
            method="POST"
        )
        