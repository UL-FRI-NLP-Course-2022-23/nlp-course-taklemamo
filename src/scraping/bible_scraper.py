from bs4 import BeautifulSoup
import requests
import re
import string
import unicodedata

POETRY = ['Job', 'Ps', 'Prg', 'Prd', 'Vp', 'Žal']
SKIP = ['2 Jn', '3 Jn']

class BibleScraper():

    def __init__(self):
        self.url = "https://www.biblija.net/biblija.cgi?m=4+Mz+7&id13=1&id14=1&pos=0&set=6&l=sl&idp0=14&idp1=15"

    def get_book_refs(self):
        response = requests.get(self.url)
        soup = BeautifulSoup(response.content, 'html.parser')

        book_refs = []
        for book3 in soup.find_all(class_='book-3'):
            matches = re.search(r'\((.*?)\)', book3['href']).group(1)
            t = matches.replace("'", "").split(", ")[1]
            book_refs.append(t)

        book_refs = [bk for bk in book_refs if bk not in POETRY]
        return book_refs


    def get_chapt_refs(self, book_ref):
        book_url = self.url + f"&m={book_ref}"
        response = requests.get(book_url)
        soup = BeautifulSoup(response.content, 'html.parser')

        chapts = soup.find(class_='bar-chapters')
        if chapts is None:
            return []
        num_chapters = len(chapts.find_all('a'))
        chapt_refs = []
        for n in range(num_chapters):
            chapt_refs.append(f"{n+1}")

        return chapt_refs


    def get_verses(self, book_ref, chapt_ref):
        chapt_url = self.url + f"&m={book_ref} {chapt_ref}"
        response = requests.get(chapt_url)
        soup = BeautifulSoup(response.content, 'html.parser')

        table = soup.select('table:has(.tr-odd)')[-1]
        rows = table.find_all('tr')
        rows[-1].decompose()
        verses = {}
        for row in table.find_all('tr'):
            vnum = None
            sl_verse = ''
            en_verse = ''
            for lang, col in enumerate(row.find_all('td', class_='text')):

                elements_to_keep = soup.find_all(class_=re.compile(r'^(p|mi|nd|m|v|qt|q\d*)$'))
                elements_to_keep += soup.find_all('font')
                for element in col.find_all():
                    if element not in elements_to_keep:
                        element.extract()

                text = unicodedata.normalize('NFKD', col.text).strip()
                if not text[0].isdigit():
                    return None

                try:
                    vnum, verse = text.split(' ', 1)
                except ValueError:
                    continue

                if not vnum.isnumeric():
                    continue

                vnum = int(vnum)

                if lang:
                    en_verse = verse
                else:
                    sl_verse = verse

            if en_verse == '' or sl_verse == '':
                return None

            if vnum is not None:
                verses[vnum] = {'sl': sl_verse, 'en': en_verse}

        return verses


if __name__ == "__main__":

    import pprint
    import json

    bs = BibleScraper()

    bible = {}
    for bk in bs.get_book_refs():
        book = {}
        if bk in SKIP:
            continue
        for ch in bs.get_chapt_refs(bk):
            ps = bs.get_verses(bk, ch)
            print(bk, ch)
            book[ch] = ps
        bible[bk] = book

    # pprint.pprint(bible, indent=4)

    with open("../../data/bible_EN_SL.json", "w") as outfile:
        json.dump(bible, outfile, indent=4)






