from bs4 import BeautifulSoup
import requests
import re
import string
import unicodedata

POETRY = ['Job', 'Ps', 'Prg', 'Prd', 'Vp', 'Žal']

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

        num_chapters = len(soup.find(class_='bar-chapters').find_all('a'))
        chapt_refs = []
        for n in range(num_chapters):
            chapt_refs.append(f"{book_ref} {n+1}")

        return chapt_refs


    def get_verses(self, chapt_ref):
        chapt_url = self.url + f"&m={chapt_ref}"
        response = requests.get(chapt_url)
        soup = BeautifulSoup(response.content, 'html.parser')

        table = soup.select('table:has(.tr-odd)')[-1]
        for row in table.find_all('tr'):
            for col in row.find_all('td', class_='text'):

                elements_to_keep = soup.find_all(class_=re.compile(r'^(p|v|q\d+)$'))
                for element in col.find_all():
                    if element not in elements_to_keep:
                        element.extract()

                text = unicodedata.normalize('NFKD', col.text).strip()


                print(text)
                print(10*'-')


if __name__ == "__main__":

    bs = BibleScraper()
    book_refs = bs.get_book_refs()
    print(book_refs)
    chapt_refs = bs.get_chapt_refs(book_refs[0])
    ps = bs.get_verses(chapt_refs[0])

    print(ps)

    # import pprint
    # pprint.pprint(ps, indent=4)






