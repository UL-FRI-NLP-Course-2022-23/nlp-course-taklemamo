from bs4 import BeautifulSoup
import requests
import re
import string


class BibleScraper():

    def __init__(self, lang='sl'):
        self.lang = lang
        self.url = f"https://www.biblija.net/biblija.cgi?l={lang}"


    def get_book_refs(self):
        response = requests.get(self.url)
        soup = BeautifulSoup(response.content, 'html.parser')

        book_refs = []
        for book3 in soup.find_all(class_='book-3'):
            matches = re.search(r'\((.*?)\)', book3['href']).group(1)
            t = matches.replace("'", "").split(", ")[1]
            book_refs.append(t)

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



if __name__ == "__main__":

    bs = BibleScraper(lang='sl')
    book_refs = bs.get_book_refs()
    res = bs.get_chapt_refs(book_refs[0])


