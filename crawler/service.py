# from

class Service:
    def __init__(self):
        pass

    def bugs_music(self, payload):
        soup = BeautifulSoup(urlopen(payload.url), payload.parser)
        n_artist = 0
        n_title = 0
        for i in soup.find_all(name='p', attr=({'class':'title'})):
            n_title += 1
            print(str(n_title)+'위')
            print('노래제목'+)