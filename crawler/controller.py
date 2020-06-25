from crawler.model import Model
from crawler.service import Service

class Controller:
    def __init__(self):
        self.service = Service()
        self.model = Model()

    def bugs_music(self, url):
        self.model.url = url
        self.model.parser = parser
        self.model.path = path
        self.model.api = api