from crawler.model import Model
from crawler.service import Service

class Controller:
    def __init__(self):
        self._service = Service()

    def cralwer(self, url, parser, path, api):
        model = Model()
        model.url = url
        model.parser = parser
        model.path = path
        model.api = api
