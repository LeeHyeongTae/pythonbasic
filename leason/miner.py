from attr import dataclass


@dataclass
class Model:
    def __init__(self):
        context: str
        fname: str
        target: str

    @property
    def context(self) -> str: return self._context

    @context.setter
    def context(self, context): self.context = context

    @property
    def fname(self) -> str: return self._fname

    @fname.setter
    def fname(self, fname): self.fname = fname

    @property
    def target(self) -> str: return self._target

    @target.setter
    def target(self, target): self.target = target


class Service:
    def __init__(self):
        pass


class Controller:
    def __init__(self):
        pass
