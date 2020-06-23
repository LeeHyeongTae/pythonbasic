class Service:
    def __init__(self, payload):
        self._num1 = payload.num1
        self._num2 = payload.num2

    # 장점 : 후행적으로 주입을 하므로 미리 만들어 놓을 필요가 없다.

    def add(self):
        return self._num1 + self._num2

    def minus(self):
        return self._num1 - self._num2

    def multi(self):
        return self._num1 * self._num2

    def divide(self):
        return self._num1 / self._num2
