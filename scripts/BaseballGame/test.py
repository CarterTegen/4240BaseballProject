class Test:

    def __init__(self):
        self.a = Car()
        print(self.a.b)


class Car:

    def __init__(self):
        self.b = 3

test = Test()