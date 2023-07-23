from copy import deepcopy

class A():
    def __init__(self) -> None:
        self.x = [1,2,3]
        self.y = [4,5,6]

class B():
    def __init__(self) -> None:
        self.a = A()

class C():
    def __init__(self) -> None:
        self.b = B()

a = A()
b = B()
b.a = a
c1 = C()
c1.b = b
c2 = deepcopy(c1)
a.x[0] = 10
print(c1.b.a.x)
print(c2.b.a.x)