class InnerClass:

    def __init__(self, initVal):
        self.value = initVal

    def getValue(self):
        return self.value


class OuterClass:

    def __init__(self, innerClass):
        self.value = innerClass.getValue


if __name__ == "__main__":
    inner = InnerClass(1)
    outer = OuterClass(inner)

    print(f"Before change: Outer/inner vals: {outer.value()}/{inner.value}.")
    inner.value = 2
    print(f"After change: Outer/inner vals: {outer.value()}/{inner.value}.")
