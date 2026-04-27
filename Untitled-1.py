name = input()
i = 0
while "Александра" not in str(name):
    while "Левон" not in str(name):
        i += 1
        name = input()

print(i)
