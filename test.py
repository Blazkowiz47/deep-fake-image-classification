a = [True] * 7


def f(i, rem, x):
    if rem == 0:
        ts = list(set(a))
        if len(ts) == 1 and ts[0] == False:
            print(7 - rem)
            return True
        return False
    x.append(i)
    a[i % 7] = not a[i % 7]
    a[(i + 1) % 7] = not a[(i + 1) % 7]
    a[(i + 2) % 7] = not a[(i + 2) % 7]
    ts = list(set(a))
    if len(ts) == 1 and ts[0] == False:
        print(7 - rem)
        return True

    for i in range(7):
        if f(i, rem - 1, x):
            return True
    return False


for i in range(7):
    a = [True] * 7
    ans = []
    print(f(i, 7, ans))


def sample(args):
    args.insert(args.index(args[-1]), 2)
    args.pop(0)


numbers = replica = [3, 4, 5, 6]
sample(replica)
print(replica)


def extra_end(inputString):
    return inputString[-2:] * 3


def doubleChar(inputString):
    return "".join([x + x for x in inputString])


print(doubleChar("The"))
