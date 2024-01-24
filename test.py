def ParallelSums(arr):
    if len(arr) % 2:
        return -1
    arr.sort()
    if len(arr) == 2:
        if arr[0] == arr[1]:
            return arr
        else:
            return -1

    i, j = 0, len(arr) - 1
    k, l = len(arr) // 2 - 1, len(arr) // 2
    osum, isum = 0, 0
    while i < k and l < j:
        osum = osum + arr[i] + arr[j]
        isum = isum + arr[k] + arr[l]
        i += 1
        j -= 1
        k -= 1
        l += 1
    print(osum, isum, [*arr[:i], *arr[j + 1 :], arr[i : j + 1]])
    if osum != isum:
        return -1
    return [*arr[:i], *arr[j + 1 :], arr[i : j + 1]]


print(ParallelSums([16, 22, 35, 8, 20, 1, 21, 11]))
exit()
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
