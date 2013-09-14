dic = {}

def combination(n, k):
    if n + k == n or n == k:
        return 1
    if (n, k) in dic:
        return dic[(n, k)]
    else:
        result = combination(n - 1, k) + combination(n - 1, k - 1)
        dic[(n, k)] = result
        return result

t = raw_input()
lines = []
for i in range(int(t)):
    lines.append(raw_input())
for line in lines:
    curr = line.split(" ")
    print combination(int(curr[0]), int(curr[1])) % 142857