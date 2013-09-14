def bubble(lst):
    unfinished = True
    while (unfinished):
        unfinished = False
        for i in range(0, len(lst) - 1):
            if lst[i] > lst[i + 1]:
                unfinished = True
                temp = lst[i]
                lst[i] = lst[i + 1]
                lst[i + 1] = temp
    return lst

def quicksort(lst):
    if len(lst) < 2:
        return lst
    pivot = lst[len(lst) / 2]
    less = [x for x in lst if x < pivot]
    greater = [x for x in lst if x > pivot]
    return quicksort(less) + [pivot] + quicksort(greater)

def merge(lst):
    if len(lst) < 2:
        return lst
    left = lst[:len(lst)/2]
    right = lst[len(lst)/2:]
    left = merge(left)
    right = merge(right)
    return comb(left, right)

def comb(left, right):
    result = []
    i, j = 0, 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    print result
    result += left[i:]
    result += right[j:]
    return result

    # print left, right
    # res = []
    # i, j = 0, 0
    # while i < len(left) and j < len(right):
    #   if left[i] < right[j]:
    #       res.append(left[i])
    #       i += 1
    #   else:
    #       res.append(right[j])
    #       j += 1
    # res += left[i:]
 #    res += right[j:]

    # print res
    # return res

numbers = [i * -1 for i in range(0, 10)]
# print numbers
print merge(numbers)