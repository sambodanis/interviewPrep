import sys
import random

class Node:
    def __init__(self, data, next):
        self.data = data
        self.next = next

class Stack:
    def __init__(self):
        self.top = None
        self.count = 0
    
    def push(self, item):
        if self.top == None:
            self.top = Node(item, None)
        else:
            new = Node(item, self.top)
            self.top = new
        self.count += 1

    def pop(self):
        if self.top != None:
            result = self.top.data
            self.top = self.top.next
            self.count -= 1
            return result
        else:
            return None

    def peek(self):
        if self.count == 0:
            return None
        return self.top.data

    def size(self):
        return self.count

    def isEmpty(self):
        return self.count == 0

class Queue:
    def __init__(self):
        self.first = None
        self.last = None
        self.count = 0

    def enqueue(self, item):
        if self.first == None:
            self.last = Node(item, None)
            self.first = self.last
        else:
            self.last.next = Node(item, None)
            self.last = self.last.next
        self.count += 1

    def dequeue(self):
        if self.first != None:
            result = self.first.data
            self.first = self.first.next
            self.count -= 1
            return result
        return None

    def peek(self):
        if self.count == 0:
            return None
        return self.first.data

    def size(self):
        return self.count

    def isEmpty(self):
        return self.count == 0


class MinStack:
    def __init__(self):
        self.mainStack = Stack()
        self.minStack = Stack()

    def push(self, item):
        if self.mainStack.isEmpty():
            self.mainStack.push(item)
            self.minStack.push(item)
        else:
            self.mainStack.push(item)
            if item < self.minStack.peek():
                self.minStack.push(item)

    def pop(self):
        if self.mainStack.isEmpty():
            return None
        poppedItem = self.mainStack.pop()
        if poppedItem == self.minStack.peek():
            self.minStack.pop()
        return poppedItem

    def min(self):
        return self.minStack.peek()

    def size(self):
        return self.mainStack.size()

    def isEmpty(self):
        return self.mainStack.isEmpty()

class Tree:
    def __init__(self, value):
        self.data = value
        self.left = None
        self.right = None

    def add(self, value):
        if value < self.data:
            if self.left == None:
                self.left = Tree(value)
            else:
                self.left.add(value)
        if value > self.data:
            if self.right == None:
                self.right = Tree(value)
            else:
                self.right.add(value)
    
    def preorderPrint(self):
        print self.data
        if self.left != None:
            self.left.preorderPrint()
        if self.right != None:
            self.right.preorderPrint()

    def preorderArray(self):
        result = [self.data]
        left, right = [], []
        if self.left != None:
            left += self.left.preorderArray()
        if self.right != None:
            right += self.right.preorderArray()
        return result + left + right

    def equals(self, b):
        if self.data != b.data:
            return False
        left_eq, right_eq = True, True
        if self.left != None and b.left != None:
            left_eq = self.left.equals(b.left)
        if self.right != None and b.left != None:
            right_eq = self.right.equals(b.right)
        return left_eq and right_eq

def printList(head):
    vals = []
    while head != None:
        vals.append(head.data)
        head = head.next
    print vals

def printStack(st):
    while not st.isEmpty():
        print st.pop()

def isAllUnique(str1):
    letters = [0] * 256
    for char in str1:
        currIndex = ord(char)
        if letters[currIndex] != 0:
            return False
        letters[currIndex] += 1
    return True

def reverse(str1):
    # result = ""
    for i in range(len(str1)):
        # result += str1[len(str1) - 1 - i]
        if i >= len(str1) / 2:
            return str1
        temp = str1[i]
        str1[i] = str1[len(str1) - 1 - i]
        str1[len(str1) - 1 - i] = temp
    # return result

def makeUnique(str1):
    letters = [False] * 256
    result = ""
    for ch in str1:
        if not letters[ord(ch)]:
            result += ch
            letters[ord(ch)] = True
    return result

def isAnagram(str1, str2):
    return qsort(str1) == qsort(str2)

def qsort(lst):
    if len(lst) <= 1:
        return lst
    pivot = lst[len(lst) / 2]
    less = []
    pivots = []
    more = []
    for elem in lst:
        if elem < pivot:
            less.append(elem)
        elif elem > pivot:
            more.append(elem)
        else:
            pivots.append(elem)
    # less = [x for x in lst if x < pivot]
    # more = [x for x in lst if x > pivot]
    # pivots = [x for x in lst if x == pivot]
    return qsort(less) + pivots + qsort(more)

def spaceToM20(str1):
    result = ""
    for ch in str1:
        if ch == " ":
            result += "%20"
        else:
            result += ch
    return result

def rotate90(mat):
    for i in range(len(mat) / 2):
        first = i
        last = len(mat) - 1 - i
        j = first
        while j < last:
            offset = j - first
            top = mat[first][j]
            mat[first][j] = mat[last - offset][first]
            mat[last - offset][first] = mat[last][last - offset]
            mat[last][last - offset] = mat[j][last]
            mat[j][last] = top
            j += 1

    return mat

def zeroize(matrix):
    rows = [0] * len(matrix)
    cols = [0] * len(matrix[0])
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] == 0:
                rows[i] = 1
                cols[j] = 1
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if rows[i] == 1 or cols[j] == 1:
                matrix[i][j] = 0
    return matrix

def removeDuplicatesList(head):
    front = head
    prev = None
    copies = set()
    while head != None:
        if head.data in copies:
            prev.next = head.next
        else:
            copies.add(head.data)
            prev = head
        head = head.next
    return front

def nThLastNode(head, col):
    if col < 1:
        return -1
    p1 = head
    p2 = head
    for i in range(col):
        if p2 == None:
            return -1
        p2 = p2.next
    while True:
        if p2 == None:
            return p1.data
        else:
            p2 = p2.next
            p1 = p1.next

def removeNode(head):
    while head.next != None:
        head.data = head.next.data
        head = head.next
    head = None

def addLists(aHead, bHead, carry):
    if aHead == None and bHead == None:
        return None
    result = Node(0, None)
    value = carry
    if aHead != None:
        value += aHead.data
        aHead = aHead.next
    if bHead != None:
        value += bHead.data
        bHead = bHead.next
    result.data = value % 10
    if value > 10:
        carry = 1
    next = addLists(aHead, bHead, carry)
    result.next = next
    return result

# iffy, seem to have some sort of mental block against doing this question 
def towersOfHanoi(col, source, helper, target):
    # for i, st in enumerate(hanoiArray):
    #   print "col: " + str(col) + ", Tower " + str(i) + " size is " + str(st.size())
    # print hanoiArray[0].size(), hanoiArray[1].size(), hanoiArray[2].size()
    print source.size(), helper.size(), target.size()
    if col > 0:
        towersOfHanoi(col - 1, source, target, helper) # hanoiArray, [sth[0], sth[2], sth[1]])
        # if not hanoiArray[0].isEmpty():
        if not source.isEmpty():
            # hanoiArray[sth[1]].push(hanoiArray[sth[0]].pop())
            helper.push(source.pop())
        towersOfHanoi(col - 1, helper, source, target) # hanoiArray, [sth[1], sth[0], sth[2]])

def sortStack(st):
    st2 = Stack()
    while not st.isEmpty():
        temp = st.pop()
        while not st2.isEmpty() and st2.peek() > temp:
            st.push(st2.pop())
        st2.push(temp)
    return st2

def treeIsBalanced(tr):
    depths = treeIsBalancedR(tr, 0)
    return max(depths) - min(depths) < 2

def treeIsBalancedR(tr, depth):
    if tr.left == None and tr.right == None:
        return [depth]
    else:
        result = []
        if tr.left != None:
            result += treeIsBalancedR(tr.left, depth + 1)
        if tr.right != None:
            result += treeIsBalancedR(tr.right, depth + 1)
        return result

def minHeightTree(arr):
    if len(arr) == 0:
        return None
    mid = len(arr) / 2
    root = Tree(arr[mid])
    root.left = minHeightTree(arr[:mid])
    root.right = minHeightTree(arr[mid + 1:])
    return root

def makeDepthLists(tr):
    qu = Queue()
    result = []
    depth = 0
    makeDepthListsR(tr, depth, result)
    return result

def makeDepthListsR(root, depth, resultList):
    if root == None:
        return
    if depth >= len(resultList):
        resultList.append(Node(root.data, None))
    else:
        currNode = resultList[depth]
        while currNode.next != None:
            currNode = currNode.next
        currNode.next = Node(root.data, None)
    makeDepthListsR(root.left, depth + 1, resultList)
    makeDepthListsR(root.right, depth + 1, resultList)

def printBinary(number):
    for i in range(32):
        print((number >> i) & 1),
    print "" 

def mergeBitsBetweenIndices(col, M, i, j):
    for index in range(i, j):
        col |= ((M >> index) & 1) << index
    return col

def getBit(num, i):
    return (num >> i) & 1

def setBit(num, i):
    return num | (1 << i)

def toggleBit(num, i):
    return num ^ (1 << i)

def bitCountConvert(col, M):
    resultCount = 0
    for i in range(32):
        if getBit(col, i) != getBit(M, i):
            resultCount += 1
    return resultCount

def swapEvenOdd(col):
    odd = (col & 0xAAAAAAAA) >> 1
    even = (col & 0x55555555) << 1
    return odd | even

def findMissingInArray(arr):
    first = True
    result = 0
    for i in arr:
        if first:
            result = i
            first = False
            continue
        for j in range(5):
            if getBit(result, j) != getBit(i, j):
                result = toggleBit(result, j)
    return result

def fibonacci(col):
    if col == 0:
        return 0
    elif col == 1:
        return 1
    else:
        return fibonacci(col-1) + fibonacci(col-2)  

def numRoutes(testGrid, i, j):
    if i == len(testGrid) - 1 and j == len(testGrid[0]) - 1:
        return 1
    else:
        sumRoutes = 1
        if i != len(testGrid) - 1:
            sumRoutes += numRoutes(testGrid, i+1, j)
        if j != len(testGrid[0]) - 1:
            sumRoutes += numRoutes(testGrid, i, j+1)
        return sumRoutes

def allSubsets(sets, index):
    result = [[]]
    if len(sets) != index:
        result = allSubsets(sets, index + 1)
        curr = sets[index]
        newSubSets = [[]]
        for elem in result:
            newSub = []
            newSub += elem
            newSub.append(curr)
            newSubSets.append(newSub)
        result += newSubSets
    return result

def permutations(word):
    if len(word) == 0:
        return [""]
    curr = word[0]
    subWords = permutations(word[1:])
    perms = []
    for subWord in subWords:
        for i in range(len(subWord) + 1):
            perms.append(subWord[:i] + curr + subWord[i:])
    return perms

def nParens(col, l, r, st):
    if l < 0 or l > r:
        return
    if l == 0 and r == 0:
        print "".join(st)
    else:
        if l > 0:
            st[col] = "("
            nParens(col + 1, l - 1, r, st)
        if r > l:
            st[col] = ")"
            nParens(col + 1, l, r - 1, st)

def coinValuePerms(v, coins, coinTypes):
    if sum(coins) > v:
        return
    if sum(coins) == v:
        print coins
    else:
        for coin in coinTypes:
            coins.append(coin)
            coinValuePerms(v, coins, coinTypes)
            if len(coins) > 0:
                coins.pop()

def mergeBInA(a, b):
    ac = len(a) - 1
    bc = len(b) - 1
    kc = len(a) + len(b) - 1
    result = [0] * (kc + 1)
    while ac >= 0 and bc >= 0:
        print (a, b, result)
        if a[ac] > b[bc]:
            result[kc] = a[ac]
            ac -= 1
            kc -= 1
        else:
            result[kc] = b[bc]
            bc -= 1
            kc -= 1
    while bc >= 0:
        result[kc] = b[bc]
        bc -= 1
        kc -= 1
    return result

def shiftedBSearch(a, start, end, x):
    while start < end:
        m = (start + end) / 2
        if x == a[m]:
            return m
        elif a[start] <= a[m]:
            if x > a[m]:
                start = m + 1
            elif x >= a[start]:
                end = m - 1
            else:
                start = m + 1
        elif x < a[m]:
            end = m - 1
        elif x <= a[end]:
            start = m + 1
        else:
            end = m - 1
    return -1

def largestContSum(arr):
    maxSum = -sys.maxint - 1
    maxSumIndex = 0
    for idx, elem in enumerate(arr):
        currSum = elem
        currMax = currSum
        for val in arr[idx+1:]:
            currSum += val
            if currSum > currMax:
                currMax = currSum
        if currMax > maxSum:
            maxSum = currMax
            maxSumIndex = idx
    return maxSum, maxSumIndex

def largestContSumLinear(arr):
    maxSum = 0
    currSum = 0
    for i, v in enumerate(arr):
        currSum += v
        if currSum > maxSum:
            maxSum = currSum
        elif currSum < 0:
            currSum = 0
    return maxSum

def twoSum(arr, sumv):
    start = 0
    end = len(arr) - 1
    while start < end:
        currSum = arr[start] + arr[end]
        if currSum == sumv:
            # Increment one of them
            print arr[start], arr[end]
            start += 1
            end -= 1
        elif currSum > sumv:
            end -= 1
        else:
            start += 1


  # mat = [[0, 0, 0, 0, 1, 0], 
    #      [0, 1, 0, 0, 1, 0], 
    #      [0, 0, 0, 0, 1, 0],
    #      [0, 0, 0, 0, 0, 1], 
    #      [0, 0, 0, 0, 1, 0], 
    #      [0, 0, 0, 0, 1, 1]]

# def rectZeros(mat):
#     maxSize = 0
#     print len(mat), len(mat[0])
#     for row in range(len(mat)):
#         for col in range(len(mat[row])):
#             currSize = 0
#             curr_row = row
#             curr_col = col
#             inc_r = True
#             inc_c = True
#             print "mat[", row, "][", col, "] = ", mat[row][col]
#             while True:
#                 if not inc_r and not inc_c:
#                     break
#                 if inc_r:
#                     for col_t in range(col, curr_col):
#                         if curr_row + 1 >= len(mat) or mat[curr_row + 1][col_t] == 1:
#                             inc_r = False
#                             break
#                     if inc_r:
#                         curr_row += 1
#                 if inc_c:
#                     for row_t in range(row, curr_row):
#                         if curr_col + 1 >= len(mat[0]) or mat[row_t][curr_col + 1] == 1:
#                             inc_c = False
#                             break
#                     if inc_c:
#                         curr_col += 1
#                 currSize = (curr_row - row) * (curr_col - col)
#                 # if inc_r:
#                 #     curr_row += 1
#                 #     for col_t in range(col, curr_col):
#                 #         print "row, col_t", curr_row, col_t
#                 #         if (curr_row == len(mat) or col_t == len(mat[0]) - 1 
#                 #             or mat[curr_row][col_t] == 1):
#                 #             curr_row -= 1
#                 #             inc_r = False
#                 # if inc_c:
#                 #     curr_col += 1
#                 #     for row_t in range(row, curr_row):
#                 #         print "col, row_t", row_t, curr_col
#                 #         if (curr_col == len(mat[0]) or 
#                 #             row_t == len(mat) - 1 or mat[row_t][curr_col] == 1):
#                 #             curr_col -= 1
#                 #             inc_c = False
#                 # currSize = (curr_row - row) * (curr_col - col)
#                 # print currSize
#             if currSize > maxSize:
#                 maxSize = currSize
#     return maxSize

def randomRange(low, high):
    return random.randrange(low, high, 1)

def buildRandomTree(num_elements):
    result = None
    for val in range(num_elements):
        new_value = randomRange(-100, 100)
        if result == None:
            result = Tree(new_value)
        else:
            result.add(new_value)
    return result

def routeSumValue(tr, curr_sum, target_sum, current_path):
    if curr_sum == target_sum:
        print current_path
    if tr.left != None:
        routeSumValue(tr.left, curr_sum + tr.data, target_sum, current_path + [tr.data])
    if tr.right != None:
        routeSumValue(tr.right, curr_sum + tr.data, target_sum, current_path + [tr.data])

def reconstructTree(tree_arr):
    if len(tree_arr) == 0:
        return None
    curr_value = tree_arr[0]
    result = Tree(curr_value)
    if len(tree_arr) == 1:
        return result
    mid_idx = len(tree_arr) / 2
    result.left = reconstructTree(tree_arr[1:mid_idx])
    result.right = reconstructTree(tree_arr[mid_idx:])
    return result




def main():
    #----- 1.x
    # str1 = "abcdefg"
    # print isAllUnique(str1) # 1.1
    # print "".join(reverse(list("sentence")))
    # print reverse(list(str1)) # 1.2 
    # print makeUnique("abcdefgabcdefg") # 1.3
    # print isAnagram("abcdb", "abbdc") # 1.4
    # print spaceToM20("a b  c def g ") # 1.5
    # print rotate90([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) # 1.6
    # print zeroize([[0, 2, 3], [4, 5, 6], [7, 8, 9]]) # 1.7
    
    #----- 2.x
    # d = Node(3, None)
    # c = Node(2, d)
    # b = Node(2, c)
    # a = Node(4, b)
    # d1 = Node(7, None)
    # c1 = Node(3, d1)
    # b1 = Node(2, c1)
    # a1 = Node(7, b1)
    # printList(removeDuplicatesList(a)) # 2.1
    # print nThLastNode(a, 1) # 2.2
    # removeNode(b) # 2.3
    # printList(a)
    # printList(addLists(a, a1, 0)) # 2.4

    #----- 3.x
    # st = Stack()
    # st.push(5)
    # st.push(2)
    # st.push(3)
    # st.push(7)
    # st.push(1)
    # print st.pop()
    # print st.pop()
    # qu = Queue()
    # qu.enqueue(1)
    # qu.enqueue(2)
    # print qu.dequeue()
    # print qu.dequeue()
    # ms = MinStack() # 3.2
    # ms.push(1)
    # ms.push(2)
    # ms.push(3)
    # ms.push(-1)
    # ms.push(-2)
    # while not ms.isEmpty():
    #   print ms.min()
    #   ms.pop()
    # hanoiArray = [Stack()] * 3
    # hanoiArray[0].push(3)
    # hanoiArray[0].push(2)
    # hanoiArray[0].push(1)
    # st1 = Stack()
    # st1.push(3)
    # st1.push(2)
    # st1.push(1)
    # st2 = Stack()
    # st3 = Stack()
    # towersOfHanoi(st1.size(), st1, st2, st3)
    # print st1.size(), st2.size(), st3.size()
    # towersOfHanoi(hanoiArray[0].size(), hanoiArray, [0, 1, 2]) # 3.4
    # for i, stH in enumerate(hanoiArray):
    #   print "Tower %d", i, printStack(stH)
    # printStack(sortStack(st)) # 3.6
    # tr = Tree(15)
    # tr.add(5)
    # tr.add(3)
    # tr.add(12)
    # tr.add(10)
    # tr.add(6)
    # tr.add(7)
    # tr.add(13)
    # tr.add(16)
    # tr.add(20)
    # tr.add(18)
    # tr.add(23)
    # print tr.preorderArray()

    # tr.preorderPrint()
    # print treeIsBalanced(tr) # 4.1
    # arrayToBuildTreeFrom = [1, 2, 3, 4, 5, 6, 7, 8]
    # minHeightTree(arrayToBuildTreeFrom).preorderPrint() # 4.3
    
    # depthListArray = makeDepthLists(tr) # 4.4
    # for i, iNode in enumerate(depthListArray):
    #   while iNode != None:
    #       print i, iNode.data
    #       iNode = iNode.next
    # printBinary(5)
    # col = 31 # 1024
    # M = 14 # 75
    # printBinary(col)
    # printBinary(M)
    # printBinary(mergeBitsBetweenIndices(col, M, 0, 6)) # 5.1
    # print bitCountConvert(col, M) # 5.5 
    # printBinary(swapEvenOdd(91238457)) # 5.6
    # arrayWithMissing = []
    # shiftLen = 64 - 5
    # print (sys.maxint + 1) >> shiftLen
    # for i in range((sys.maxint + 1) >> shiftLen):
    #   if i != 31:
    #       arrayWithMissing.append(i)
    # print findMissingInArray(arrayWithMissing) # 5.7
    # print fibonacci(10) # 8.1 
    # testGrid = [[True, True, True, True],[True, True, True, True],[True, True, True, True],[True, True, True, True]]
    # testGrid = [[0] * 20] * 20
    # testGrid = [[True, True, True],[True, True, True],[True, True, True]]
    # print numRoutes(testGrid, 0, 0) # 8.2
    # print allSubsets([1, 2, 3], 0) # 8.3
    # print permutations("abcd") # 8.4
    # col = 100
    # nParens(0, col, col, [""]* 2 * col) # 8.5
    # coinTypes = [25, 10, 5, 1]
    # coinValuePerms(26, [], coinTypes)
    # a = [1, 3, 7, 8, 9, 12]
    # b = [2, 4, 5, 6, 10, 14]
    # print mergeBInA(a, b) # 9.1
    # shiftedSorted = [15, 16, 19, 20, 25, 1, 3, 4, 5, 7, 10, 14]
    # print shiftedBSearch(shiftedSorted, 0, len(shiftedSorted) - 1, 5) # 9.3
    # arr = [2, -8, 3, -2, 4, -10]
    # print largestContSum(arr) # 19.7 O(col^2)
    # print largestContSumLinear(arr) # 19.7 O(col)
    # test = [-2, -1, 0, 3, 5, 6, 7, 9, 13, 14]
    # twoSum(test, 12) # 19.11
    # mat = [[0, 0, 0, 0, 1, 0], 
    #        [0, 1, 0, 0, 1, 0], 
    #        [0, 0, 0, 0, 1, 0],
    #        [0, 0, 0, 0, 0, 1],
    #        [0, 0, 0, 0, 0, 1],  
    #        [0, 0, 0, 0, 1, 1]]
    # print rectZeros(mat)
    tree_r = buildRandomTree(10)
    # print tree_r.preorderArray()
    # tree_r.preorderPrint()
    # routeSumValue(tree_r, 0, 50, [])
    print tree_r.preorderArray()
    print reconstructTree(tree_r.preorderArray()).preorderArray()
    print tree_r.equals(reconstructTree(tree_r.preorderArray()))



if __name__ == '__main__':
    main()





