import time

mulDict = {}

num1 = 9**3000000
num2 = 9**3000000

mulDict[(num1, num2)] = num1 * num2

mulTimeStart = time.time()
mulBF = num1 * num2
mulTimeEnd = time.time()
print "Raw multiplication took: " + str(mulTimeEnd - mulTimeStart)

a = time.time()
hashMulLookup = mulDict[(num1, num2)]
b = time.time()

print "Hashmap lookup took: " + str(b - a)
