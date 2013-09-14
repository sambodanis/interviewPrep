filename = 'most1000commonWords.txt'
lines = [line.rstrip('\n') for line in open(filename)]
freq = [0] * 26

for line in lines:
	for ch in line:
		freq[ord(ch) - ord('A')] += 1

fTags = [(0,0)] * 26

for i in range(0, len(freq)):
	fTags[i] = (i, freq[i])

fTags.sort(key=lambda tup: tup[1])

count = 0
while count < 10:
	charV = fTags[len(fTags) - count - 1][0]
	chFreq = fTags[len(fTags) - count - 1][1]
	print chr(charV + ord('A')) + ' ' + str(chFreq)
	count += 1