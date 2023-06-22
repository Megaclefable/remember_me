# Python list concatenation
A = [3, 4, 5]
B = [1, 2]
current_a = 1

sorted = B
sorted += A[current_a:]

print(sorted)  # prints: [1, 2, 4, 5]
