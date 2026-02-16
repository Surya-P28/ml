import csv

a = []

# Read CSV file
with open('enjoysport.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        a.append(row)

print("\nThe training instances are:\n", a)
print("\nThe total number of training instances are:", len(a))

# Number of attributes (excluding target)
num_attribute = len(a[0]) - 1

print("\nThe initial hypothesis is:")

# Initialize hypothesis with '0'
hypothesis = ['0'] * num_attribute
print(hypothesis)

# Apply Find-S algorithm
for i in range(len(a)):
    if a[i][num_attribute] == "Yes":  # If example is positive
        for j in range(num_attribute):
            if hypothesis[j] == '0':
                hypothesis[j] = a[i][j]
            elif hypothesis[j] != a[i][j]:
                hypothesis[j] = '?'
        
        print("\nThe hypothesis for training instance {} is:".format(i+1), hypothesis)

print("\nThe maximally specific hypothesis is:")
print(hypothesis)
