import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# Step 1: Load the dataset
df = pd.read_csv('arxiv.txt', sep='\t', header=None, names=['article_id', 'words', 'date'])

# Step 2: Use a Counter to count word frequencies
word_counter = Counter()

# Step 3: Iterate through each row, split words, and update the counter
for words in df['words']:
    word_list = words.split()  # Split each article's words by space
    word_counter.update(word_list)

# Step 4a: Compute the average frequency of the words in the stream.
total_word_count = sum(word_counter.values())  # Total number of words
unique_word_count = len(word_counter)  # Number of unique words
average_frequency = total_word_count / unique_word_count

print(f"Total word count: {total_word_count}")
print(f"Number of unique words: {unique_word_count}")
print(f"Average frequency of words: {average_frequency}")

# Step 4b: Compute true frequencies and plot the curve
# Sort words by frequency in descending order
sorted_word_counts = sorted(word_counter.values(), reverse=True)

# Plotting frequencies
plt.figure(figsize=(10, 6))
plt.plot(sorted_word_counts)
plt.xlabel('Words (ranked by frequency)')
plt.ylabel('Frequency')
plt.title('Word Frequency Distribution')
plt.grid(True)
plt.show()
