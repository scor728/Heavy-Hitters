import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import random
import hashlib

# Task 0 - Dataset Preprocessing
# Load the dataset
df = pd.read_csv('arxiv.txt', sep='\t', header=None, names=['article_id', 'words', 'date'])

# Use a counter to store the frequency of each word (This structure will be more useful later)
word_counter = Counter()

# Load the counter with the dataset words
for words in df['words']:
    word_list = words.split()  # Split by space
    word_counter.update(word_list)

# Task 1 - Brute Force Frequency Analysis
# 1a
word_count = sum(word_counter.values()) 
unique_word_count = len(word_counter)
average_frequency = word_count / unique_word_count

print("\nWord Count:",  word_count)
print("Unique words:", unique_word_count)
print("Average word frequency:", average_frequency)

# 1b
sorted_word_count = sorted(word_counter.values(), reverse=True) # Descending order

figure_size = (10, 6)

# Plot True Word Frequencies
plt.figure(figsize=figure_size)
plt.plot(sorted_word_count)
plt.xlabel('Words (ranked by frequency)')
plt.ylabel('Frequency')
plt.title('True Word Frequency Distribution')
plt.grid(True)
plt.show()



# Task 2 - Reservoir Sampling
def reservoir_sample(stream, sample_size):
    reservoir = []
    num_replacements = 0

    # Process each word in the stream
    for i, word in enumerate(stream):
        if i < sample_size:
            reservoir.append(word)  # Fill the reservoir
        else:
            # Randomly replace words in the reservoir
            j = random.randint(0, i)
            if j < sample_size:
                reservoir[j] = word
                num_replacements += 1

    return reservoir, num_replacements

# Concatenate all words into a stream (different to the word counter)
word_stream = []
for words in df['words']:
    word_stream.extend(words.split())

# Perform sampling with a size of 10000
sample_size = 10000
sample, replacements = reservoir_sample(word_stream, sample_size)

# 2a
# Compute the estimated frequencies from the sample
sample_counter = Counter(sample)
sorted_sample_counts = sorted(sample_counter.values(), reverse=True)

# Plot the estimated frequencies
plt.figure(figsize=figure_size)
plt.plot(sorted_sample_counts)
plt.xlabel('Words (ranked by frequency)')
plt.ylabel('Estimated Frequency')
plt.title('Estimated Word Frequency Distribution - Reservoir Sampling (s=10000)')
plt.grid(True)
plt.show()

print("\nNumber of replacements (1 run):", replacements)

# 2b
num_runs = 5
replacement_counts = []

for _ in range(num_runs):
    _, replacements = reservoir_sample(word_stream, sample_size)
    replacement_counts.append(replacements)

# Find the average number of replacements
average_replacements = sum(replacement_counts) / num_runs

print("\nReplacements over 5 runs:", replacement_counts)
print("Average Replacements:", average_replacements)



# Task 3 - Misra-Gries Algorithm
def misra_gries(word_stream, k):
    summary = {}
    decrements = 0

    for word in word_stream:
        if word in summary: # Word exists in summary
            summary[word] += 1
        elif len(summary) < k: # Summary is not full yet
            summary[word] = 1
        else: # Summary is full
            # Decrement every word count
            for key in list(summary.keys()):
                summary[key] -= 1

                # Remove words with count 0
                if summary[key] == 0:
                    del summary[key]
                    decrements += 1

    # Adjust the summary to hold the approximate counts

    scale_factor = len(word_stream) // k
    for key in summary:
        summary[key] *= scale_factor

    return summary, decrements

# 3a
k = 1000
mg_summary, num_decrements = misra_gries(word_stream, k)

# convert to counter
mg_counter = Counter(mg_summary)
sorted_mg_counts = sorted(mg_counter.values(), reverse=True)

# Plot the estimated frequencies in descending order
plt.figure(figsize=figure_size)
plt.plot(sorted_mg_counts)
plt.xlabel('Words (ranked by frequency)')
plt.ylabel('Estimated Frequency')
plt.title('Estimated Word Frequency Distribution - Misra-Gries (k=1000)')
plt.grid(True)
plt.show()

print("\nDecrement steps (k=1000):", num_decrements)

# 3b
k_values = [2000, 4000, 6000, 8000, 10000]
max_errors = []

# True word frequencies using Counter
true_counter = Counter(word_stream)

for k in k_values:
    mg_summary, _ = misra_gries(word_stream, k)
    max_error = max(abs(true_counter[word] - mg_summary.get(word, 0)) for word in true_counter)
    max_errors.append(max_error)

# Plot the maximum absolute error vs. summary size k
plt.figure(figsize=figure_size)
plt.plot(k_values, max_errors, marker='o')
plt.xlabel('Summary size (k)')
plt.ylabel('Maximum Absolute Error')
plt.title('Summary Size vs Maximum Error (Misra-Gries)')
plt.grid(True)
plt.show()

# 3c
chosen_k = 4000

# 3d
mg_summary_chosen, num_decrements_chosen = misra_gries(word_stream, chosen_k)
print(f"\nNumber of decrement steps for chosen k={chosen_k}: {num_decrements_chosen}")



# TASK 4 - CountMin Sketch


# Step 3: Implement the CountMin Sketch algorithm
class CountMinSketch:
    def __init__(self, w, d):
        self.w = w  # number of buckets
        self.d = d  # number of hash functions
        self.table = np.zeros((d, w), dtype=int)
        self.hash_seeds = [i * 100 + 7 for i in range(d)]  # using distinct seeds for hash functions

    def _hash(self, word, seed):
        return int(hashlib.md5(f"{word}{seed}".encode()).hexdigest(), 16) % self.w

    def update(self, word):
        for i in range(self.d):
            hash_value = self._hash(word, self.hash_seeds[i])
            self.table[i][hash_value] += 1

    def estimate(self, word):
        estimates = [
            self.table[i][self._hash(word, self.hash_seeds[i])]
            for i in range(self.d)
        ]
        return min(estimates)

# Step 4a: Run CountMin Sketch with w=2000, d=2
w, d = 2000, 2
cms = CountMinSketch(w, d)

# Update CMS with the word stream
for word in word_stream:
    cms.update(word)

# Estimate frequencies of words using CMS
estimated_freqs = defaultdict(int)
for word in set(word_stream):
    estimated_freqs[word] = cms.estimate(word)

# Sort estimated frequencies in descending order
sorted_estimated_freqs = sorted(estimated_freqs.values(), reverse=True)

# Plot the estimated frequencies
plt.figure(figsize=(10, 6))
plt.plot(sorted_estimated_freqs)
plt.xlabel('Words (ranked by estimated frequency)')
plt.ylabel('Estimated Frequency')
plt.title(f'Estimated Word Frequency Distribution from CountMin Sketch (w={w}, d={d})')
plt.grid(True)
plt.show()

# Step 4b: Investigate impact of varying w and d on maximum absolute error
w_values = [2000, 4000, 6000, 8000, 10000]
d_values = [2, 4, 8, 16]
true_counter = Counter(word_stream)

# Store maximum errors for each combination of w and d
error_matrix = np.zeros((len(d_values), len(w_values)))

for i, d in enumerate(d_values):
    print("d = ", d)
    for j, w in enumerate(w_values):
        print("w = ", w)
        cms = CountMinSketch(w, d)
        for word in word_stream:
            cms.update(word)
        max_error = max(
            abs(true_counter[word] - cms.estimate(word))
            for word in true_counter
        )
        error_matrix[i][j] = max_error

# Plot the heatmap of maximum absolute error
plt.figure(figsize=(10, 6))
sns.heatmap(error_matrix, xticklabels=w_values, yticklabels=d_values, annot=True, fmt='.0f', cmap='viridis')
plt.xlabel('Number of buckets (w)')
plt.ylabel('Number of hash functions (d)')
plt.title('Maximum Absolute Error for CountMin Sketch')
plt.show()

# Step 4c: Explanation for choosing w and d
# To keep the error below 100 with high probability (more than 90%), we select w and d based on
# the error formula. The error is inversely proportional to w and decreases with higher d.
# With high probability (1 - (1/e^d)), the error is bounded by the stream size/w.

# For a stream size of 200,000 and error threshold 100:
# Choose w such that 200000/w <= 100. Thus, w >= 2000.
# To ensure a 90% probability, use d >= 4.
chosen_w, chosen_d = 4000, 4

print("About to run CMS")
# Run CountMin Sketch with chosen parameters and report frequencies of words with true frequency > 5000
cms = CountMinSketch(chosen_w, chosen_d)
for word in word_stream:
    cms.update(word)

print("Updated Words, About to estimate word counts")

# Estimate the frequency of words whose true frequency is above 5000
high_freq_words = [word for word, count in true_counter.items() if count > 5000]
estimated_high_freqs = {word: cms.estimate(word) for word in high_freq_words}

print(f"Estimated frequencies for words with true frequency > 5000 using CountMin Sketch (w={chosen_w}, d={chosen_d}):")
for word, est_freq in estimated_high_freqs.items():
    print(f"{word}: {est_freq}")