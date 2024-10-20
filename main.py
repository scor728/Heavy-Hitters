import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import random
import hashlib
import seaborn as sns
import numpy as np

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

# Task 2 - Reservoir Sampling

# Step 2: Define Reservoir Sampling function
def reservoir_sampling(stream, sample_size):
    """
    Perform reservoir sampling on a stream of words.
    
    Parameters:
    - stream: An iterable of words (the word stream).
    - sample_size: The size of the reservoir.
    
    Returns:
    - A sample of words with the specified size.
    - The number of replacements that occurred during sampling.
    """
    reservoir = []
    num_replacements = 0

    # Process each word in the stream
    for i, word in enumerate(stream):
        if i < sample_size:
            reservoir.append(word)  # Fill the reservoir initially
        else:
            # Randomly decide if the new element should replace an existing one
            j = random.randint(0, i)
            if j < sample_size:
                reservoir[j] = word
                num_replacements += 1

    return reservoir, num_replacements

# Step 3: Prepare the word stream from the dataset
# Concatenate all words from each article into a single stream of words
word_stream = []
for words in df['words']:
    word_stream.extend(words.split())

# Step 4a: Perform reservoir sampling and analyze frequency distribution
sample_size = 10000
sample, replacements = reservoir_sampling(word_stream, sample_size)

# Compute the estimated frequencies from the sample
sample_counter = Counter(sample)
sorted_sample_counts = sorted(sample_counter.values(), reverse=True)

# Plot the estimated frequencies
plt.figure(figsize=(10, 6))
plt.plot(sorted_sample_counts)
plt.xlabel('Words (ranked by frequency)')
plt.ylabel('Estimated Frequency')
plt.title('Estimated Word Frequency Distribution from Reservoir Sampling')
plt.grid(True)
plt.show()

print(f"Number of replacements in one run: {replacements}")

# Step 4b: Run Reservoir Sampling 5 times and calculate the average replacements
num_runs = 5
replacement_counts = []

for _ in range(num_runs):
    _, replacements = reservoir_sampling(word_stream, sample_size)
    replacement_counts.append(replacements)

# Calculate the average number of replacements
average_replacements = sum(replacement_counts) / num_runs

print(f"Replacements over 5 runs: {replacement_counts}")
print(f"Average number of replacements over 5 runs: {average_replacements}")


# Step 3: Implement the Misra-Gries algorithm
def misra_gries(stream, k):
    """
    Misra-Gries algorithm to find frequent elements.
    Parameters:
    - stream: An iterable of words.
    - k: Size of the summary.
    Returns:
    - A dictionary with estimated frequencies of words.
    - The number of decrement steps.
    """
    summary = {}
    num_decrements = 0

    for word in stream:
        if word in summary:
            summary[word] += 1
        elif len(summary) < k:
            summary[word] = 1
        else:
            # Decrement counts for all words
            for key in list(summary.keys()):
                summary[key] -= 1
                if summary[key] == 0:
                    del summary[key]
                    num_decrements += 1

    # Adjust the summary to hold the approximate counts
    for key in summary:
        summary[key] *= len(stream) // k

    return summary, num_decrements

# Step 4a: Run Misra-Gries with k=1000 and plot estimated frequencies
k = 1000
mg_summary, num_decrements = misra_gries(word_stream, k)

# Convert the summary to a Counter for easier analysis
mg_counter = Counter(mg_summary)
sorted_mg_counts = sorted(mg_counter.values(), reverse=True)

# Plot the estimated frequencies in descending order
plt.figure(figsize=(10, 6))
plt.plot(sorted_mg_counts)
plt.xlabel('Words (ranked by frequency)')
plt.ylabel('Estimated Frequency')
plt.title('Estimated Word Frequency Distribution from Misra-Gries (k=1000)')
plt.grid(True)
plt.show()

print(f"Number of decrement steps for k=1000: {num_decrements}")

# Step 4b: Evaluate the impact of varying k on the maximum absolute error
k_values = [2000, 4000, 6000, 8000, 10000]
max_errors = []

# True word frequencies using Counter
true_counter = Counter(word_stream)

for k in k_values:
    mg_summary, _ = misra_gries(word_stream, k)
    max_error = max(abs(true_counter[word] - mg_summary.get(word, 0)) for word in true_counter)
    max_errors.append(max_error)

# Plot the maximum absolute error vs. summary size k
plt.figure(figsize=(10, 6))
plt.plot(k_values, max_errors, marker='o')
plt.xlabel('Summary size (k)')
plt.ylabel('Maximum Absolute Error')
plt.title('Impact of Summary Size on Maximum Absolute Error (Misra-Gries)')
plt.grid(True)
plt.show()

print(f"Maximum absolute errors for k values {k_values}: {max_errors}")

# Step 4c: Explanation for choosing k to find frequent words (>5000 occurrences)
# If we want to capture words with frequency > 5000, a common approach is to set k such that
# it can represent 1 / k fraction of the total stream length. Since the stream has a large number
# of words, choosing k to be in the range of 2000 to 4000 should be reasonable.

chosen_k = 4000
print(f"For words with frequency > 5000, a suitable k might be around {chosen_k}.")

# Step 4d: Run Misra-Gries with the chosen k and report decrement steps
mg_summary_chosen, num_decrements_chosen = misra_gries(word_stream, chosen_k)
print(f"Number of decrement steps for chosen k={chosen_k}: {num_decrements_chosen}")


# TASK 4 ....


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