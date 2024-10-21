import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import random
import hashlib

# Samuel Cordner
# scor728
# 808836611

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
def reservoir_sample(words, reservoir_size):
    reservoir = []
    replacement_count = 0

    # Process each word in the stream
    for i, word in enumerate(words):
        if i < reservoir_size:
            reservoir.append(word)  # Fill the reservoir
        else:
            # Randomly replace words in the reservoir
            j = random.randint(0, i)

            if j < reservoir_size:
                reservoir[j] = word
                replacement_count += 1

    return reservoir, replacement_count

# Concatenate all words into a stream (different to the word counter)
word_stream = []
for words in df['words']:
    word_stream.extend(words.split())

# Perform sampling with a size of 10000
sample_size = 10000
sample, replacements = reservoir_sample(word_stream, sample_size)

scale_factor = word_count // sample_size
for s in sample:
    s = s * scale_factor

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
runs = 5
replacement_counts = []

for i in range(runs):
    _, replacements = reservoir_sample(word_stream, sample_size)
    replacement_counts.append(replacements)

# Find the average number of replacements
average_replacements = sum(replacement_counts) / runs

print("\nReplacements over 5 runs:", replacement_counts)
print("Average Replacements:", average_replacements)



# Task 3 - Misra-Gries Algorithm
def misra_gries(words, k):
    summary_object = {}
    decrement_count = 0

    for word in words:
        if word in summary_object: # Word exists in summary
            summary_object[word] += 1

        elif len(summary_object) < k: # Summary is not full yet
            summary_object[word] = 1

        else: # Summary is full

            # Decrement every word count
            decrement_count += 1
            for key in list(summary_object.keys()):
                summary_object[key] -= 1
                
                # Remove words with count 0
                if summary_object[key] == 0:
                    del summary_object[key]
                    

    # Adjust summary counts to given scale
    scale_factor = len(words) // k
    for key in summary_object:
        summary_object[key] *= scale_factor

    return summary_object, decrement_count

# 3a
k = 1000
summary, decrement_count = misra_gries(word_stream, k)

# convert to counter
counter = Counter(summary)
sorted_counts = sorted(counter.values(), reverse=True)

# Plot the estimated frequencies in descending order
plt.figure(figsize=figure_size)
plt.plot(sorted_counts)
plt.xlabel('Words (ranked by frequency)')
plt.ylabel('Estimated Frequency')
plt.title('Estimated Word Frequency Distribution - Misra-Gries (k=1000)')
plt.grid(True)
plt.show()

print("\nDecrement steps (k=1000):", decrement_count)

# 3b
k_values = [2000, 4000, 6000, 8000, 10000]
max_errors = []

# Rename for clarity
true_word_counts = word_counter

for k in k_values:
    summary, _ = misra_gries(word_stream, k)
    max_error = max(abs(true_word_counts[word] - summary.get(word, 0)) for word in true_word_counts)
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
chosen_k = 500

# 3d
chosen_summary, chosen_decrements = misra_gries(word_stream, chosen_k)
print(f"\nNumber of decrement steps for chosen k={chosen_k}: {chosen_decrements}")



# Task 4 - CountMin Sketch

# Create unique hash from word and seed
def generate_hash(word, seed, w):
    value = int(hashlib.md5(f"{word}{seed}".encode()).hexdigest(), 16)
    return value % w

 # Increment given counts for word in the sketch
def update_table(word, table, hash_function_seeds, d, w):
    for i in range(d):
        # Hash value varies based on given hash function
        hash_value = generate_hash(word, hash_function_seeds[i], w)
        table[i][hash_value] += 1

def estimate_word(word, table, hash_function_seeds, d, w):
    # Go through all hash values for the given word
    estimates = [table[i][generate_hash(word, hash_function_seeds[i], w)] for i in range(d)]
    # Return minimum estimate (to mitigate collisions)
    return min(estimates)

def setup_table(w, d):
    return np.zeros((d, w), dtype=int)

def setup_hash_function_seeds(d):
    return [i * 100 + 7 for i in range(d)] # using distinct seeds for hash functions (ensure they are different)

# 4a
w, d = 2000, 2
table = setup_table(w, d)
hash_function_seeds = setup_hash_function_seeds(d)

# Update CMS with the word stream
for word in word_stream:
    update_table(word, table, hash_function_seeds, d, w)


# Estimate frequencies of words using CMS
estimated_freqs = defaultdict(int)
for word in set(word_stream):
    estimated_freqs[word] = estimate_word(word, table, hash_function_seeds, d, w)

sorted_estimated_freqs = sorted(estimated_freqs.values(), reverse=True) # Descending order

# Plot the estimated frequencies
plt.figure(figsize=figure_size)
plt.plot(sorted_estimated_freqs)
plt.xlabel('Words (ranked by estimated frequency)')
plt.ylabel('Estimated Frequency')
plt.title(f'Estimated Word Frequency - CountMin Sketch (w={w}, d={d})')
plt.grid(True)
plt.show()

# 4b
w_values = [2000, 4000, 6000, 8000, 10000]
d_values = [2, 4, 8, 16]
true_counter = Counter(word_stream)

# Store Maximum errors for each pair of w and d values
error_matrix = np.zeros((len(d_values), len(w_values)))

for i, d in enumerate(d_values):
    print("\nd = ", d)
    for j, w in enumerate(w_values):
        print("w = ", w)

        table = setup_table(w, d)
        hash_function_seeds = setup_hash_function_seeds(d)

        for word in word_stream:
            update_table(word, table, hash_function_seeds, d, w)

        max_error = max(
            abs(true_counter[word] - estimate_word(word, table, hash_function_seeds, d, w))
            for word in true_counter
        )
        error_matrix[i][j] = max_error

# Plot Maximum Absolute Error Heatmap
plt.figure(figsize=figure_size)
sns.heatmap(error_matrix, xticklabels=w_values, yticklabels=d_values, annot=True, fmt='.0f', cmap='viridis')
plt.xlabel('Number of buckets (w)')
plt.ylabel('Number of hash functions (d)')
plt.title('Maximum Absolute Error - CountMin Sketch')
plt.show()

# 4c
chosen_w, chosen_d = 22000, 4

# Run CountMinSketch with selected w and d values

table = setup_table(chosen_w, chosen_d)
hash_function_seeds = setup_hash_function_seeds(chosen_d)


for word in word_stream:
    update_table(word, table, hash_function_seeds, chosen_d, chosen_w)

# Estimate frequency of words with true frequency > 5000
freq_words = [word for word, count in true_counter.items() if count > 5000]
estimated_word_freqs = {word: estimate_word(word, table, hash_function_seeds, chosen_d, chosen_w) for word in freq_words}

# Display Frequency Estimates
print(f"\nEstimated frequencies for words with true frequency > 5000 using CountMin Sketch (w={chosen_w}, d={chosen_d}):")
for word, estimated_freq in estimated_word_freqs.items():
    print(f"{word}: {estimated_freq}")