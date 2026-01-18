import csv
# Read CSV data
def load_data(filename):
    with open(filename, 'r') as file:
        data = list(csv.reader(file))
    return data[1:]  # skip header
def candidate_elimination(data):
    num_attributes = len(data[0]) - 1
    # Initialize S and G
    S = ['0'] * num_attributes
    G = [['?'] * num_attributes]
    for example in data:
        attributes = example[:-1]
        label = example[-1]
        # POSITIVE example
        if label == 'Yes':
            for i in range(num_attributes):
                if S[i] == '0':
                    S[i] = attributes[i]
                elif S[i] != attributes[i]:
                    S[i] = '?'
            # Remove inconsistent hypotheses from G
            G = [g for g in G if all(
                g[i] == '?' or g[i] == S[i] for i in range(num_attributes)
            )]
        # NEGATIVE example
        else:
            new_G = []
            for g in G:
                for i in range(num_attributes):
                    if g[i] == '?' and S[i] != attributes[i]:
                        new_hypothesis = g.copy()
                        new_hypothesis[i] = S[i]
                        new_G.append(new_hypothesis)
            G = new_G
    return S, G
# Load dataset
training_data = load_data("training_data.csv")
# Run Candidate Elimination
S_final, G_final = candidate_elimination(training_data)
# Display results
print("Final Specific Boundary (S):")
print(S_final)
print("\nFinal General Boundary (G):")
for g in G_final:
    print(g)
