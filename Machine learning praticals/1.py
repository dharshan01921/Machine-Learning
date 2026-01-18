# FIND-S Algorithm Implementation

def find_s(training_data):
    # Initialize hypothesis with the first positive example
    hypothesis = None
    
    for instance in training_data:
        if instance[-1] == "Yes":
            hypothesis = instance[:-1]
            break

    # If no positive example found
    if hypothesis is None:
        return None

    # Process remaining training examples
    for instance in training_data:
        if instance[-1] == "Yes":
            for i in range(len(hypothesis)):
                if hypothesis[i] != instance[i]:
                    hypothesis[i] = '?'

    return hypothesis


# Training data
training_data = [
    ["Sunny", "Warm", "Normal", "Strong", "Warm", "Same", "Yes"],
    ["Sunny", "Warm", "High", "Strong", "Warm", "Same", "Yes"],
    ["Rainy", "Cold", "High", "Strong", "Warm", "Change", "No"],
    ["Sunny", "Warm", "High", "Strong", "Cool", "Change", "Yes"]
]

# Run FIND-S
final_hypothesis = find_s(training_data)

# Display result
print("Most Specific Hypothesis:")
print(final_hypothesis)
