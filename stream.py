import pandas as pd
import random

# Define the structure of the Excel file
columns = [f"Q{i}" for i in range(1, 11)]
responses = []  # List to store responses

# Function to generate a single response
def generate_response():
    response = []
    for i in range(1, 11):
        if i in [4, 6, 9]:  # Multi-correct questions
            if i == 4:
                options = ["A", "B", "C", "D", "E"]
            elif i == 6:
                options = ["A", "B", "C", "D", "E"]
            elif i == 9:
                options = ["A", "B", "C", "D", "E"]
            selected = random.sample(options, random.randint(1, len(options)))
            response.append(",".join(selected))
        else:  # Single-correct questions
            if i == 1:
                response.append(random.choice(["A", "B", "C", "D"]))
            elif i == 2:
                response.append(random.choice(["A", "B", "C", "D"]))
            elif i == 3:
                response.append(random.choice(["A", "B", "C", "D"]))
            elif i == 5:
                response.append(random.choice(["A", "B", "C", "D"]))
            elif i == 7:
                response.append(random.choice(["A", "B", "C", "D"]))
            elif i == 8:
                response.append(random.choice(["A", "B", "C", "D", "E"]))
            elif i == 10:
                response.append(random.choice(["A", "B", "C", "D"]))
    return response

# Generate multiple responses (simulating less knowledge about ethics)
for _ in range(50):  # Adjust the number of responses as needed
    responses.append(generate_response())

# Create a DataFrame
responses_df = pd.DataFrame(responses, columns=columns)

# Save to Excel
output_file = "responses.xlsx"
responses_df.to_excel(output_file, index=False)

print(f"Responses have been written to {output_file}")
