import csv

# Define a geographical grid layout for US states
# Each sublist represents a row in the grid, grouping neighboring states
geographical_grid = [
    ["Washington", "Oregon", "Idaho", "Montana"],
    ["California", "Nevada", "Utah", "Wyoming"],
    ["Arizona", "New Mexico", "Colorado", "North Dakota"],
    ["Texas", "Oklahoma", "Kansas", "South Dakota"],
    ["Minnesota", "Iowa", "Missouri", "Nebraska"],
    ["Wisconsin", "Illinois", "Indiana", "Michigan"],
    ["Ohio", "Kentucky", "Tennessee", "West Virginia"],
    ["Pennsylvania", "New York", "Virginia", "North Carolina"],
    ["South Carolina", "Georgia", "Florida", "Alabama"],
    ["Arkansas", "Louisiana", "Mississippi", "Delaware"],
    ["Maryland", "New Jersey", "Connecticut", "Rhode Island"],
    ["Massachusetts", "Vermont", "New Hampshire", "Maine"]
]

# Flatten the grid into a single list of states in geographical order
state_order = [state for row in geographical_grid for state in row]

# List of valid US states to filter the CSV and remove the non-US states
us_states = [
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut",
    "Delaware", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa",
    "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan",
    "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire",
    "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio",
    "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota",
    "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington", "West Virginia",
    "Wisconsin", "Wyoming"
]

# Load the CSV file
input_file = "/home/nada/polimi/amsc/disease-simulation/data/initial_conditions.csv"
output_file = "/home/nada/polimi/amsc/disease-simulation/data/sorted_initial_conditions.csv"

# Read the CSV file
with open(input_file, "r") as infile:
    reader = csv.reader(infile)
    header = next(reader)  # Extract the header
    rows = list(reader)

# Filter rows to include only valid US states
filtered_rows = [row for row in rows if row[0] in us_states]

# Adjust column indices to match the correct structure
# Province_State, Population, Date, Lat, Long, Confirmed, Deaths, Recovered, Active
state_to_row = {row[0]: row for row in filtered_rows}  # Province_State is the first column

# Dynamically determine the number of blocks based on the dataset size
num_rows = len(rows)
num_blocks = max(1, num_rows // 100)  # Example heuristic: 100 rows per block

# Sort the rows based on the geographical order or other criteria
sorted_rows = [state_to_row[state] for state in state_order if state in state_to_row]
remaining_states = [row for row in filtered_rows if row[0] not in state_order]
sorted_rows.extend(remaining_states)

# Write the sorted rows to a new CSV file
with open(output_file, "w", newline="") as outfile:
    writer = csv.writer(outfile)
    writer.writerow(header)  # Write the header
    writer.writerows(sorted_rows)  # Write the sorted rows

print(f"Sorted CSV file saved to {output_file}")
print(f"Dataset divided into {num_blocks} blocks.")
