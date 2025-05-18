#pip install ydata-profiling pandas kagglehub python-dateutil
import kagglehub
import pandas as pd
import os
import json
from dateutil.parser import parse
from collections import Counter, OrderedDict

# Download the Instagram dataset
path = kagglehub.dataset_download("vasileiosmpletsos/1100-instagram-users-datetime-posts-data")
print("Path to dataset files:", path)

# Load the dataset
csv_file = os.path.join(path, "Instagram_Data.csv")
df = pd.read_csv(csv_file)

# Load or generate the JSON report
json_file = "instagram_profile_report.json"
if not os.path.exists(json_file):
    from ydata_profiling import ProfileReport
    profile = ProfileReport(df, title="Instagram Users Datetime Posts Profiling Report", explorative=True, minimal=False)
    profile.to_file(json_file)

with open(json_file, 'r') as f:
    report_data = json.load(f)

# Function to infer datetime format using dateutil
def infer_datetime_format(values, sample_size=100):
    formats = []
    for value in values.dropna()[:sample_size]:  # Sample non-null values
        try:
            parsed = parse(str(value), fuzzy=False)
            # Construct format based on parsed components
            format_parts = []
            if parsed.year: format_parts.append('%Y')
            if parsed.month: format_parts.append('%m')
            if parsed.day: format_parts.append('%d')
            if parsed.hour: format_parts.append('%H')
            if parsed.minute: format_parts.append('%M')
            if parsed.second: format_parts.append('%S')
            format_str = '-'.join(format_parts[:3]) if len(format_parts) >= 3 else ''
            if len(format_parts) > 3:
                format_str += ' ' + ':'.join(format_parts[3:])
            formats.append(format_str)
        except ValueError:
            continue
    return Counter(formats).most_common(1)[0][0] if formats else 'N/A'

# Modify the variables section
variables = report_data.get('variables', {})
modified_variables = {}

for var_name, var_data in variables.items():
    # Create a new OrderedDict to control key order
    new_var_data = OrderedDict()
    
    # Get the variable type
    var_type = var_data.get('type', 'Unknown')
    
    # Copy keys and insert datetime_format after type
    for key, value in var_data.items():
        new_var_data[key] = value
        if key == 'type':
            # Add datetime_format only for DateTime variables
            if var_type == 'DateTime' and var_name in df.columns:
                new_var_data['datetime_format'] = infer_datetime_format(df[var_name])
            else:
                new_var_data['datetime_format'] = 'N/A'
    
    modified_variables[var_name] = new_var_data

# Update the report data
report_data['variables'] = modified_variables

# Save the modified JSON
modified_json_file = "instagram_profile_report_modified.json"
with open(modified_json_file, 'w') as f:
    json.dump(report_data, f, indent=4)
    #print(f"\nModified JSON saved to '{modified_json_file}'")

# Print a DateTime variable if available, else first variable
datetime_vars = [var for var, data in modified_variables.items() if data.get('type') == 'DateTime']
sample_var = datetime_vars[0] if datetime_vars else next(iter(modified_variables))
#print(f"\nSample variable ({sample_var}):")
#print(json.dumps(modified_variables[sample_var], indent=2))
