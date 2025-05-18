#pip install sdv pandas
import pandas as pd
import os
import json
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import Metadata
from sdv.constraints.base import Constraint

# Custom constraint for text length based on JSON report
class TextLengthConstraint(Constraint):
    def __init__(self, column_name, min_length, max_length):
        super().__init__()  # Initialize base class without arguments
        self.column_name = column_name
        self.min_length = min_length
        self.max_length = max_length
        self._columns = [column_name]  # Define columns for SDV

    def is_valid(self, data):
        """Check if text lengths are within min_length and max_length."""
        if self.column_name not in data:
            return pd.Series([True] * len(data))
        lengths = data[self.column_name].astype(str).str.len()
        return (lengths >= self.min_length) & (lengths <= self.max_length)

    def transform(self, data):
        """Transform data to enforce length constraints."""
        data = data.copy()
        if self.column_name in data:
            data[self.column_name] = data[self.column_name].astype(str).apply(
                lambda x: x[:self.max_length].ljust(self.min_length, ' ')
            )
        return data

    def reverse_transform(self, data):
        """Reverse transform to maintain synthetic data format."""
        return data

    def to_dict(self):
        """Serialize constraint to dictionary for SDV."""
        return {
            'constraint_class': self.__class__.__name__,
            'constraint_parameters': {
                'column_name': self.column_name,
                'min_length': self.min_length,
                'max_length': self.max_length
            }
        }

# Load the banking dataset
csv_file = r"C:\Rakesh\pythonws\bank-additional-full.csv"
if not os.path.exists(csv_file):
    raise FileNotFoundError(f"Dataset not found at {csv_file}. Please ensure the file exists.")
df = pd.read_csv(csv_file, sep=';')  # Assume semicolon-separated for UCI Bank Marketing dataset

# Add a synthetic text column for demonstration
df['text_column'] = pd.Series(['sample text ' + str(i) for i in range(len(df))])

# Load the JSON report
json_file = r"C:\Rakesh\pythonws\bank_m_report.json"
if not os.path.exists(json_file):
    raise FileNotFoundError(f"JSON report not found at {json_file}. Please ensure the file exists.")

with open(json_file, 'r') as f:
    report_data = json.load(f)

# Extract text columns and their length constraints from JSON
variables = report_data.get('variables', {})
text_columns = []
constraints = []

for var_name, var_data in variables.items():
    var_type = var_data.get('type', 'Unknown')
    if var_type in ['Text', 'Categorical']:
        min_length = var_data.get('min_length', None)
        max_length = var_data.get('max_length', None)
        if isinstance(min_length, (int, float)) and isinstance(max_length, (int, float)) and min_length <= max_length:
            # Skip malformed variable names and ensure column exists
            if ';' not in var_name and var_name in df.columns:
                text_columns.append(var_name)
                constraint = TextLengthConstraint(
                    column_name=var_name,
                    min_length=int(min_length),
                    max_length=int(max_length)
                )
                constraints.append(constraint.to_dict())
                print(f"Found text column: {var_name} (min_length={min_length}, max_length={max_length})")

# If no text columns found, use the synthetic text_column with default constraints
if not text_columns:
    print("No valid text columns with min_length and max_length found in JSON. Using synthetic text_column.")
    text_columns = ['text_column']
    constraint = TextLengthConstraint(column_name='text_column', min_length=5, max_length=20)
    constraints = [constraint.to_dict()]
    print("Applied default constraints for text_column: min_length=5, max_length=20")

# Create SDV metadata
metadata = Metadata()
metadata.detect_table_from_dataframe(table_name='bank', data=df)
for col in text_columns:
    metadata.update_column(table_name='bank', column_name=col, sdtype='text')

# Create and fit the synthesizer
synthesizer = GaussianCopulaSynthesizer(
    metadata,
    enforce_min_max_values=True,
    enforce_rounding=False
)
synthesizer.add_constraints(constraints=constraints)
synthesizer.fit(df)

# Generate synthetic data
synthetic_data = synthesizer.sample(num_rows=len(df))

# Verify text length constraints
for col in text_columns:
    lengths = synthetic_data[col].str.len()
    print(f"\nText column '{col}' lengths - Min: {lengths.min()}, Max: {lengths.max()}")

# Save synthetic data
synthetic_data.to_csv('synthetic_banking_data.csv', index=False)
print("\nSynthetic data saved to 'synthetic_banking_data.csv'")
