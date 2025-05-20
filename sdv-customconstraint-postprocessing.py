import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer

# Create a sample dataset with a text column
data = pd.DataFrame({
    'id': [1, 2, 3, 4, 5],
    'description': ['Apple', 'Banana', 'Cherry', 'Date', 'Elderberry']
})

# Define the metadata for the dataset
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data)
metadata.update_column(column_name='description', sdtype='text')

# Initialize the GaussianCopulaSynthesizer
synthesizer = GaussianCopulaSynthesizer(
    metadata,
    enforce_min_max_values=True,
    enforce_rounding=False
)

# Fit the synthesizer to the data
synthesizer.fit(data)

# Generate synthetic data
synthetic_data = synthesizer.sample(num_rows=10)

# Postprocess the synthetic data to enforce text length constraints
def enforce_text_length(data, column_name, min_length=4, max_length=8):
    """
    Ensure text column values have length between min_length and max_length.
    
    Args:
        data (pd.DataFrame): Synthetic data
        column_name (str): Name of the text column
        min_length (int): Minimum allowed length for text
        max_length (int): Maximum allowed length for text
    
    Returns:
        pd.DataFrame: Data with text column values adjusted to meet length constraints
    """
    def adjust_length(text):
        if not isinstance(text, str):
            text = str(text)  # Handle non-string cases
        text_len = len(text)
        if text_len < min_length:
            # Pad with 'x' if too short
            padding = ''.join(['x' for _ in range(min_length - text_len)])
            return text + padding
        elif text_len > max_length:
            # Truncate if too long
            return text[:max_length]
        return text

    # Apply length adjustment to the text column
    data[column_name] = data[column_name].apply(adjust_length)
    return data

# Apply the text length constraint to the synthetic data
synthetic_data = enforce_text_length(
    synthetic_data,
    column_name='description',
    min_length=4,
    max_length=8
)

# Display the synthetic data
print("\nSynthetic Data:")
print(synthetic_data)

# Verify text lengths in the synthetic data
print("\nVerifying text lengths (should be between 4 and 8):")
print(synthetic_data['description'].apply(len))
