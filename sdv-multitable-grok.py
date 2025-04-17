import pandas as pd
from sdv.metadata import MultiTableMetadata
from sdv.multi_table import HMASynthesizer
from sdv.evaluation.multi_table import evaluate_quality, get_column_plot
import matplotlib.pyplot as plt

# Step 1: Create sample data for two related tables
hotels_data = pd.DataFrame({
    'hotel_id': ['HID_001', 'HID_002', 'HID_003'],
    'city': ['New York', 'Chicago', 'San Francisco'],
    'rating': [4.5, 3.8, 4.2]
})

guests_data = pd.DataFrame({
    'guest_id': [1, 2, 3, 4],
    'hotel_id': ['HID_001', 'HID_001', 'HID_002', 'HID_003'],
    'guest_email': ['john@example.com', 'jane@example.com', 'bob@example.com', 'alice@example.com'],
    'checkin_date': ['2023-06-01', '2023-06-02', '2023-06-03', '2023-06-04'],
    'room_rate': [150.0, 200.0, 180.0, 220.0]
})

# Combine data into a dictionary
real_data = {
    'hotels': hotels_data,
    'guests': guests_data
}

# Step 2: Define the metadata
metadata = MultiTableMetadata()

# Detect metadata from dataframes (automatically infers relationships)
metadata.detect_from_dataframes(data=real_data)

# Update hotels table metadata
metadata.update_column(
    table_name='hotels',
    column_name='hotel_id',
    sdtype='id',
    regex_format='HID_[0-9]{3}'
)
metadata.update_column(
    table_name='hotels',
    column_name='city',
    sdtype='categorical'
)
metadata.update_column(
    table_name='hotels',
    column_name='rating',
    sdtype='numerical'
)
metadata.set_primary_key(table_name='hotels', column_name='hotel_id')

# Update guests table metadata
metadata.update_column(
    table_name='guests',
    column_name='guest_id',
    sdtype='id'
)
metadata.update_column(
    table_name='guests',
    column_name='hotel_id',
    sdtype='id'
)
metadata.update_column(
    table_name='guests',
    column_name='guest_email',
    sdtype='email',
    pii=True
)
metadata.update_column(
    table_name='guests',
    column_name='checkin_date',
    sdtype='datetime',
    datetime_format='%Y-%m-%d'
)
metadata.update_column(
    table_name='guests',
    column_name='room_rate',
    sdtype='numerical'
)
metadata.set_primary_key(table_name='guests', column_name='guest_id')

# Skip add_relationship since detect_from_dataframes likely added it
# Validate metadata
metadata.validate()

# Step 3: Initialize and fit the synthesizer
synthesizer = HMASynthesizer(metadata)

# Add a constraint to ensure checkin_date is reasonable (optional)
checkin_constraint = {
    'constraint_class': 'Inequality',
    'table_name': 'guests',
    'constraint_parameters': {
        'low_column_name': 'checkin_date',
        'high_column_name': 'checkin_date',
        'strict_boundaries': True
    }
}
synthesizer.add_constraints(constraints=[checkin_constraint])

# Fit the synthesizer to the real data
synthesizer.fit(real_data)

# Step 4: Generate synthetic data
synthetic_data = synthesizer.sample(scale=1.0)  # scale=1.0 generates same number of rows as original

# Step 5: Evaluate the synthetic data
quality_report = evaluate_quality(
    real_data=real_data,
    synthetic_data=synthetic_data,
    metadata=metadata
)

# Print quality report details
print("Quality Report:")
print(quality_report.get_details('Column Shapes'))

# Step 6: Visualize a column comparison
fig = get_column_plot(
    real_data=real_data,
    synthetic_data=synthetic_data,
    metadata=metadata,
    table_name='guests',
    column_name='room_rate'
)
plt.title('Real vs Synthetic Data: Room Rate Distribution')
plt.show()

# Save synthetic data to CSV for inspection
synthetic_data['hotels'].to_csv('synthetic_hotels.csv', index=False)
synthetic_data['guests'].to_csv('synthetic_guests.csv', index=False)

print("Synthetic data saved to 'synthetic_hotels.csv' and 'synthetic_guests.csv'")
