from sdv.metadata import MultiTableMetadata
from sdv.multi_table import HMASynthesizer
import pandas as pd

# 1. Define the metadata for our multi-table dataset
metadata = MultiTableMetadata()

# Define tables and their columns
metadata.detect_table_from_dataframe(
    table_name='customers',
    data=pd.DataFrame({
        'customer_id': pd.Series(dtype='int'),
        'name': pd.Series(dtype='str'),
        'email': pd.Series(dtype='str'),
        'age': pd.Series(dtype='int'),
        'gender': pd.Series(dtype='str'),
        'address': pd.Series(dtype='str')
    })
)

metadata.detect_table_from_dataframe(
    table_name='products',
    data=pd.DataFrame({
        'product_id': pd.Series(dtype='int'),
        'name': pd.Series(dtype='str'),
        'category': pd.Series(dtype='str'),
        'price': pd.Series(dtype='float'),
        'in_stock': pd.Series(dtype='bool')
    })
)

metadata.detect_table_from_dataframe(
    table_name='orders',
    data=pd.DataFrame({
        'order_id': pd.Series(dtype='int'),
        'customer_id': pd.Series(dtype='int'),
        'product_id': pd.Series(dtype='int'),
        'quantity': pd.Series(dtype='int'),
        'order_date': pd.Series(dtype='datetime64[ns]'),
        'total_price': pd.Series(dtype='float')
    })
)

# Set primary keys
metadata.set_primary_key(
    table_name='customers',
    column_name='customer_id'
)

metadata.set_primary_key(
    table_name='products',
    column_name='product_id'
)

metadata.set_primary_key(
    table_name='orders',
    column_name='order_id'
)

# Define relationships
metadata.add_relationship(
    parent_table_name='customers',
    child_table_name='orders',
    parent_primary_key='customer_id',
    child_foreign_key='customer_id'
)

metadata.add_relationship(
    parent_table_name='products',
    child_table_name='orders',
    parent_primary_key='product_id',
    child_foreign_key='product_id'
)

# Validate the metadata
metadata.validate()

# 2. Create sample data
sample_customers = pd.DataFrame({
    'customer_id': [1, 2, 3],
    'name': ['Alice', 'Bob', 'Charlie'],
    'email': ['alice@example.com', 'bob@example.com', 'charlie@example.com'],
    'age': [28, 35, 42],
    'gender': ['Female', 'Male', 'Male'],
    'address': ['123 Main St', '456 Oak Ave', '789 Pine Rd']
})

sample_products = pd.DataFrame({
    'product_id': [101, 102, 103],
    'name': ['Laptop', 'Phone', 'Tablet'],
    'category': ['Electronics', 'Electronics', 'Electronics'],
    'price': [999.99, 699.99, 399.99],
    'in_stock': [True, True, False]
})

sample_orders = pd.DataFrame({
    'order_id': [1001, 1002, 1003],
    'customer_id': [1, 2, 3],
    'product_id': [101, 102, 103],
    'quantity': [1, 2, 1],
    'order_date': pd.to_datetime(['2023-01-15', '2023-01-16', '2023-01-17']),
    'total_price': [999.99, 1399.98, 399.99]
})

# 3. Create and fit the synthesizer
synthesizer = HMASynthesizer(metadata)
synthesizer.fit({
    'customers': sample_customers,
    'products': sample_products,
    'orders': sample_orders
})

# 4. Generate synthetic data
synthetic_data = synthesizer.sample(scale=2)

# 5. Access and display results
print("Synthetic Customers:")
print(synthetic_data['customers'].head())
print("\nSynthetic Products:")
print(synthetic_data['products'].head())
print("\nSynthetic Orders:")
print(synthetic_data['orders'].head())
