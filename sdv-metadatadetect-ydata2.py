import pandas as pd
from ydata_profiling import ProfileReport
from presidio_analyzer import AnalyzerEngine
from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer
import json

def profile_and_generate_metadata(file_path):
    # Load dataset
    df = pd.read_csv(file_path)
    
    # Validate DataFrame
    if df.empty or len(df.columns) == 0:
        raise ValueError("DataFrame is empty or has no columns. Cannot generate metadata.")
    
    # Debug: Print DataFrame info
    print(f"DataFrame shape: {df.shape}")
    print(f"DataFrame columns: {list(df.columns)}")
    print(f"DataFrame dtypes:\n{df.dtypes}")
    
    # Optimize dtypes for memory (for 2M rows)
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    # Step 1: Profile with ydata-profiling
    profile = ProfileReport(df, title="Dataset Metadata Report", minimal=True)
    report = profile.description_set  # Access report data as object
    
    # Save profiling report as JSON (optional, for reference)
    profile.to_file("profile_report.json")
    
    # Step 2: PII detection with Presidio
    analyzer = AnalyzerEngine()
    pii_columns = []
    for col in df.select_dtypes(include=['object']).columns:
        sample_text = df[col].iloc[0] if not df[col].isna().iloc[0] else ""
        if sample_text:
            results = analyzer.analyze(text=sample_text, entities=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER"], language="en")
            if results:
                pii_columns.append(col)
    
    # Step 3: Generate SDV metadata
    metadata = Metadata()
    metadata.add_table(table_name='data')  # Explicitly add table
    
    # Validate table addition
    if 'data' not in metadata.tables:
        raise ValueError("Failed to add table 'data' to metadata.")
    
    # Debug: Print metadata tables
    print(f"Metadata tables: {metadata.tables}")
    
    # Add columns to metadata based on ydata-profiling and Presidio
    for col, stats in report.variables.items():
        col_type = stats['type']  # e.g., Numeric, Categorical, DateTime, Text
        is_sensitive = col in pii_columns
        
        if col_type in ['Numeric', 'Unsupported']:  # Unsupported often indicates numerical with NaNs
            metadata.add_column(column_name=col, sdtype='numerical', table_name='data')
        elif col_type == 'DateTime':
            metadata.add_column(column_name=col, sdtype='datetime', table_name='data')
        elif col_type == 'Categorical':
            metadata.add_column(column_name=col, sdtype='categorical', table_name='data')
        elif col_type == 'Boolean':
            metadata.add_column(column_name=col, sdtype='boolean', table_name='data')
        elif is_sensitive:
            metadata.add_column(column_name=col, sdtype='text', pii=True, table_name='data')
        else:
            metadata.add_column(column_name=col, sdtype='text', table_name='data')
    
    # Update column metadata with additional details (e.g., PII)
    for col, stats in report.variables.items():
        col_type = stats['type']
        is_sensitive = col in pii_columns
        if is_sensitive:
            metadata.update_column(column_name=col, sdtype='text', pii=True, table_name='data')
    
    # Save metadata
    metadata.save_to_json('metadata.json')
    return df, metadata

def generate_synthetic_data(df, metadata, synthesizer_type='GaussianCopula', num_rows=1000):
    if synthesizer_type.lower() == 'gaussiancopula':
        synthesizer = GaussianCopulaSynthesizer(metadata.get_table_metadata('data'))
    elif synthesizer_type.lower() == 'ctgan':
        synthesizer = CTGANSynthesizer(metadata.get_table_metadata('data'), epochs=100)
    else:
        raise ValueError("Unsupported synthesizer")
    
    synthesizer.fit(df)
    synthetic_data = synthesizer.sample(num_rows=num_rows)
    return synthetic_data

# Example usage
if __name__ == "__main__":
    file_path = "/content/sample_data/california_housing_test.csv"
    
    # Profile and generate metadata
    original_data, metadata = profile_and_generate_metadata(file_path)
    print("Generated Metadata:")
    print(json.dumps(metadata.to_dict(), indent=2))
    
    # Generate synthetic data with both synthesizers
    for synth_type in ['GaussianCopula', 'CTGAN']:
        synthetic_data = generate_synthetic_data(original_data, metadata, synthesizer_type=synth_type, num_rows=1000)
        synthetic_data.to_csv(f'synthetic_data_{synth_type}.csv', index=False)
        print(f"Generated synthetic data with {synth_type}")
