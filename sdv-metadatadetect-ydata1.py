import pandas as pd
from ydata_profiling import ProfileReport
from presidio_analyzer import AnalyzerEngine
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer
import json

def profile_and_generate_metadata(file_path):
    # Load dataset
    df = pd.read_csv(file_path)
    
    # Optimize dtypes for memory (for 2M rows)
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    # Step 1: Profile with ydata-profiling
    profile = ProfileReport(df, title="Dataset Metadata Report", minimal=True)
    report = profile.to_dict()  # Get report as dictionary
    
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
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)  # Initial detection
    
    # Update metadata based on ydata-profiling and Presidio
    for col, stats in report['variables'].items():
        col_type = stats['type']  # e.g., Numeric, Categorical, DateTime, Text
        is_sensitive = col in pii_columns
        
        if col_type in ['Numeric', 'Unsupported']:  # Unsupported often indicates numerical with NaNs
            metadata.update_column(column_name=col, sdtype='numerical')
        elif col_type == 'DateTime':
            metadata.update_column(column_name=col, sdtype='datetime')
        elif col_type == 'Categorical':
            metadata.update_column(column_name=col, sdtype='categorical')
        elif col_type == 'Boolean':
            metadata.update_column(column_name=col, sdtype='boolean')
        elif is_sensitive:
            metadata.update_column(column_name=col, sdtype='text', pii=True)
        else:
            metadata.update_column(column_name=col, sdtype='text')
    
    # Save metadata
    metadata.save_to_json('metadata.json')
    return df, metadata

def generate_synthetic_data(df, metadata, synthesizer_type='GaussianCopula', num_rows=1000):
    if synthesizer_type.lower() == 'gaussiancopula':
        synthesizer = GaussianCopulaSynthesizer(metadata)
    elif synthesizer_type.lower() == 'ctgan':
        synthesizer = CTGANSynthesizer(metadata, epochs=100)
    else:
        raise ValueError("Unsupported synthesizer")
    
    synthesizer.fit(df)
    synthetic_data = synthesizer.sample(num_rows=num_rows)
    return synthetic_data

# Example usage
if __name__ == "__main__":
    file_path = "input_data.csv"
    
    # Profile and generate metadata
    original_data, metadata = profile_and_generate_metadata(file_path)
    print("Generated Metadata:")
    print(json.dumps(metadata.to_dict(), indent=2))
    
    # Generate synthetic data with both synthesizers
    for synth_type in ['GaussianCopula', 'CTGAN']:
        synthetic_data = generate_synthetic_data(original_data, metadata, synthesizer_type=synth_type, num_rows=1000)
        synthetic_data.to_csv(f'synthetic_data_{synth_type}.csv', index=False)
        print(f"Generated synthetic data with {synth_type}")
