import pandas as pd
from dataprofiler import Data, Profiler
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
    
    # Step 1: Profile with DataProfiler
    data = Data(df)
    profile = Profiler(data)
    report = profile.report()
    
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
    
    # Update metadata based on DataProfiler and Presidio
    for col, stats in report['data_stats'].items():
        col_type = stats['data_type']
        is_sensitive = stats.get('is_sensitive', False) or col in pii_columns
        if col_type in ['int', 'float']:
            metadata.update_column(column_name=col, sdtype='numerical')
        elif col_type == 'datetime':
            metadata.update_column(column_name=col, sdtype='datetime')
        elif stats['categorical']:
            metadata.update_column(column_name=col, sdtype='categorical')
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
#-----------------------------
# Update your profile_and_generate_metadata function (from your previous script) to handle sdtypes explicitly:

# def profile_and_generate_metadata(file_path):
#     df = pd.read_csv(file_path, dtype={'age': 'float32', 'gender': 'category'})
    
#     # Profile with DataProfiler
#     from dataprofiler import Profiler
#     data = Data(df)
#     profile = Profiler(data)
#     report = profile.report()
    
#     # PII detection with Presidio
#     from presidio_analyzer import AnalyzerEngine
#     analyzer = AnalyzerEngine()
#     pii_columns = []
#     for col in df.select_dtypes(include=['object']).columns:
#         sample_text = df[col].iloc[0] if not df[col].isna().iloc[0] else ""
#         if sample_text:
#             results = analyzer.analyze(text=sample_text, entities=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER"], language="en")
#             if results:
#                 pii_columns.append(col)
    
#     # Generate SDV metadata
#     from sdv.metadata import SingleTableMetadata
#     metadata = SingleTableMetadata()
#     metadata.detect_from_dataframe(df)
    
#     for col, stats in report['data_stats'].items():
#         col_type = stats['data_type']
#         is_sensitive = stats.get('is_sensitive', False) or col in pii_columns
#         if col_type in ['int', 'float']:
#             metadata.update_column(column_name=col, sdtype='numerical')
#         elif col_type == 'datetime':
#             metadata.update_column(column_name=col, sdtype='datetime')
#         elif stats['categorical']:
#             metadata.update_column(column_name=col, sdtype='categorical')
#         elif is_sensitive:
#             pii_type = 'email' if 'email' in col.lower() else 'text'
#             metadata.update_column(column_name=col, sdtype=pii_type, pii=True)
#         else:
#             metadata.update_column(column_name=col, sdtype='text')
    
#     return df, metadata
