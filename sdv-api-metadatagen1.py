import pandas as pd
from ydata_profiling import ProfileReport
from presidio_analyzer import AnalyzerEngine
from sdv.metadata import Metadata
import json
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI(title="Data Profiling API")

# Define input model
class ProfileRequest(BaseModel):
    file_name: str

def profile_and_generate_metadata(file_path: str, output_dir: str = "/content/sample_data/output"):
    """
    Generate metadata and profiling report for the input file.
    Save outputs in output_dir and return their file paths.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Validate file path
    if not os.path.exists(file_path):
        raise ValueError(f"File not found: {file_path}")
    
    # Load dataset
    df = pd.read_csv(file_path)
    
    # Validate DataFrame
    if df.empty or len(df.columns) == 0:
        raise ValueError("DataFrame is empty or has no columns. Cannot generate metadata.")
    
    # Debug: Print DataFrame info
    print(f"DataFrame shape: {df.shape}")
    print(f"DataFrame columns: {list(df.columns)}")
    print(f"DataFrame dtypes:\n{df.dtypes}")
    
    # Optimize dtypes for memory
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    # Step 1: Profile with ydata-profiling
    profile = ProfileReport(df, title="Dataset Metadata Report", minimal=True)
    report = profile.description_set
    
    # Save profiling report as JSON
    profile_output_path = os.path.join(output_dir, "profile_report.json")
    profile.to_file(profile_output_path)
    
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
    metadata.add_table(table_name='data')
    
    # Validate table addition
    if 'data' not in metadata.tables:
        raise ValueError("Failed to add table 'data' to metadata.")
    
    # Debug: Print metadata tables
    print(f"Metadata tables: {metadata.tables}")
    
    # Add columns to metadata based on ydata-profiling and Presidio
    for col, stats in report.variables.items():
        col_type = stats['type']
        is_sensitive = col in pii_columns
        
        if col_type in ['Numeric', 'Unsupported']:
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
    metadata_output_path = os.path.join(output_dir, "metadata.json")
    metadata.save_to_json(metadata_output_path)
    
    return {
        "metadata_path": metadata_output_path,
        "profile_report_path": profile_output_path
    }

@app.post("/profile")
async def profile_data(request: ProfileRequest):
    """
    API endpoint to generate metadata and profiling report for a given file.
    Returns paths to the generated files.
    """
    try:
        # Construct full file path (assuming files are in /content/sample_data)
        file_path = os.path.join("/content/sample_data", request.file_name)
        
        # Generate metadata and profiling report
        result = profile_and_generate_metadata(file_path)
        
        # Return file paths
        return {
            "status": "success",
            "metadata_link": result["metadata_path"],
            "profile_report_link": result["profile_report_path"]
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Run the FastAPI app (default: http://127.0.0.1:8000)
    uvicorn.run(app, host="0.0.0.0", port=8000)
