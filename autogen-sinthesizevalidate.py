#generated using grok
import pandas as pd
import numpy as np
import json
import re
from datetime import datetime
from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata
from ydata_profiling import ProfileReport
import os
from autogen import AssistantAgent, UserProxyAgent, config_list_from_json

# AutoGen configuration
config_list = config_list_from_json(
    env_or_file="OAI_CONFIG_LIST",
    default_config={
        "model": "gpt-4",
        "api_key": os.getenv("OPENAI_API_KEY")
    }
)

# Ensure output directory exists
output_dir = "synthetic_data"
os.makedirs(output_dir, exist_ok=True)

class DataLoaderAgent:
    def __init__(self):
        self.name = "DataLoaderAgent"
        self.agent = AssistantAgent(
            name=self.name,
            system_message="I load datasets from CSV or Excel files and return the pandas DataFrame.",
            llm_config={"config_list": config_list}
        )

    def load_data(self, input_file_path):
        try:
            if input_file_path.endswith('.csv'):
                df = pd.read_csv(input_file_path)
            elif input_file_path.endswith('.xlsx'):
                df = pd.read_excel(input_file_path)
            else:
                raise ValueError("Unsupported file format. Use CSV or Excel.")
            print("Data loaded successfully.")
            return df
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")

class DataProfilerAgent:
    def __init__(self):
        self.name = "DataProfilerAgent"
        self.agent = AssistantAgent(
            name=self.name,
            system_message="I profile datasets using ydata_profiling and generate a report with statistics.",
            llm_config={"config_list": config_list}
        )

    def profile_data(self, df, output_dir):
        profile = ProfileReport(df, title="Dataset Profile Report", minimal=True)
        profile_file = os.path.join(output_dir, "profile_report.html")
        profile.to_file(profile_file)
        
        profile_dict = profile.to_dict()
        profile_summary = {
            "columns": {},
            "row_count": profile_dict['table']['n'],
            "column_count": profile_dict['table']['n_var'],
            "missing_cells": profile_dict['table']['n_cells_missing']
        }
        
        for col in df.columns:
            profile_summary["columns"][col] = {
                "dtype": str(df[col].dtype),
                "missing_values": profile_dict['variables'][col]['n_missing'],
                "unique_values": profile_dict['variables'][col]['n_distinct']
            }
        
        print(f"Profile report saved to {profile_file}")
        return profile_summary, profile_file

class SensitiveDataAgent:
    def __init__(self):
        self.name = "SensitiveDataAgent"
        self.agent = AssistantAgent(
            name=self.name,
            system_message="I detect sensitive data (e.g., emails, SSNs, phone numbers) in datasets.",
            llm_config={"config_list": config_list}
        )

    def detect_sensitive_data(self, df, profile_summary):
        sensitive_patterns = {
            "email": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            "ssn": r'\d{3}-\d{2}-\d{4}',
            "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
        }
        
        sensitive_columns = []
        for col in df.columns:
            if df[col].dtype == "object":
                for name, pattern in sensitive_patterns.items():
                    if df[col].astype(str).str.contains(pattern, regex=True, na=False).any():
                        sensitive_columns.append(col)
                        profile_summary["columns"][col]["sensitive"] = name
                        break
                else:
                    profile_summary["columns"][col]["sensitive"] = None
        
        print(f"Sensitive columns detected: {sensitive_columns}")
        return sensitive_columns, profile_summary

class MetadataAgent:
    def __init__(self):
        self.name = "MetadataAgent"
        self.agent = AssistantAgent(
            name=self.name,
            system_message="I create and save metadata for SDV synthesis, incorporating sensitive data flags.",
            llm_config={"config_list": config_list}
        )

    def create_metadata(self, df, sensitive_columns, output_dir):
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(df)
        
        for col in sensitive_columns:
            metadata.update_column(column_name=col, sdtype='unknown', pii=True)
        
        metadata_dict = metadata.to_dict()
        metadata_file = os.path.join(output_dir, "metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata_dict, f, indent=4)
        
        print(f"Metadata saved to {metadata_file}")
        return metadata, metadata_file

class SynthesizerSelectorAgent:
    def __init__(self):
        self.name = "SynthesizerSelectorAgent"
        self.agent = AssistantAgent(
            name=self.name,
            system_message="I select the best SDV synthesizer based on data characteristics.",
            llm_config={"config_list": config_list}
        )

    def select_synthesizer(self, df, metadata):
        numerical_cols = [col for col in df.columns if np.issubdtype(df[col].dtype, np.number)]
        categorical_cols = [col for col in df.columns if df[col].dtype == "object"]
        
        if len(numerical_cols) > len(categorical_cols):
            print("Selecting GaussianCopulaSynthesizer for numerical-heavy data.")
            return GaussianCopulaSynthesizer(metadata=metadata)
        else:
            print("Selecting CTGANSynthesizer for categorical-heavy or mixed data.")
            return CTGANSynthesizer(metadata=metadata)

class DataValidatorAgent:
    def __init__(self):
        self.name = "DataValidatorAgent"
        self.agent = AssistantAgent(
            name=self.name,
            system_message="I validate synthetic data against the original dataset.",
            llm_config={"config_list": config_list}
        )

    def validate_synthetic_data(self, original_df, synthetic_df, output_dir):
        validation_results = {
            "row_count_match parent's": len(synthetic_df) == len(original_df),
            "column_count_match": len(synthetic_df.columns) == len(original_df.columns),
            "missing_values": synthetic_df.isna().sum().to_dict()
        }
        
        for col in original_df.select_dtypes(include=[np.number]).columns:
            if col in synthetic_df.columns:
                orig_min, orig_max = original_df[col].min(), original_df[col].max()
                synth_min, synth_max = synthetic_df[col].min(), synthetic_df[col].max()
                validation_results[f"{col}_range"] = {
                    "original": [float(orig_min), float(orig_max)] if not pd.isna([orig_min, orig_max]).any() else [None, None],
                    "synthetic": [float(synth_min), float(synth_max)] if not pd.isna([synth_min, synth_max]).any() else [None, None],
                    "within_range": (orig_min <= synth_min and synth_max <= orig_max) if not pd.isna([orig_min, orig_max, synth_min, synth_max]).any() else False
                }
        
        validation_file = os.path.join(output_dir, "validation_results.json")
        with open(validation_file, 'w') as f:
            json.dump(validation_results, f, indent=4)
        
        print(f"Validation results saved to {validation_file}")
        return validation_results, validation_file

class DataSynthesizerAgent:
    def __init__(self):
        self.name = "DataSynthesizerAgent"
        self.agent = AssistantAgent(
            name=self.name,
            system_message="I generate synthetic data using the selected synthesizer and save it to a file.",
            llm_config={"config_list": config_list}
        )

    def synthesize_data(self, df, synthesizer, num_rows, output_dir):
        synthesizer.fit(df)
        synthetic_df = synthesizer.sample(num_rows=num_rows)
        
        output_file = os.path.join(output_dir, f"synthetic_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        synthetic_df.to_csv(output_file, index=False)
        
        print(f"Synthetic data saved to {output_file}")
        return synthetic_df, output_file

class OrchestratorAgent:
    def __init__(self, input_file_path, output_dir="synthetic_data"):
        self.input_file_path = input_file_path
        self.output_dir = output_dir
        self.user_proxy = UserProxyAgent(
            name="UserProxy",
            system_message="I coordinate tasks among agents.",
            human_input_mode="NEVER",
            llm_config={"config_list": config_list}
        )
        self.agents = {
            "loader": DataLoaderAgent(),
            "profiler": DataProfilerAgent(),
            "sensitive": SensitiveDataAgent(),
            "metadata": MetadataAgent(),
            "selector": SynthesizerSelectorAgent(),
            "validator": DataValidatorAgent(),
            "synthesizer": DataSynthesizerAgent()
        }

    def run(self, user_approved_metadata=True):
        results = {}

        # Step 1: Load data
        df = self.agents["loader"].load_data(self.input_file_path)
        results["input_file"] = self.input_file_path

        # Step 2: Profile data
        profile_summary, profile_file = self.agents["profiler"].profile_data(df, self.output_dir)
        results["profile"] = profile_summary
        results["profile_file"] = profile_file

        # Step 3: Detect sensitive data
        sensitive_columns, profile_summary = self.agents["sensitive"].detect_sensitive_data(df, profile_summary)
        results["profile"] = profile_summary
        results["sensitive_columns"] = sensitive_columns

        # Step 4: Create metadata
        metadata, metadata_file = self.agents["metadata"].create_metadata(df, sensitive_columns, self.output_dir)
        results["metadata"] = metadata.to_dict()
        results["metadata_file"] = metadata_file

        # Step 5: Validate metadata (simulated user approval)
        if not user_approved_metadata:
            raise ValueError("Metadata not approved by user.")
        results["metadata_approved"] = True

        # Step 6: Select synthesizer
        synthesizer = self.agents["selector"].select_synthesizer(df, metadata)
        results["synthesizer_type"] = type(synthesizer).__name__

        # Step 7: Synthesize data
        num_rows = len(df)
        synthetic_df, output_file = self.agents["synthesizer"].synthesize_data(df, synthesizer, num_rows, self.output_dir)
        results["output_file"] = output_file

        # Step 8: Validate synthetic data
        validation_results, validation_file = self.agents["validator"]. Eğ

System: I notice the code was cut off again at the end of the `OrchestratorAgent` class. I'll regenerate the complete AutoGen-based data synthesis system, ensuring all parts are included. The artifact will retain the same `artifact_id` as it's an update to the previous code. The code will be a downloadable Python file (`data_synthesis_system_autogen.py`) that implements the full pipeline with all agents.

### Approach
- **Agents**:
  - `DataLoaderAgent`: Loads the dataset.
  - `DataProfilerAgent`: Profiles the dataset using `ydata_profiling`.
  - `SensitiveDataAgent`: Detects sensitive data (e.g., emails, SSNs).
  - `MetadataAgent`: Creates SDV metadata.
  - `SynthesizerSelectorAgent`: Selects the synthesis algorithm.
  - `DataValidatorAgent`: Validates synthetic data.
  - `DataSynthesizerAgent`: Generates synthetic data.
  - `OrchestratorAgent`: Coordinates all agents.
- **AutoGen**: Manages agent collaboration with a `UserProxyAgent` for orchestration.
- **Dependencies**: `autogen`, `pandas`, `ydata_profiling`, `sdv`, `numpy`.
- **Output**: Generates profiling reports, metadata, validation results, and synthetic data in the `synthetic_data` directory.

<xaiArtifact artifact_id="ba5f28fd-56c5-4e7c-b0bd-f1e4e1d634c3" artifact_version_id="6ad0bd7e-1781-4686-bcaf-a3bebe60ab03" title="data_synthesis_system_autogen.py" contentType="text/python">
import pandas as pd
import numpy as np
import json
import re
from datetime import datetime
from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata
from ydata_profiling import ProfileReport
import os
from autogen import AssistantAgent, UserProxyAgent, config_list_from_json

# AutoGen configuration
config_list = config_list_from_json(
    env_or_file="OAI_CONFIG_LIST",
    default_config={
        "model": "gpt-4",
        "api_key": os.getenv("OPENAI_API_KEY")
    }
)

# Ensure output directory exists
output_dir = "synthetic_data"
os.makedirs(output_dir, exist_ok=True)

class DataLoaderAgent:
    def __init__(self):
        self.name = "DataLoaderAgent"
        self.agent = AssistantAgent(
            name=self.name,
            system_message="I load datasets from CSV or Excel files and return the pandas DataFrame.",
            llm_config={"config_list": config_list}
        )

    def load_data(self, input_file_path):
        try:
            if input_file_path.endswith('.csv'):
                df = pd.read_csv(input_file_path)
            elif input_file_path.endswith('.xlsx'):
                df = pd.read_excel(input_file_path)
            else:
                raise ValueError("Unsupported file format. Use CSV or Excel.")
            print("Data loaded successfully.")
            return df
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")

class DataProfilerAgent:
    def __init__(self):
        self.name = "DataProfilerAgent"
        self.agent = AssistantAgent(
            name=self.name,
            system_message="I profile datasets using ydata_profiling and generate a report with statistics.",
            llm_config={"config_list": config_list}
        )

    def profile_data(self, df, output_dir):
        profile = ProfileReport(df, title="Dataset Profile Report", minimal=True)
        profile_file = os.path.join(output_dir, "profile_report.html")
        profile.to_file(profile_file)
        
        profile_dict = profile.to_dict()
        profile_summary = {
            "columns": {},
            "row_count": profile_dict['table']['n'],
            "column_count": profile_dict['table']['n_var'],
            "missing_cells": profile_dict['table']['n_cells_missing']
        }
        
        for col in df.columns:
            profile_summary["columns"][col] = {
                "dtype": str(df[col].dtype),
                "missing_values": profile_dict['variables'][col]['n_missing'],
                "unique_values": profile_dict['variables'][col]['n_distinct']
            }
        
        print(f"Profile report saved to {profile_file}")
        return profile_summary, profile_file

class SensitiveDataAgent:
    def __init__(self):
        self.name = "SensitiveDataAgent"
        self.agent = AssistantAgent(
            name=self.name,
            system_message="I detect sensitive data (e.g., emails, SSNs, phone numbers) in datasets.",
            llm_config={"config_list": config_list}
        )

    def detect_sensitive_data(self, df, profile_summary):
        sensitive_patterns = {
            "email": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            "ssn": r'\d{3}-\d{2}-\d{4}',
            "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
        }
        
        sensitive_columns = []
        for col in df.columns:
            if df[col].dtype == "object":
                for name, pattern in sensitive_patterns.items():
                    if df[col].astype(str).str.contains(pattern, regex=True, na=False).any():
                        sensitive_columns.append(col)
                        profile_summary["columns"][col]["sensitive"] = name
                        break
                else:
                    profile_summary["columns"][col]["sensitive"] = None
        
        print(f"Sensitive columns detected: {sensitive_columns}")
        return sensitive_columns, profile_summary

class MetadataAgent:
    def __init__(self):
        self.name = "MetadataAgent"
        self.agent = AssistantAgent(
            name=self.name,
            system_message="I create and save metadata for SDV synthesis, incorporating sensitive data flags.",
            llm_config={"config_list": config_list}
        )

    def create_metadata(self, df, sensitive_columns, output_dir):
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(df)
        
        for col in sensitive_columns:
            metadata.update_column(column_name=col, sdtype='unknown', pii=True)
        
        metadata_dict = metadata.to_dict()
        metadata_file = os.path.join(output_dir, "metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata_dict, f, indent=4)
        
        print(f"Metadata saved to {metadata_file}")
        return metadata, metadata_file

class SynthesizerSelectorAgent:
    def __init__(self):
        self.name = "SynthesizerSelectorAgent"
        self.agent = AssistantAgent(
            name=self.name,
            system_message="I select the best SDV synthesizer based on data characteristics.",
            llm_config={"config_list": config_list}
        )

    def select_synthesizer(self, df, metadata):
        numerical_cols = [col for col in df.columns if np.issubdtype(df[col].dtype, np.number)]
        categorical_cols = [col for col in df.columns if df[col].dtype == "object"]
        
        if len(numerical_cols) > len(categorical_cols):
            print("Selecting GaussianCopulaSynthesizer for numerical-heavy data.")
            return GaussianCopulaSynthesizer(metadata=metadata)
        else:
            print("Selecting CTGANSynthesizer for categorical-heavy or mixed data.")
            return CTGANSynthesizer(metadata=metadata)

class DataValidatorAgent:
    def __init__(self):
        self.name = "DataValidatorAgent"
        self.agent = AssistantAgent(
            name=self.name,
            system_message="I validate synthetic data against the original dataset.",
            llm_config={"config_list": config_list}
        )

    def validate_synthetic_data(self, original_df, synthetic_df, output_dir):
        validation_results = {
            "row_count_match": len(synthetic_df) == len(original_df),
            "column_count_match": len(synthetic_df.columns) == len(original_df.columns),
            "missing_values": synthetic_df.isna().sum().to_dict()
        }
        
        for col in original_df.select_dtypes(include=[np.number]).columns:
            if col in synthetic_df.columns:
                orig_min, orig_max = original_df[col].min(), original_df[col].max()
                synth_min, synth_max = synthetic_df[col].min(), synthetic_df[col].max()
                validation_results[f"{col}_range"] = {
                    "original": [float(orig_min), float(orig_max)] if not pd.isna([orig_min, orig_max]).any() else [None, None],
                    "synthetic": [float(synth_min), float(synth_max)] if not pd.isna([synth_min, synth_max]).any() else [None, None],
                    "within_range": (orig_min <= synth_min and synth_max <= orig_max) if not pd.isna([orig_min, orig_max, synth_min, synth_max]).any() else False
                }
        
        validation_file = os.path.join(output_dir, "validation_results.json")
        with open(validation_file, 'w') as f:
            json.dump(validation_results, f, indent=4)
        
        print(f"Validation results saved to {validation_file}")
        return validation_results, validation_file

class DataSynthesizerAgent:
    def __init__(self):
        self.name = "DataSynthesizerAgent"
        self.agent = AssistantAgent(
            name=self.name,
            system_message="I generate synthetic data using the selected synthesizer and save it to a file.",
            llm_config={"config_list": config_list}
        )

    def synthesize_data(self, df, synthesizer, num_rows, output_dir):
        synthesizer.fit(df)
        synthetic_df = synthesizer.sample(num_rows=num_rows)
        
        output_file = os.path.join(output_dir, f"synthetic_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        synthetic_df.to_csv(output_file, index=False)
        
        print(f"Synthetic data saved to {output_file}")
        return synthetic_df, output_file

class OrchestratorAgent:
    def __init__(self, input_file_path, output_dir="synthetic_data"):
        self.input_file_path = input_file_path
        self.output_dir = output_dir
        self.user_proxy = UserProxyAgent(
            name="UserProxy",
            system_message="I coordinate tasks among agents.",
            human_input_mode="NEVER",
            llm_config={"config_list": config_list}
        )
        self.agents = {
            "loader": DataLoaderAgent(),
            "profiler": DataProfilerAgent(),
            "sensitive": SensitiveDataAgent(),
            "metadata": MetadataAgent(),
            "selector": SynthesizerSelectorAgent(),
            "validator": DataValidatorAgent(),
            "synthesizer": DataSynthesizerAgent()
        }

    def run(self, user_approved_metadata=True):
        results = {}

        # Step 1: Load data
        df = self.agents["loader"].load_data(self.input_file_path)
        results["input_file"] = self.input_file_path

        # Step 2: Profile data
        profile_summary, profile_file = self.agents["profiler"].profile_data(df, self.output_dir)
        results["profile"] = profile_summary
        results["profile_file"] = profile_file

        # Step 3: Detect sensitive data
        sensitive_columns, profile_summary = self.agents["sensitive"].detect_sensitive_data(df, profile_summary)
        results["profile"] = profile_summary
        results["sensitive_columns"] = sensitive_columns

        # Step 4: Create metadata
        metadata, metadata_file = self.agents["metadata"].create_metadata(df, sensitive_columns, self.output_dir)
        results["metadata"] = metadata.to_dict()
        results["metadata_file"] = metadata_file

        # Step 5: Validate metadata (simulated user approval)
        if not user_approved_metadata:
            raise ValueError("Metadata not approved by user.")
        results["metadata_approved"] = True

        # Step 6: Select synthesizer
        synthesizer = self.agents["selector"].select_synthesizer(df, metadata)
        results["synthesizer_type"] = type(synthesizer).__name__

        # Step 7: Synthesize data
        num_rows = len(df)
        synthetic_df, output_file = self.agents["synthesizer"].synthesize_data(df, synthesizer, num_rows, self.output_dir)
        results["output_file"] = output_file

        # Step 8: Validate synthetic data
        validation_results, validation_file = self.agents["validator"].validate_synthetic_data(df, synthetic_df, self.output_dir)
        results["validation_results"] = validation_results
        results["validation_file"] = validation_file

        return results

if __name__ == "__main__":
    # Example usage
    try:
        orchestrator = OrchestratorAgent(input_file_path="input_data.csv")
        results = orchestrator.run()
        print(json.dumps(results, indent=4))
    except Exception as e:
        print(f"Error running the pipeline: {str(e)}")
