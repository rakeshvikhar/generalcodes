#generated using grok- not tested-
import pandas as pd
import autogen
from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from table_evaluator import TableEvaluator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.stats import ks_2samp
import json
import os

# Configuration for AutoGen
config_list = [
    {
        "model": "gpt-4o-mini",
        "api_key": os.environ.get("OPENAI_API_KEY"),
    }
]

# Load and preprocess input dataset
def load_dataset(file_path):
    df = pd.read_csv(file_path)
    # Optimize dtypes for memory efficiency
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() / len(df) < 0.5:  # Convert low-cardinality strings to category
            df[col] = df[col].astype('category')
    return df

# Synthesizer functions
def generate_gaussian_copula(original_data, num_rows):
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(original_data)
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(original_data)
    synthetic_data = synthesizer.sample(num_rows=num_rows)
    return synthetic_data

def generate_ctgan(original_data, num_rows):
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(original_data)
    synthesizer = CTGANSynthesizer(metadata, epochs=100)  # Reduced epochs for speed
    synthesizer.fit(original_data)
    synthetic_data = synthesizer.sample(num_rows=num_rows)
    return synthetic_data

# Validation function
def validate_synthetic_data(original_data, synthetic_data, target_column=None):
    evaluator = TableEvaluator(original_data, synthetic_data)
    
    # Statistical similarity (KS test for numerical columns)
    ks_scores = []
    numerical_cols = original_data.select_dtypes(include=['float32', 'int32']).columns
    for col in numerical_cols:
        stat, _ = ks_2samp(original_data[col], synthetic_data[col])
        ks_scores.append(1 - stat)  # Higher is better (less difference)
    ks_score = sum(ks_scores) / len(ks_scores) if ks_scores else 0
    
    # ML utility (if target column is provided)
    ml_score = 0
    if target_column and target_column in original_data.columns:
        X_synth = synthetic_data.drop(columns=[target_column])
        y_synth = synthetic_data[target_column]
        X_orig = original_data.drop(columns=[target_column])
        y_orig = original_data[target_column]
        
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_synth, y_synth)
        y_pred = model.predict(X_orig)
        ml_score = accuracy_score(y_orig, y_pred)
    
    # Combined score (weighted average)
    combined_score = 0.5 * ks_score + 0.5 * ml_score if ml_score else ks_score
    return {"ks_score": ks_score, "ml_score": ml_score, "combined_score": combined_score}

# AutoGen Agent Definitions
manager_agent = autogen.AssistantAgent(
    name="ManagerAgent",
    llm_config={"config_list": config_list},
    system_message="You are the Manager Agent. Parse user input, select synthesizers (GaussianCopula, CTGAN, or both), coordinate synthesis, and trigger validation. Return validation scores and recommend the best synthesizer."
)

gaussian_copula_agent = autogen.AssistantAgent(
    name="GaussianCopulaAgent",
    llm_config={"config_list": config_list},
    system_message="You are the GaussianCopula Agent. Generate synthetic data using the Gaussian Copula model when requested."
)

ctgan_agent = autogen.AssistantAgent(
    name="CTGANSynthesizerAgent",
    llm_config={"config_list": config_list},
    system_message="You are the CTGAN Agent. Generate synthetic data using the CTGAN model when requested."
)

validator_agent = autogen.AssistantAgent(
    name="ValidatorAgent",
    llm_config={"config_list": config_list},
    system_message="You are the Validator Agent. Validate synthetic data against the original using statistical similarity (KS test) and ML utility (RandomForestClassifier accuracy). Return scores."
)

# Group chat setup
group_chat = autogen.GroupChat(
    agents=[manager_agent, gaussian_copula_agent, ctgan_agent, validator_agent],
    messages=[],
    max_round=10
)
group_chat_manager = autogen.GroupChatManager(
    groupchat=group_chat,
    llm_config={"config_list": config_list}
)

# Function to process user request
def process_user_request(file_path, num_rows, target_column=None, synthesizers=["GaussianCopula", "CTGAN"]):
    # Load dataset
    original_data = load_dataset(file_path)
    
    # Store synthetic data and validation results
    synthetic_results = {}
    
    # Generate synthetic data
    for synth in synthesizers:
        if synth.lower() == "gaussiancopula":
            synthetic_data = generate_gaussian_copula(original_data, num_rows)
            synthetic_results["GaussianCopula"] = synthetic_data
        elif synth.lower() == "ctgan":
            synthetic_data = generate_ctgan(original_data, num_rows)
            synthetic_results["CTGAN"] = synthetic_data
    
    # Validate synthetic data
    validation_results = {}
    for synth_name, synth_data in synthetic_results.items():
        scores = validate_synthetic_data(original_data, synth_data, target_column)
        validation_results[synth_name] = scores
    
    # Compare and recommend
    best_synthesizer = max(validation_results.items(), key=lambda x: x[1]["combined_score"])
    
    return {
        "validation_results": validation_results,
        "recommendation": f"Best synthesizer: {best_synthesizer[0]} with combined score {best_synthesizer[1]['combined_score']:.4f}"
    }

# Example usage
if __name__ == "__main__":
    # Simulate user input
    file_path = "input_data.csv"
    num_rows = 1000
    target_column = "target"  # Optional: specify if dataset has a target column for ML validation
    
    # Process request
    result = process_user_request(file_path, num_rows, target_column)
    
    # Print results
    print(json.dumps(result, indent=2))
