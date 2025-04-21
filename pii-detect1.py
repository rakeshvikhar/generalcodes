# Install the required packages
# pip install presidio-analyzer presidio-anonymizer spacy
# python -m spacy download en_core_web_lg

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

# Initialize the Presidio Analyzer and Anonymizer
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

# Input text containing PII
text = "My name is John Doe and my phone number is 123-456-7890."

# Analyze the text to identify PII entities
results = analyzer.analyze(
    text=text,
    entities=["PERSON", "PHONE_NUMBER"],  # Specify the types of PII to detect
    language="en"
)

# Print detected PII entities
print("Detected PII entities:")
for result in results:
    detected_text = text[result.start:result.end]  # Extract the detected text using start and end indices
    print(f"Entity: {result.entity_type}, Text: {detected_text}")

# Anonymize the detected PII
anonymized_text = anonymizer.anonymize(
    text=text,
    analyzer_results=results,
    operators={
        "PERSON": OperatorConfig("replace", {"new_value": "<NAME>"}),
        "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "<PHONE>"})
    }
)

# Print the anonymized text
print("\nAnonymized text:")
print(anonymized_text.text)
