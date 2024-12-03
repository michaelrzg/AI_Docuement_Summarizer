from transformers import TFBartForConditionalGeneration, BartTokenizer
import tensorflow as tf

# Load the trained BART model and tokenizer
model = TFBartForConditionalGeneration.from_pretrained("bart_cnn_dailymail_finetuned")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

# Example test input (e.g., an article to summarize)
test_input = (
    "The global economy has seen significant changes in recent years, "
    "with advancements in technology driving growth in many industries. "
    "However, challenges such as climate change and geopolitical tensions remain pressing concerns."
)

# Tokenize the input for the BART model
input_ids = tokenizer.encode(
    test_input, 
    return_tensors="tf", 
    max_length=1024, 
    truncation=True
)

# Generate the summary using the model
summary_ids = model.generate(
    input_ids, 
    max_length=128, 
    num_beams=4, 
    length_penalty=2.0, 
    early_stopping=True
)

# Decode and print the summary
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print("Summary:")
print(summary)
