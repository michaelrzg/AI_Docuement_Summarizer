from transformers import BartTokenizer, TFBartForConditionalGeneration
import tensorflow as tf
from datasets import load_dataset

# Check GPU availability
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Maximum sequence length for BART
max_input_length = 1024  # Maximum tokenized length for the article
max_target_length = 128  # Maximum tokenized length for the summary

# Batch size for training
batch_size = 4  # Lower batch size for sequence-to-sequence tasks
learning_rate = 2e-5
number_of_epochs = 1

# Initialize the tokenizer
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')


# Preprocess input and target text for BART
def convert_example_to_feature(article, summary):
    inputs = tokenizer(
        article,
        max_length=max_input_length,
        padding="max_length",
        truncation=True,
        return_tensors="tf"
    )
    targets = tokenizer(
        summary,
        max_length=max_target_length,
        padding="max_length",
        truncation=True,
        return_tensors="tf"
    )
    return inputs["input_ids"], inputs["attention_mask"], targets["input_ids"]


# Map dataset examples to model inputs
def map_example_to_dict(input_ids, attention_masks, labels):
    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "labels": labels,
    }


# Encode examples from the dataset
def encode_examples(dataset, limit=-1):
    input_ids_list = []
    attention_mask_list = []
    label_list = []

    if limit > 0:
        dataset = dataset.select(range(limit))

    for example in dataset:
        article = example["article"]
        summary = example["highlights"]  # Use "highlights" as the target
        input_ids, attention_mask, labels = convert_example_to_feature(article, summary)
        input_ids_list.append(input_ids[0])  # Extract tensor from batch dimension
        attention_mask_list.append(attention_mask[0])
        label_list.append(labels[0])

    return tf.data.Dataset.from_tensor_slices(
        (input_ids_list, attention_mask_list, label_list)
    ).map(map_example_to_dict)


# Step 1: Load CNN/DailyMail dataset
dataset = load_dataset("cnn_dailymail", "3.0.0")

# Split into training and testing datasets
train_dataset = dataset["train"]
test_dataset = dataset["validation"]

# Step 2: Preprocess the datasets
print("Encoding Training Dataset..")
ds_train_encoded = encode_examples(train_dataset, limit=20000).shuffle(10000).batch(batch_size)

print("Encoding Testing Dataset.. ")
ds_test_encoded = encode_examples(test_dataset, limit=2000).batch(batch_size)

# Step 3: Initialize BART model
print("Initializing BART model..")
model = TFBartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-08)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

print("Compiling model..")
model.compile(optimizer=optimizer, loss=loss)

# Step 4: Train the model
print("Training BART model..")
bart_history = model.fit(ds_train_encoded, epochs=number_of_epochs, validation_data=ds_test_encoded)

# Step 5: Save the model
print("Training Complete. \nSaving...")
model.save_pretrained("bart_cnn_dailymail_finetuned")
