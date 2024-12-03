from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
from datasets import load_dataset

# Check GPU availability
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Maximum sequence length for BERT
max_length = 512

# Batch size for training
batch_size = 6

# Learning rate for optimizer
learning_rate = 2e-5

# Number of epochs
number_of_epochs = 1

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


# Convert examples to features
def convert_example_to_feature(review):
    return tokenizer.encode_plus(
        review,
        add_special_tokens=True,  # Add [CLS] and [SEP]
        max_length=max_length,    # Maximum sequence length
        padding="max_length",     # Pad to max_length
        truncation=True,          # Truncate if text exceeds max_length
        return_attention_mask=True,  # Add attention mask
    )


# Map dataset examples to model inputs
def map_example_to_dict(input_ids, attention_masks, token_type_ids, label):
    return {
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "attention_mask": attention_masks,
    }, label


# Encode examples from the dataset
def encode_examples(dataset, limit=-1):
    input_ids_list = []
    token_type_ids_list = []
    attention_mask_list = []
    label_list = []

    if limit > 0:
        dataset = dataset.select(range(limit))

    for example in dataset:
        review = example["article"]
        label = example["highlights"]  # Use "highlights" as labels
        bert_input = convert_example_to_feature(review)
        input_ids_list.append(bert_input['input_ids'])
        token_type_ids_list.append(bert_input['token_type_ids'])
        attention_mask_list.append(bert_input['attention_mask'])
        label_list.append(label)

    return tf.data.Dataset.from_tensor_slices(
        (input_ids_list, attention_mask_list, token_type_ids_list, label_list)
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

# Step 3: Initialize BERT model
print("Initializing model..")
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)  # Summarization has one output label

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-08)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

print("Compiling model..")
model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

# Step 4: Train the model
print("Training model..")
bert_history = model.fit(ds_train_encoded, epochs=number_of_epochs, validation_data=ds_test_encoded)

# Step 5: Save the model
print("Training Complete. \nSaving...")
model.save_pretrained("bert_cnn_dailymail_finetuned")
