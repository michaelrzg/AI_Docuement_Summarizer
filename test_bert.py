from transformers import TFBertForSequenceClassification, BertTokenizer
from transformers import BertTokenizer
import tensorflow as tf
# Load the model and tokenizer
model = TFBertForSequenceClassification.from_pretrained("trained")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
# Example of testing, you will need to modify that part to accomodate the full test
test_sentence = "This is a really good movie. I loved it and will watch again"

# don't forget to tokenize your test inputs
predict_input = tokenizer.encode(test_sentence, truncation=True, padding=True, return_tensors="tf")

tf_output = model.predict(predict_input)[0]
tf_prediction = tf.nn.softmax(tf_output, axis=1)

labels = ['Negative','Positive'] #(0:negative, 1:positive)
label = tf.argmax(tf_prediction, axis=1)
label = label.numpy()
print(labels[label[0]])