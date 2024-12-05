from transformers import TFBartForConditionalGeneration, BartTokenizer
import tensorflow as tf

# Load the trained BART model and tokenizer (trained on cnn dataset in colab)
model = TFBartForConditionalGeneration.from_pretrained("bart_cnn_dailymail_finetuned")
tokenizer = BartTokenizer.from_pretrained("bart_cnn_dailymail_finetuned")

# Example test input (e.g., an article to summarize)
test_input = "KYIV, Ukraine — Russia fired an experimental intermediate-range ballistic missile at Ukraine overnight, Russian President Vladimir Putin said in a TV speech Thursday, warning that the Kremlin could use it against military installations of countries that have allowed Ukraine to use their missiles to strike inside Russia. Putin said the new missile, called \"Oreshnik,\" Russian for \"hazel,\" used a nonnuclear warhead. Ukraine's air force said a ballistic missile hit the central Ukrainian city of Dnipro, saying it was launched from the Astrakhan region in southeastern Russia, more than 770 miles away. Ukrainian officials said it and other rockets damaged an industrial facility, a rehabilitation center for people with disabilities and residential buildings. Three people were injured, according to regional authorities. \"This is an obvious and serious increase in the scale and brutality of this war,\" Ukrainian President Volodymyr Zelenskyy wrote on his Telegram messaging app. The attack came during a week of intense fighting in the nearly three years of war since Russia invaded Ukraine, and it followed U.S. authorization earlier this week for Ukraine to use its sophisticated weapons to strike targets deep inside Russia. Putin said Ukraine had carried out attacks in Russia this week using long-range U.S.-made Army Tactical Missile System (ATACMS) and British-French Storm Shadow missiles. He said Ukraine could not have carried out these attacks without NATO involvement. \"Our test use of Oreshnik in real conflict conditions is a response to the aggressive actions by NATO countries towards Russia,\" Putin said. He also warned: \"We believe that we have the right to use our weapons against military facilities of the countries that allow to use their weapons against our facilities.\""


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
