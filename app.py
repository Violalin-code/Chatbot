import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import difflib
import random
import re  # For regex operations

# Load the pre-trained model and tokenizer
def load_model_and_tokenizer():
    model_name = "microsoft/DialoGPT-medium"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
    except Exception as e:
        raise ValueError(f"Failed to load model or tokenizer: {str(e)}")

    # Set device (CUDA or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model loaded on: {device}")
    
    return model, tokenizer, device

model, tokenizer, device = load_model_and_tokenizer()

# Expanded dataset for customer service training with fun facts and small talk
response_map = {
    "what are your business hours?": ["Our business hours are Monday to Friday, 9 AM to 6 PM."],
    "where are you located?": ["We are located at 123 Main Street, Springfield."],
    "do you offer delivery services?": ["Yes, we offer delivery services within a 20-mile radius."],
    "how can I contact customer support?": ["You can contact customer support at support@example.com or call (123) 456-7890."],
    "what payment methods do you accept?": ["We accept credit cards, debit cards, and PayPal."],
    "tell me a fun fact": ["Did you know honey never spoils? Archaeologists have found pots of honey in ancient Egyptian tombs that are over 3000 years old and still perfectly edible!"],
    "how are you?": ["I'm just a bot, but I'm here to help you! How about you?"],
    "what's your name?": ["I'm your friendly customer service assistant. What's yours?"],
}


# Function to normalize input
def normalize_input(input_text):
    # Lowercase, remove punctuation and extra spaces
    input_text = input_text.lower()
    input_text = re.sub(r'[^\w\s]', '', input_text)  # Remove punctuation
    return input_text.strip()

# Response generation function
def respond(message, history):
    try:
        # If history is empty, start with a greeting
        if len(history) == 0:
            greeting = "How can I help you?"
            history.append(("User", ""))  # Append a dummy user input to keep the structure
            return greeting
        
        # Normalize the user's message
        normalized_message = normalize_input(message)

        # Check if the user's message matches any predefined questions
        response_list = response_map.get(normalized_message, None)

        if response_list:
            return random.choice(response_list)  # Return a random response from the list

        # If no exact match, find the closest question
        closest_match = difflib.get_close_matches(normalized_message, response_map.keys(), n=1, cutoff=0.6)
        if closest_match:
            return random.choice(response_map[closest_match[0]])  # Return a random response for the closest match

        # If no match found, fallback to the model response
        input_text = (
            "You are a helpful customer service bot. "
            + "\n".join([f"User: {user}\nBot: {bot}" for user, bot in history])
            + f"\nUser: {message}\nBot:"
        )

        # Tokenize input
        inputs = tokenizer(input_text, return_tensors="pt").to(device)

        # Set max_new_tokens to generate a reasonable response length
        max_new_tokens = 50  # Adjust this value as needed

        # Generate response from the model
        output = model.generate(
            inputs["input_ids"],
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
        )

        # Decode output
        response = tokenizer.decode(output[0], skip_special_tokens=True)

        # Extract only the bot's response
        bot_response = response.split(f"User: {message}\nBot:")[-1].strip()

        # Ensure the response is meaningful
        if bot_response == "":
            bot_response = "I'm not sure how to respond to that. Could you rephrase?"

        return bot_response
    except Exception as e:
        return f"An error occurred during response generation: {str(e)}"

# Define Gradio interface for Hugging Face Spaces
with gr.Blocks() as demo:
    chatbot = gr.ChatInterface(fn=respond, title="Customer Service Chatbot", description="Chat with our AI Bot.")

# Run the Gradio app
demo.launch()

