import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load pre-trained model and tokenizer
def load_model_and_tokenizer():
    try:
        model_name = "distilgpt2"  # Using a lightweight and publicly available model on Hugging Face Hub
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        # Check if CUDA (GPU) is available, otherwise use CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        return model, tokenizer, device
    except Exception as e:
        raise ValueError(f"Error loading model or tokenizer: {str(e)}")

model, tokenizer, device = load_model_and_tokenizer()

# Sample customer service conversation dataset
dataset = [
    "User: What are your business hours?\nBot: Our business hours are Monday to Friday, 9 AM to 6 PM.",
    "User: Where are you located?\nBot: We are located at 123 Main Street, Springfield.",
    "User: Do you offer delivery services?\nBot: Yes, we offer delivery services within a 20-mile radius.",
    "User: How can I contact customer support?\nBot: You can contact customer support at support@example.com or call (123) 456-7890.",
    "User: What payment methods do you accept?\nBot: We accept credit cards, debit cards, and PayPal."
]

# Response generation function
def respond(message, history):
    try:
        # Build input with limited history to avoid exceeding token limits
        input_text = "\n".join([f"User: {user}\nBot: {bot}" for user, bot in history])
        input_text += f"\nUser: {message}\nBot:"

        # Tokenize input
        inputs = tokenizer(input_text, return_tensors="pt").to(device)

        # Generate response from the model
        output = model.generate(
            inputs["input_ids"],
            max_length=len(inputs["input_ids"][0]) + 50,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
        )

        # Decode output
        response = tokenizer.decode(output[0], skip_special_tokens=True)

        # Extract only the bot's response
        bot_response = response.split("User: {message}\nBot:")[-1].strip()

        return bot_response
    except Exception as e:
        return f"An error occurred during response generation: {str(e)}"

# Define Gradio interface for Hugging Face Spaces
with gr.Blocks() as demo:
    chatbot = gr.ChatInterface(fn=respond, title="Customer Service Chatbot")


# Run the Gradio app
demo.launch()
