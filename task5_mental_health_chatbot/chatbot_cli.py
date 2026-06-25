import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(model_path="./mental-health-chatbot-final"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_length=100):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    with torch.no_grad():
        outputs = model.generate(
            inputs, max_length=max_length, temperature=0.7,
            top_k=50, top_p=0.95, do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Chatbot:" in response:
        response = response.split("Chatbot:", 1)[-1].strip()
    return response

def chat():
    print("Loading model...")
    model, tokenizer = load_model()
    print("\n" + "="*50)
    print("  Mental Health Support Chatbot")
    print("  Type 'quit' to exit")
    print("="*50)
    print("\nHello! I'm here to listen and support you.")
    print("How are you feeling today?\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("\nChatbot: Take care of yourself. Remember, you're not alone. 💙")
            break
        formatted_prompt = f"User: {user_input}\nChatbot:"
        response = generate_response(model, tokenizer, formatted_prompt, max_length=100)
        print(f"\nChatbot: {response}\n")

if __name__ == "__main__":
    chat()
