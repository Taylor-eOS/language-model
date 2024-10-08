from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizer, pipeline

def main():
    model_path = 'gpt-output'
    device = -1

    try:
        print("Loading tokenizer and model...")
        tokenizer = GPTNeoXTokenizer.from_pretrained(model_path)
        model = GPTNeoXForCausalLM.from_pretrained(model_path)
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        return

    try:
        generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=device)
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        return

    while True:
        prompt = input("Enter your prompt (or 'exit' to quit): ")
        if prompt.lower() == 'exit':
            break

        try:
            print("Generating response...")
            output = generator(prompt, max_length=200, num_return_sequences=1, temperature=0.7, top_p=0.9)
            generated_text = output[0]['generated_text']
            print(f"---\n{generated_text}\n---\n")
        except Exception as e:
            print(f"Error generating response: {e}")

if __name__ == '__main__':
    main()

