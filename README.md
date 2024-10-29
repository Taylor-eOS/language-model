An attempt to train GPT-Neo with own text input. Chockes due to lack of memory. Language models, particularly large ones like GPT-Neo, require significantly more memory than other machine learning models because they handle vast amounts of sequential data and process large token embeddings. Unlike typical ML models, which may involve fixed-size inputs or more straightforward calculations, language models generate text by keeping track of complex relationships across potentially thousands of tokens, demanding extensive memory for both the model weights and intermediate states during training. Even with 16GB of RAM, running these models on a CPU can be difficult because they aren't just constrained by RAM alone but also by the CPU's slower processing speed and lack of dedicated memory bandwidth compared to a GPU. Optimization, like reducing sequence length, batch size, or gradient accumulation, can help, but memory demands may still limit training feasibility for larger language models.
