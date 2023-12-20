# ElevateMind: Your Digital Mental Health Al

### Empathetic Virtual Chatbot for Mental Health Support

Facing the global challenge of mental health, which accounts for 8 million annual deaths and leaves over 28 million in the U.S. without adequate care, we introduce a novel solution: an **empathetic virtual chatbot**. This tool is designed to provide accessible and empathetic support, bridging the gap in mental healthcare.

## Our Approach
Our mission is clear: to offer immediate, compassionate support through advanced technology. Here's what we're doing:

### 📊 Dataset Optimization
- **BERT Summarization**: Condensing chatbot interactions into efficient, context-rich summaries within a 512-token limit.

### 🤖 Chatbot Development
- **Fine-Tuning LLaMA 2 [8]**: Creating a chatbot that excels in empathetic and engaging conversations.

### 🔍 Benchmarking Efficiency
- **Model Evaluation**: Testing different prompting methods to ensure meaningful, context-focused interactions.

Join us in revolutionizing mental health care, making empathetic support accessible to all.



### Dataset
- **Data Source**: Experiment utilizes a HuggingFace Dataset with 850K conversations between users (`usr`) in distress and a system (`sys`) acting as a counselor.
- **Data Selection**: A subset of 25,000 conversations is chosen due to compute constraints.

### Dataset Transformation and Preprocessing
- **Conversion to Instruct-based Format**: Multi-turn dataset is transformed to suit LLaMA 2, a prompt-based model. This involves summarizing long conversations.
- **Preprocessing Steps**:
  - Conversations are merged into a single string using escape characters.
  - BERT Sentence Summarizer creates summarized contexts from past conversations.

### Model Selection and Constraints
- **Model Choice**: The LLaMA-2-7B-HF variant is selected for its feasibility within compute constraints.
- **Context Token Length Constraint**: Memory limitations of Nvidia Tesla V100 GPUs make the LLaMA "chat" version infeasible, leading to the use of preprocessing techniques.

### Training
- **Memory Constraints and Solutions**:
  - LLaMA-7B-Instruct requires up to 112 GB of memory, exceeding the 16 GB VRAM of Nvidia Tesla V100 GPUs.
  - Techniques like 4-bit quantized Low-Rank Adaptation (LoRA) and gradient accumulation are used to reduce memory footprint.
- **Framework**: The Ludwig framework, based on PyTorch, is used for training.
- **Hyperparameters**: [Insert additional information about hyperparameters here]

### Inference
- **Preprocessing for Inference**: Prior conversations are summarized and integrated into a custom prompt for inference.
- **Execution**: This prompt is inputted into the fine-tuned LLaMA model for multi-turn conversations with users.
