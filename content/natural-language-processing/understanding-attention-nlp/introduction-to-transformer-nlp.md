# Introduction to Transformer Models for NLP

[GitHub Code Repository](https://github.com/sinanuozdemir/oreilly-transformers-video-series)

## Introduction to Attention and Language Models

### History of NLP

Learning from large-scale corpora was already causing bias issues.

```python
word2vec_model.most_similar(positive=['woman', 'job'], negative=['man'])
word2vec_model.most_similar(positive=['man', 'job'], negative=['woman'])
```

*Direct bias* - terms like 'football' are inherently closer to male words, where 'nurse' is closer to female words.

```python
wv_model.similarity('nurse', 'woman') # 0.45
wv_model.similarity('nurse', 'man')   # 0.22
```



Indirect bias - more nuanced correlations in the learned embeddings leading to "teacher" being closer to "volleyball" than "football", due to their larger female associations.

```python
wv_model.similarity('nurse', 'volleyball') # 0.43
wv_model.similarity('nurse', 'football')   # 0.19
```

- Large scale bag of words (2013)
- RNN and CNN for NLP (2013-2014)
- Sequence to sequence model (2014)
- Attention (2015)
- Transformers (2017)
- Pretrained Language Model (2018)



### Attention

[Neural Machine Translation](https://arxiv.org/pdf/1409.0473.pdf)

> When traslating English to French, we see that the French word _marin_ is paying attention to the English word _marine_ even though they don't occur at the same index.

With self-attention, we alter the idea of the attention to relating each word in the phrase with every other word to teach the model the relationships between them.

### Transformer

Transformer consists of an encoder and a decoder. The encoder takes in our **input** and outputs a matrix **representation** of that input. The decoder takes in that representation and iteratively generates an **output**.

The encoder is actually an **encoder stack** with multiple encoders.

The decoder is actually a **decoder stack** with multiple decoders.

### How language models look at text

In a language modeling task, a model is trained to predict a missing word in a sequence of words. There are two types of language models:

- **Auto-regressive**: to predict a future token given either the past tokens or the future tokens but not both. _If you don't <blank> (forward prediction)_ or _<blank> at the sign, you will get a ticket. (Backward prediction)_ 
  - Predicting the next word in a sentence (auto-complete)
  - Natural language generation (NLG)
  - GPT family
- **Auto-encoding**:  to learn representations of the entire sequence by predicting tokens given both the past and future tokens. _If you don't <blank> at the sign, you will get a ticket._ 
  - Comprehensive understanding and encoding of entire sequences of tokens
  - Natural language understanding (NLU)
  - BERT



## How Transformers Use Attention to Process Text

### Scaled dot product attention

$X_{m\times n}$, as $m$ is the number of tokens, and $n$ is the token dimension.

Each row is the token embedding for each token in the input.

This input has already gone through BERT's preprocessing (token, segment, and position).

Linear transformation:

$$X_{m\times n} \cdot W^Q_{n\times k} = Q_{m\times k}$$ to obtain the **query space**.

$$X_{m\times n} \cdot W^K_{n\times k} = K_{m\times k}$$ to obtain the **key space**.

$$X_{m\times n} \cdot W^V_{n\times k} = V_{m\times k}$$ to obtain the **value space**.

Why? We want to impart meaning to these matrices. And this is one of those aspects of mathematics where a mathematician trying to create a formula will decide or impart intuitive representations or intuitive meanings behind matrices and then allow a training phase to learn the proper weight matrices, these Ws, WQ, WK, and WV. We want to learn the values in this matrix that impart the meaning as the architect intended. So the intended intuitive meaning of these matrices is:

- The query matrix is meant to represent a piece of information we are looking for in a query we have.
- The key matrix is intuitively meant to represent the relevance of each word to our query. And the key matrix represents how important each word is to my overall query.
- The value matrix intuitively represents the contextless meaning of our input tokens.

$$Q\cdot K^T$$ is attention score. The larger the value is, the closer the vectors are and the more attention the word has.

For example, in the sentence, `I like cats`, the attention score for `like` is `[.23, .87, .70]`. Divide the $\sqrt {d_k}$ and then apply softmax to it, resulting in `[.22, .42, .36]`. In other words, when focusing on the word "like," I should pay about 22% attention to "I," 35% to "cats," and the rest 42% to "like."

Finally, we multiply the attention probability with its contextual embedding of the word "like."

![Screenshot 2023-02-14 at 1.52.49 PM](/Users/gejun/Library/Application Support/typora-user-images/Screenshot 2023-02-14 at 1.52.49 PM.png)

So by taking the attention scores and multiplying them by our value matrix, we are explicitly taking those tokens, forcing them to look at each other. Then once they've looked at each other, they adjust the value matrix to represent all tokens in our sequence. And the dimension we have for each of those tokens is now no longer a semantic representation as $V$ was. It is now a relevance embedding dimension. Each token now has 300 dimensions of how it is relevant to the sequence and why we're saying it in the first place. 

```python
from transformers import BertModel

model = BertModel.from_pretrained('bert-based-uncased')
model.encoder.layer[0]
model.encoder.layer[0].attention
```

### Multi-headed attention

In each encoder, we apply our multi-headed attention.

In the first encoder, we do the calculation on our initial matrix $X$ like what we did in the previous example.

Every encoder afterward uses the representation matrix outputted by the preceding encoder.

![img](https://jalammar.github.io/images/t/transformer_multi-headed_self-attention-recap.png)

```python
config = BertConfig()
config
```

Why use multi-headed self-attention?

1. Helps BERT  focus on multiple relationships simultaneously.
2. Helps BERT learn different relations between words by transforming our inputs to multiple sub-spaces of Q/K/V.

`Head 8-10` means eight encoders and 10-headed attention.

```python
text = "My friend told me about this class and I love it so far! She was right."
tokens = tokenizer.encode(text)
inputs = torch.tensor(tokens).unsqueeze(0) # (20, ) -> (1, 20)
inputs

attention = model(inputs, output_attentions=True)[2] # grab the attention score from BERT
final_attention = attention[-1].mean(1)[0]

# attention_df
df = pd.DataFrame(final_attention.detach()).applymap(float).round(3)
df.columns = tokenizer.convert_ids_to_tokens(tokens)
df.index = tokenizer.convert_ids_to_tokens(tokens)
df

# https://nlp.stanford.edu/pubs/clark2019what.pdf
# layer index 2 seems to be attending to the previous token
# layer index 6 seems to be for pronouns
tokens_as_list = tokenizer.convert_ids_to_tokens(inputs[0])
head_view(attention, tokens_as_list)
# head 3-1 attends to the previous token
head_view(attention, tokenizer.convert_ids_to_tokens(inputs[0]), layer=2, heads=[0])
# head 8-10 relating direct objects to their verbs eg, told -> me
head_view(attention, tokenizer.convert_ids_to_tokens(inputs[0]), layer=7, heads=[9])

# attention in the 8th encoder's 10th head to see direct object attention
eight_ten = attention[7][0][9]

df = pd.DataFrame(eight_ten.detach()).applymap(float).round(3)
df.columns = tokenizer.convert_ids_to_tokens(tokens)
df.index = tokenizer.convert_ids_to_tokens(tokens)
df

```



## Transfer Learning

A model trained for one task is reused as the starting point for a model for a second task.

Transfer learning in NLP is similar. A model is trained on an unlabeled text corpus with an unsupervised task that generally doesn't have a useful objective. It's just meant to learn language/context in general.

Three approaches to fine-tuning.

1. Update the whole model on labeled data, plus any additional layers added on top. (Slow, but best performance)
2. Freeze a subset of the model. (In the middle)
3. Freeze the whole model and only train the added layers on the top. (fast, but worst performance)

### Fine-Tune transformers with PyTorch

Fine-tunning with HuggingFace's Trainer



## BERT

Bi-directional Encoder Representation from Transformers

- Auto-encoding language model
- Use only the encoder from the transformer.
- Relying on self-attention
- The encoder is taken from the transformer architecture.

`[CLS]` stands for classification, and it's used to represent sentence-level classification.

> BERT is designed primarily for transfer learning, i.e., finetuning on task-specific datasets. If you average the states, every state is averaged with the same weight: including stop words or other things irrelevant to the task. The `[CLS]` vector gets computed using self-attention (like everything in BERT), so it can only collect the relevant information from the rest of the hidden states. So, the `[CLS]` vector is also average over token vectors, only more cleverly computed, specifically for the tasks that you fine-tune.

https://albertauyeung.github.io/2020/06/19/bert-tokenization.html/

> - Language modeling is a practical task for using unlabeled data to pre-train neural networks in NLP.
> - Traditional language models take the previous n tokens and predict the next one. In contrast, BERT trains a language model that considers both the previous and next tokens when predicting.
> - BERT is also trained on a next-sentence prediction task to handle better tasks that require reasoning about the relationship between two sentences (e.g., question answering)
> - BERT uses the Transformer architecture for encoding sentences.
> - BERT performs better when given more parameters, even on small datasets.

https://datasciencetoday.net/index.php/en-us/nlp/211-paper-dissected-bert-pre-training-of-deep-bidirectional-transformers-for-language-understanding-explained

- [ ] blog post on BERT



### BERT's Wordpiece Tokenization

BERT's tokenizer handles OOV tokens (out of vocabulary) by breaking them into smaller chunks of known tokens. For example, the sentence "Sinan loves a beautiful day" would be tokenized as `["[CLS]", "sin", "##an", "love", "a", "beautiful", "day", "[SEP]"]` with `##` indicating a subword.

BERT has a maximum sequence length of 512 tokens for efficiency. Any sequence with less than 512 tokens will be padded to reach 512, and the model may error out if the sequence is over 512.

`uncased` removes accents and changes the input to lower cases.

Cased tokenization works well in cases where the case does matter, like Named Entity Recognition.

Consider the following sentences:

1. I love my pet Python.
2. I love coding in Python.

The token "Python" will end up with a vector representation from each sentence via BERT. What's interesting is that the vector representation "Python" will be different for each sentence because of the surrounding words in the sentence.

```python
pet = tokenizer.encode("I love my pet python")
language = tokenizer.encode("I love coding in python")

model(torch.tensor(pet).unsqueeze(0))[0][:, 5, :].detach().numpy()
model(torch.tensor(language).unsqueeze(0))[0][:, 5, :].detach().numpy()
model(torch.tensor(tokenizer.encode('snake')).unsqueeze(0))[0][:, 5, :].detach().numpy()
model(torch.tensor(tokenizer.encode('programming')).unsqueeze(0))[0][:, 5, :].detach().numpy()

cosine_similarity(pet_embedding, snake_embedding)
```

### The many embeddings of BERT

BERT applies three separate types of embeddings to tokenized sentences:

1. Token embeddings

   a. Represents context-less meaning of each token

   b. A lookup of 30,522 possible vectors (for BERT-base)

   c. This is learnable during training

2. Segment Embeddings

​		a. Distinguishes between multiple inputs (for Q/A example)

​		b. A lookup of 2 possible vectors (one for sentence A and one for sentence B)

​		c. This is **not** learnable

3. Position Embeddings

   a. Used to represent the token's position in the sentence

   b. This is **not** learnable

![Screenshot 2023-02-16 at 10.24.18 AM](/Users/gejun/Library/Application Support/typora-user-images/Screenshot 2023-02-16 at 10.24.18 AM.png)

Positional Embeddings

$i$ - position of the token (between 0 and 511)

$j$ - position of the embedding dimension (between 0 and 768 for BERT-base)

$d_{emb\_dim}$ - embedding dimension (e.g., 768 for BERT-base)


$$
p_{i,j} = 
\begin{cases}
    sin \biggl( \frac{i}{10000^{\frac{j}{d_{emb\_dim}}}} \biggl), \text{if } j \text{ is even}\\
    sin \biggl( \frac{i}{10000^{\frac{j-1}{d_{emb\_dim}}}} \biggl), \text{if } j \text{ is odd}
\end{cases}
$$


![Screenshot 2023-02-16 at 11.32.01 AM](/Users/gejun/Library/Application Support/typora-user-images/Screenshot 2023-02-16 at 11.32.01 AM.png)





## Pre-training and fine-tuning BERT

Masked Lanugage Modeling (MLM)

- replace 15% of words in corpus with special [MASK] token and as BERT to fill in the blank
- Think back to our "___ at the light" example.

![Screenshot 2023-02-16 at 12.46.17 PM](/Users/gejun/Library/Application Support/typora-user-images/Screenshot 2023-02-16 at 12.46.17 PM.png)



### Next Sentence Prediction (NSP)

- Classification problem
- Given two sentences, did sentence B come **directly** after sentence A? True or False.
  - A: "Istanbul is a great city to visit." 
  - B: "I was just there."
  - Does sentence B come directly after sentence A?

![Screenshot 2023-02-16 at 12.58.33 PM](/Users/gejun/Library/Application Support/typora-user-images/Screenshot 2023-02-16 at 12.58.33 PM.png)



### Fine-tuning BERT

- Classification: positive vs. negative

  ![Screenshot 2023-02-16 at 3.23.35 PM](/Users/gejun/Library/Application Support/typora-user-images/Screenshot 2023-02-16 at 3.23.35 PM.png)

- Token classification: classify every token![Screenshot 2023-02-16 at 3.24.04 PM](/Users/gejun/Library/Application Support/typora-user-images/Screenshot 2023-02-16 at 3.24.04 PM.png)

- Question/Answer![Screenshot 2023-02-16 at 3.24.49 PM](/Users/gejun/Library/Application Support/typora-user-images/Screenshot 2023-02-16 at 3.24.49 PM.png)



## Hands-on BERT

### Flavors of BERT

- RoBERTa - Robustly Optimized BERT Approach
  - Authors claim BERT was vastly under-trained
  - 10x training data (16GB->160GB)
  - 15% more parameters (architecture)
  - Removed the next sentence prediction task (training) 
  - Dynamic Masking Pattern -> 4x the masking to learn from (training)
- DistilBERT - Distilled BERT
  - Distillation is a technique to train a "student" model to replicate a "teacher" model.
  - 40% fewer parameters, 60% faster (97% of BERT performance)
  - Trained via knowledge distillation
- ALBERT - A Lite BERT
  - Optimize model performance/number of parameters (90% fewer)
  - Factorized embedding parameterization
  - Cross-layer parameter sharing (architecture)
  - Next sentence prediction task became the Sentence order prediction task (training)

```python
def tokenize_and_align_labels(examples):
  tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
  labels = []
  for i, label in enumerate(examples[f"token_labels"]):
    word_ids = tokenized_inputs.word_ids(batch_index=i)
    previous_word_idx = None
    label_ids = []
    for word_idx in word_ids:
      if not word_idx:
        label_ids.append(-100)
      elif word_idx != previous_word_idx:
        label_ids.append(label[word_idx])
      else:
        label_ids.append(-100)
      previous_word_idx = word_idx
    labels.append(label_ids)
  tokenized_inputs["labels"] = labels
  return tokenized_inputs

```



```python
# freeze all but the last 2 encoder layers in BERT to speed up training
from name, param in qa_bert.bert.named_parameters():
  if 'encoder.layer.22' in name:
    break
  param.requires_grad = False
```

Q/A models are very large and take a long time to train.

```python
squad_pipeline = pipeline("question-answering", "bert-large-uncased-whole-word-masking-finetuned-squad")
```

Visualize the logits

```python
large_tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
qa_input = large_tokenizer("Where is Sinan living these days?", "Sinan lives in California but Matt lives in Boston.", return_tensors="pt")

large_qa_bert = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

output = large_qa_bert(**qa_input)


token_labels = large_tokenizer.convert_ids_to_tokens(qa_input['input_ids'].squeeze())

```





## Natural Language Generation with GPT

Generative Pre-trained Transformers

- Auto-regressive language model
- decoders are trained on huge corpora of data
- The decoder is taken from the transformer architecture

Recall that each decoder combines multi-headed attention and feedforward/add/norm layers.

### Tokenization

"Another beautiful day" => ["Another", "beautiful", "day", "<|endoftext|>"]

```python
tokenizer.convert_ids_to_tokens(tokenizer.encode('Sinan loves a beautiful day'))
```

The $\dot{G}$ indicates the beginning of a word.

GPT's tokenizer treats spaces like parts of the tokens, so a word will be encoded differently where it is at the beginning of the sentence (without space) or not

GPT-2 applies two separate types of embeddings to tokenized sentences:

1. Word Token Embedding (WTE)

   a. Represents the context-free meaning of each token

   b. A lookup of over 50,000 possible vectors

   c. This is learnable during training

2. Word Position Embedding(WPE)

   a. Used to represent the token's position in the sentence

   b. This is not learnable

### Masked Self-Attention

![Screenshot 2023-02-17 at 12.29.39 PM](/Users/gejun/Library/Application Support/typora-user-images/Screenshot 2023-02-17 at 12.29.39 PM.png)



The current token cannot attend to the tokens that come before.

### Parameters for inference

- **temperature** - a lower value makes the model more confident and less random. A higher value makes the generated text more random. When the temperature is less than 1, the probabilities are "shaper."
- **top_k** - how many tokens it considers when generating. 0 to deactivate. With `top_k=6`, we readjust probabilities to be sharper for the top 6 possible tokens.
- **top_p** - only considers tokens from the top percent of confidence. With `top_p=0.92`, we readjust probabilities among the minimum number of tokens that _exceed_ the given parameter.
- **beams** - how many tokens out should we consider?
- **do_sample** - if true, randomness is introduced in selection.

Greedy search

<img src="/Users/gejun/Library/Application Support/typora-user-images/Screenshot 2023-02-17 at 12.47.38 PM.png" alt="Screenshot 2023-02-17 at 12.47.38 PM" style="zoom:25%;" />

Beam search

<img src="/Users/gejun/Library/Application Support/typora-user-images/Screenshot 2023-02-17 at 12.48.27 PM.png" alt="Screenshot 2023-02-17 at 12.48.27 PM" style="zoom:25%;" />

### Pre-training GPT

Pre-trained corpus has bias and the bias will be transferred over to downstream task if we are not careful.

### Few-shot Learning

Zero-shot learning - the model may be provided with a task description and no examples prior. The task is set up using a prompt.

One-shot learning - the model is provided with a task description and a single example of the task.

**Few-shot learning** - the model is provided with a task description and as many examples as we desire / will fit into the context window of model. Fot GPT-2 this is 1024 tokens.

```python
print(generator("""Sentiment Analysis
Text: I hate it when my phone battery dies.
Sentiment: Negative
###
Text: My day has been really great!
Sentiment: Positive
###
Text: This new music video was so good.
Sentiment:""", top_k=2, temperature=0.1, max_length=55)[0]['generated_text'])
```

## Hands-on GPT

### GPT for code dictation

English -> LaTeX

x squared --> x^2 --> $x^2$

sum from 1 to 10 of x squared --> $\sum_1^{10} x^2$



## Applications of BERT and GPT

### Siamese BERT-networks for semantic searching

Symmetric search: documents and queries are roughly the same size and carry the same amount of semantic content. Example: retriving news article titles given a query.

Asymmetric search: documents are usually longer than the queries and carry larger amounts of semantic content. Example: retrieving an entire paragraph from a textbook to answer a question.

### Prompt Engineering



## T5 (Text to Text Transfer Transformer)

<img src="/Users/gejun/Library/Application Support/typora-user-images/Screenshot 2023-02-20 at 2.36.57 PM.png" alt="Screenshot 2023-02-20 at 2.36.57 PM" style="zoom:50%;" />



<img src="/Users/gejun/Library/Application Support/typora-user-images/Screenshot 2023-02-20 at 2.37.15 PM.png" alt="Screenshot 2023-02-20 at 2.37.15 PM" style="zoom:50%;" />



<img src="/Users/gejun/Library/Application Support/typora-user-images/Screenshot 2023-02-20 at 2.41.53 PM.png" alt="Screenshot 2023-02-20 at 2.41.53 PM" style="zoom:50%;" />

Three training objectives

1. Casual (auto-regressive) language modeling: predicting the next word
2. BERT-style objective: masking words and predicting the original text
3. Deshuffling: shuffling the input randomly and predicting the original text

### Cross-attention

Using keys and values from the encoder and queries from the decoder.

**Intuition**: As T5 is decoding the text, it has a query, i.e, a reason for speaking. It looks to the encoder to understand the original input as it relates to what is actively being decoded (the keys and values.)

![Screenshot 2023-02-20 at 2.47.20 PM](/Users/gejun/Library/Application Support/typora-user-images/Screenshot 2023-02-20 at 2.47.20 PM.png)









## Hands-on T5



## The Vision Transformer



## Deploying Transfer Models

