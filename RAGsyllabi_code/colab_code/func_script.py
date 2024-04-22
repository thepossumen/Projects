# imports
from datasets import load_dataset
from torch import Tensor
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder
import transformers
import torch
import re

## TO-DO import specific modules used from transformers, instead of importing entire library

def load_hf_data(text_path="anordkvist/gu-course-syllabus", embedding_path="anordkvist/gu-course-syllabus-embeddings"):
    """
    
    """
    # load text data
    cleaned_text = load_dataset(text_path)
    df_text = cleaned_text['train'].to_pandas()
    # load embeddings
    reference_embeddings = load_dataset(embedding_path)
    df_embeddings = reference_embeddings['train'].to_pandas()

    return df_text, df_embeddings

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def embed_query(query, tokenizer, model):
  """
  Given a query, tokenier and model, creates an embedding return as 1d np array
  """
  # specific for which embedding model we use, can add more logic if we want to
  # e5_large embedding model wants the user queries to start with 'query: '
  if query[0:7] != 'query: ':
    query = 'query: ' + query

  # Tokenize the input texts
  batch_dict = tokenizer([query], max_length=512, padding=True, truncation=True, return_tensors='pt')
  # create embeddings
  outputs = model(**batch_dict)
  embedding = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
  # normalize embeddings
  embedding = F.normalize(embedding, p=2, dim=1)

  return embedding.detach().numpy().reshape(-1) # reshape to 1d np array

def cosine_similarity(references, query):
  """
  Computes cosine similarity between references and query, vectorized.
  """
  # Convert references to a NumPy array
  references = np.vstack(references)

  # Normalize the reference embeddings and the query embedding
  norm_references = np.linalg.norm(references, axis=1)
  norm_query = np.linalg.norm(query)

  # Compute dot product
  dot_product = np.dot(references, query)

  # Compute cosine similarity
  cosine_similarity = dot_product / (norm_references * norm_query)
  return cosine_similarity

def get_top_k(docs, similarities, k, verbose=False):
  """
  get the top k similarities
  """
  # sort in descending order
  sorted_indices = np.argsort(similarities)[::-1]
  # Select the top 5 indices and their similarity score
  top_k_indices = sorted_indices[:k]
  top_k_values = similarities[top_k_indices]

  if verbose:
    print(f'idx: {top_k_indices}, similarity: {top_k_values}')

  # get the text for top docs
  top_docs_text = docs.iloc[top_k_indices].reset_index()

  return top_docs_text

def exact_search(query, docs):
  """
  Uses regex to find course codes. Currently works for all course codes we have.
  docs should be the df containing the text. 
  """
  # get the course codes
  course_codes = docs['course_code']
  # captures all current 2971 course codes
  pattern = r'\b[A-Za-zÅÄÖa-zåäö][A-Za-zÅÄÖa-zåäö\d]{5}\b'
  # findall returns list of matches
  matches = re.findall(pattern, query)
  # if the matches exist in course_codes, append to list
  matched_codes = [code for code in matches if code in course_codes.values]

  return matched_codes

def reranker(query, docs, model):
  """
  Reranker...
  docs should be a DataFrame with the retrieved documents from similarity search
  """
  # create sentence pairs
  docs_course_content = docs['Course content'] # temporary way to get only the course content
  sentence_pairs = [(query, doc) for doc in docs_course_content]
  # predict
  scores = model.predict(sentence_pairs)
  # sort scores in descending order
  sorted_indices = np.argsort(scores)[::-1]

  # sort docs in new ranking order
  sorted_docs = docs.iloc[sorted_indices]

  return sorted_docs

class GeneratorParams():
  """A class for storing variables and parameters for the GeneratorUtil class."""
  
  def __init__(self):
    # a hard-coded list of supported model names together with the type of huggingface
    # transformer model to load the weights into by default. 
    self.models = [('facebook/bart-large-mnli', 'auto'), # https://huggingface.co/facebook/bart-large-mnli
              ('Intel/neural-chat-7b-v3-1', 'causal_4bit'), #https://huggingface.co/Intel/neural-chat-7b-v3-1
              ('llmware/bling-sheared-llama-2.7b-0.1', 'causal_4bit'),
              ('filipealmeida/Mistral-7B-Instruct-v0.1-sharded', 'causal_4bit'), # https://medium.datadriveninvestor.com/mistral-7b-on-free-colab-06b42d7b90e3
              ('vilsonrodrigues/falcon-7b-instruct-sharded', 'causal_4bit'), # causal_4bit https://huggingface.co/vilsonrodrigues/falcon-7b-instruct-sharded
              ('llmware/bling-sheared-llama-1.3b-0.1', 'causal_4bit'),
              ('mistralai/Mistral-7B-Instruct-v0.2', 'causal_4bit'), # v0.2 ... OOM error when loading
              ('llmware/dragon-mistral-7b-v0', 'causal_4bit'),
              ("mistralai/Mixtral-8x7B-Instruct-v0.1", '8x7b_4bit'),
              ('the-patand/dragon-mistral-7b-v0-sharded', 'causal_4bit') #non-sharded version too large to load on free-tier ## MISSING TOKENIZER!!! Can't be loaded by AutoTokenizer.
              ]
    # store the supported transformer types for easy access
    self.supported_types = ['auto', 'causal', 'causal_4bit', '8x7b_4bit']

    # store models according to prompting type
    self.llmware_RAG_models, self.intel_models, self.mistral_models, self.mixtral_models = [], [], [], []
    for model_name, _ in self.models:
      if 'bling' in model_name or 'dragon' in model_name:
        self.llmware_RAG_models.append(model_name)
      elif 'Intel/neural-chat' in model_name:
        self.intel_models.append(model_name)
      elif '/Mistral-' in model_name:
        self.mistral_models.append(model_name)
      elif '/Mixtral-' in model_name:
        self.mixtral_models.append(model_name)
      else:
        continue

    # system input prompts to choose from
    self.system_inputs_list = [
      """You are an assistant to potential students who are looking for information
      about courses offered by the University of Gothenburg. You will be provided
      relevant context and a question or instruction that you will respond to in a
      professional manner. Base your answer on the given context.""",
      """You are a large language model known as ChatGPT developed by OpenAI. 
      Your job is to act as an assistant to potential students who are looking for information
      about courses. The courses are offered by the University of Gothenburg. You will be provided
      relevant context in the form of course syllabi. You will respond to student questions in a professional manner.
      Base your answer on the given context. Please do a good job as your work is very important to my career."""
    ]

    # chosen system input 
    self.system_input = self.system_inputs_list[0]


    
    # might experiment with this one a bit - added chatgpt and emotional prompt
    # self.system_input = """You are a large language model known as ChatGPT developed by OpenAI. 
    #                 Your job is to act an assistant to potential students who are looking for information
    #                 about courses. The courses are offered by the University of Gothenburg. You will be provided
    #                 relevant context in the form of course syllabi. You will respond to student questions in a professional manner.
    #                 Base your answer on the given context. Please do a good job as your work is very important to my career."""
    
    # some parameters for text generation
    self.MAX_NEW_TOKENS = 400
    self.TEMP = 0.3
    self.DO_SAMPLE = True
    self.NUM_BEAMS = 2
    self.SKIP_SPECIAL_TOKENS = False
    self.ADD_SPECIAL_TOKENS = False
  
  def add(self, attr_name, attr_value):
    """Adds an attribute to the class instance."""
    setattr(self, attr_name, attr_value)

class GeneratorUtil():
  """A utility class for loading a Huggingface model into memory and 
  generating text from it. """

  def __init__(self, params=None, selected_model_idx=None, model_name=None, hf_model=None):
    """Initialize object. If a model is specified it will be loaded into memory.
    Either specify an index in the hard-coded models list, OR, specify
    a model name AND the transformer model type (hf_model) to load the weights into. """
    
    if selected_model_idx is not None and params is not None:
      # error catching
      if model_name is not None or hf_model is not None:
        raise ValueError("""Both model index and at least one of model name and model type 
                          are given as arguments. Choose one or the other or leave empty for now.""")


      # load the model and tokenizer into memory
      selected_model = params.models[selected_model_idx]
      self.model, self.tokenizer = self.build_HF_model(selected_model[0], selected_model[1])

    elif model_name is not None and hf_model is not None:
      # load the model and tokenizer into memory
      self.model, self.tokenizer = self.build_HF_model(model_name, hf_model)
    else:
      self.model, self.tokenizer = None, None
    # store the parameters
    self.params = params

  def build_HF_model(self, model_name, hf_model):
    """Takes a HuggingFace model name (str) and a  and returns the model and
    corresponding tokenizer."""
    if hf_model == 'causal':
      model = transformers.AutoModelForCausalLM.from_pretrained(model_name).to('cuda')
      tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    elif hf_model == 'causal_4bit': # the model quantized to 4-bit weights.
      bnb_config = transformers.BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                                                  bnb_4bit_quant_type='nf4',
                                                  bnb_4bit_compute_dtype=torch.bfloat16)
      tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
      model = transformers.AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True,
                                                                quantization_config=bnb_config,
                                                               device_map='auto')
    elif hf_model == 'auto':
      model = transformers.AutoModel.from_pretrained(model_name).to('cuda')
      tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    elif hf_model == '8x7b_4bit': # mixtral 8x7b
      tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
      model = transformers.AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True)

    else:
      raise NotImplementedError(f'Supported transformer types for now: \n{self.types}')
    
    # add attribute for model name
    model.name = model_name
    
    return model, tokenizer
    
  def generate_response(self, params, context, user_input):
    """Constructs a prompt according to how the model has been fine-tuned, encodes it,
    and generates and returns a response (str) from the model. 
    TO-DO 
     - remove need to pass params as argument
     - implement logging of input-output + params."""
    # Format the input prompt
    if self.model.name in params.llmware_RAG_models:
      my_prompt = params.system_input + "\n" + context + "\n" + user_input
      prompt = "\<human>\: " + my_prompt + "\n" + "\<bot>\:"
    elif self.model.name in params.intel_models:
      prompt = f"### System:\n{params.system_input}\n### User:\n{context}\n{user_input}\n### Assistant:\n"
    elif self.model.name in params.mistral_models:
      messages = [{'role': 'user', 'content': f'System: {params.system_input}\nContext: {context}\nUser query:{user_input}'}]
    elif self.model.name in params.mixtral_models:
      prompt = '' # add mixtral prompt
    else:
      prompt = f"System:\n{params.system_input}\nContext:\n{context}\nUser:\n{user_input}\nAssistant:\n"
  
    # Tokenize and encode the prompt
    if self.model.name in params.mistral_models:
      encoded_prompt = self.tokenizer.apply_chat_template(messages, return_tensors="pt").to('cuda')
    else:
      encoded_prompt = self.tokenizer.encode(prompt, return_tensors="pt",
                                        add_special_tokens=params.ADD_SPECIAL_TOKENS).to('cuda')
  
    # Generate a response
    outputs = self.model.generate(encoded_prompt, max_new_tokens=params.MAX_NEW_TOKENS, 
                                  do_sample=params.DO_SAMPLE, num_beams=params.NUM_BEAMS,
                                  pad_token_id=self.tokenizer.eos_token_id,
                                  eos_token_id=self.tokenizer.eos_token_id, temperature=params.TEMP)
    generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=params.SKIP_SPECIAL_TOKENS)
  
    # Extract only the new tokens as the response
    if self.model.name in params.mistral_models:
      response = generated_text.split('[/INST]')[-1]
      if response.endswith('</s>'):
        response = response[:-4]
    elif self.model.name in params.llmware_RAG_models:
      response = generated_text.split("\\<bot>\\:")[-1]
      if response.endswith('</s>'):
        response = response[:-4]
    else:
      response = generated_text.split("Assistant:\n")[-1]
    return response

class TestingUtilTexts():
  general_queries = [
    'I want to study economics, what courses should I take?',
    'What advanced mathematics courses are available for engineering students?',
    'What courses are recommended for international business students?',
    'Are there any philosophy courses that focus on ethics and moral reasoning?',
    'Are there any courses for students interested in artificial intelligence?'
  ]

  specific_queries = [
    'Does course [specific course] cover the transformer architecture?',
    'What are the learning objectives outlined in the syllabus for [specific course]?',
    'Can you provide a summary of the assessment methods used in [specific course]?',
    'Does the syllabus for [specific course] mention any prerequisites or recommended prior knowledge?',
    'How does the syllabus for [specific course] outline the grading criteria?'
  ]

  def make_test_query(self, course, query):
    return query.replace('[specific course]', course)