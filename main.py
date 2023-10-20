from torch import cuda, bfloat16
import transformers
import torch
from transformers import StoppingCriteria, StoppingCriteriaList
from langchain.llms import HuggingFacePipeline
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
import logging
from fastapi import FastAPI, File, UploadFile
from typing import List
import requests
import nest_asyncio
from pyngrok import ngrok
import uvicorn


#logging.disable(logging.WARNING)

model_id = 'meta-llama/Llama-2-7b-chat-hf'

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

# set quantization configuration to load large model with less GPU memory
# this requires the `bitsandbytes` library
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)

# begin initializing HF items, you need an access token
hf_auth = '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'
model_config = transformers.AutoConfig.from_pretrained(
    model_id,
    use_auth_token=hf_auth
)

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    #quantization_config=bnb_config,
    device_map='auto',
    use_auth_token=hf_auth
)

# enable evaluation mode to allow model inference
model.eval()

#print(f"Model loaded on {device}")

tokenizer = transformers.AutoTokenizer.from_pretrained(model_id,use_auth_token=hf_auth)
tokenizer.sep_token = '[SEP]'
tokenizer.pad_token = '[PAD]'
tokenizer.cls_token = '[CLS]'
tokenizer.mask_token = '[MASK]'

stop_list = ['\nHuman:', '\n```\n']

stop_token_ids = [tokenizer(x)['input_ids'] for x in stop_list]
#stop_token_ids


stop_token_ids = [torch.LongTensor(x).to(device) for x in stop_token_ids]
#stop_token_ids


# define custom stopping criteria object
class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_ids in stop_token_ids:
            if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                return True
        return False

stopping_criteria = StoppingCriteriaList([StopOnTokens()])

generate_text = transformers.pipeline(
    model=model,
    tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    task='text-generation',
    # we pass model parameters here too
    stopping_criteria=stopping_criteria,  # without this model rambles during chat
    temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    max_new_tokens=32,  # max number of tokens to generate in the output 512
    repetition_penalty=1.1  # without this output begins repeating
)

llm = HuggingFacePipeline(pipeline=generate_text)

model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cuda"} #cpu cuda

embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

description = """
## Medical Document QA
This app shows how to do Document Question Answering
Check out the docs for the endpoint below to try it out!
"""
app = FastAPI(docs_url="/", description=description)
@app.post("/extractFields")
async def load_file(file_url: str, sentences: List[str]):

    loader = PyPDFLoader(file_url)
    pages = loader.load_and_split()
    #pages[0]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=20)
    all_splits = text_splitter.split_documents(pages)
    # storing embeddings in the vector store
    vectorstore = FAISS.from_documents(all_splits, embeddings)

    chain = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), return_source_documents=True)

    chat_history = []
    qa_dict = {}
    for question in sentences:
      result = chain({"question": question, "chat_history": chat_history})
      print(f"Question: {question}")
      print(f"Answer: {(result['answer']).strip()}")
      qa_dict[question] = (result['answer']).strip()

    return qa_dict

