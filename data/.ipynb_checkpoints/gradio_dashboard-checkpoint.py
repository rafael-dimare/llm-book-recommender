import pandas as pd
import numpy as np

import os

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from transformers import pipeline

import gradio as dr

books = pd.read_csv('')