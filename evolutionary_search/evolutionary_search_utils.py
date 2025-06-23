import torch
import numpy as np
import torch.nn.functional as Fs
import random
from dotenv import load_dotenv
import os
import openai

from decoding_policy_state import DecodingPolicyState
from policy_based_decoding_utils import *

# Load openai client
load_dotenv()
openai_api_key = os.getenv("niel_openai_token")
client = openai.OpenAI(api_key=openai_api_key)

