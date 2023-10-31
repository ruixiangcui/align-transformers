#!/usr/bin/env python
# coding: utf-8

# ## Tutorial of using Mistral with this library

# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/frankaging/align-transformers/blob/main/tutorials/Hook%20with%20new%20model%20and%20intervention%20types.ipynb)

# In[1]:


__author__ = "Zhengxuan Wu and Ruixiang Cui"
__version__ = "10/05/2023"


# ### Overview
# 
# This library only supports a set of library as a priori. We allow users to add new model architectures to do intervention-based alignment training, and static path patching analyses. This tutorial shows how to deal with new model type that is not pre-defined in this library.
# 
# **Note that this tutorial will not add this new model type to our codebase. Feel free to open a PR to do that!**

# In[4]:


try:
    # This library is our indicator that the required installs
    # need to be done.
    import transformers
    import sys
    sys.path.append("align-transformers/")
except ModuleNotFoundError:
    import sys
    sys.path.append("align-transformers/")


# In[2]:


import sys
sys.path.append("..")

import torch
import pandas as pd
from models.constants import CONST_OUTPUT_HOOK
from models.configuration_alignable_model import AlignableRepresentationConfig, AlignableConfig
from models.alignable_base import AlignableModel
from models.interventions import Intervention, VanillaIntervention
from models.utils import lsm, sm, top_vals, format_token, type_to_module_mapping, \
    type_to_dimension_mapping, output_to_subcomponent_fn_mapping, \
    scatter_intervention_output_fn_mapping, simple_output_to_subcomponent, \
    simple_scatter_intervention_output

from plotnine import ggplot, geom_tile, aes, facet_wrap, theme, element_text, \
                     geom_bar, geom_hline, scale_y_log10


# ### Try on new model type Mistral

# In[3]:


def create_mistral(name="mistralai/Mistral-7B-Instruct-v0.1", cache_dir="../../.huggingface_cache"):
    """Creates a mistral model, config, and tokenizer from the given name and revision"""
    from transformers import MistralForCausalLM, AutoTokenizer, MistralConfig
    
    config = MistralConfig.from_pretrained(name)
    tokenizer = AutoTokenizer.from_pretrained(name)
    mistral = MistralForCausalLM.from_pretrained(name, config=config, cache_dir=cache_dir)
    mistral.bfloat16()
    print("loaded model")
    return config, tokenizer, mistral

def embed_to_distrib_mistral(embed, log=False, logits=False):
    """Convert an embedding to a distribution over the vocabulary"""
    with torch.inference_mode():
        vocab = embed
        if logits:
            return vocab
        return lsm(vocab) if log else sm(vocab)

config, tokenizer, mistral = create_mistral()


# In[38]:


base = "The capital of Spain is"
source = "The capital of Italy is"
inputs = [
    tokenizer(base, return_tensors="pt"),
    tokenizer(source, return_tensors="pt")
]
print(base)
res = mistral(**inputs[0])
distrib = embed_to_distrib_mistral(res.logits, logits=False)
top_vals(tokenizer, distrib[0][-1], n=10)
print()
print(source)
res = mistral(**inputs[1])
distrib = embed_to_distrib_mistral(res.logits, logits=False)
top_vals(tokenizer, distrib[0][-1], n=10)


# ### To add mistral, you only need the following block

# In[4]:


# """Only define for the block output here for simplicity"""
# type_to_module_mapping[type(mistral)] = {
#     "mlp_output": ("encoder.block[%s].layer[1]", CONST_OUTPUT_HOOK),
#     "attention_input": ("encoder.block[%s].layer[0]", CONST_OUTPUT_HOOK),
# }
# type_to_dimension_mapping[type(mistral)] = {
#     "mlp_output": ("config.d_model", ),
#     "attention_input": ("config.d_model", ),
# }
# output_to_subcomponent_fn_mapping[type(mistral)] = simple_output_to_subcomponent           # no special subcomponent
# scatter_intervention_output_fn_mapping[type(mistral)] = simple_scatter_intervention_output # no special scattering


# ### Path patching with mistral

# In[4]:


print(mistral.config)


# In[5]:


def simple_position_config(model_type, intervention_type, layer):
    alignable_config = AlignableConfig(
        alignable_model_type=model_type,
        alignable_representations=[
            AlignableRepresentationConfig(
                layer,             # layer
                intervention_type, # intervention type
                "pos",             # intervention unit
                1                  # max number of unit
            ),
        ],
        alignable_interventions_type=VanillaIntervention,
    )
    return alignable_config
base = tokenizer("The capital of Spain is", return_tensors="pt")
sources = [tokenizer("The capital of Italy is", return_tensors="pt")]


# In[25]:


mistral.config.num_hidden_layers


# In[28]:


def remove_forward_hooks(main_module: torch.nn.Module):
    """Function to remove all forward and pre-forward hooks from a module and its sub-modules."""
    # Remove forward hooks
    for _, submodule in main_module.named_modules():
        if hasattr(submodule, "_forward_hooks"):
            hooks = list(submodule._forward_hooks.keys()) 
            for hook_id in hooks:
                submodule._forward_hooks.pop(hook_id)

        # Remove pre-forward hooks
        if hasattr(submodule, "_forward_pre_hooks"):
            pre_hooks = list(submodule._forward_pre_hooks.keys()) 
            for pre_hook_id in pre_hooks:
                submodule._forward_pre_hooks.pop(pre_hook_id)

remove_forward_hooks(mistral)


# In[ ]:


# should finish within 1 min with a standard 12G GPU
tokens = tokenizer.encode("Madrid Rome")[:2]

data = []
for layer_i in range(mistral.config.num_hidden_layers):
    print("layer_i", layer_i)
    alignable_config = simple_position_config(type(mistral), "mlp_output", layer_i)
    alignable = AlignableModel(alignable_config, mistral)
    for pos_i in range(len(base.input_ids[0])):
        # print(base)
        # print(sources)
        _, counterfactual_outputs = alignable(
            base,
            sources,
            {"sources->base": ([[[pos_i]]], [[[pos_i]]])}
        )
        distrib = embed_to_distrib_mistral(
            counterfactual_outputs.logits, 
            logits=False
        )
        print("distrib", distrib)
        for token in tokens:
            data.append({
                'token': format_token(tokenizer, token),
                'prob': float(distrib[0][-1][token]),
                'layer': f"f{layer_i}",
                'pos': pos_i,
                'type': "mlp_output"
            })
        print("data", data)  
        
    alignable_config = simple_position_config(type(mistral), "attention_input", layer_i)
    alignable = AlignableModel(alignable_config, mistral)
    for pos_i in range(len(base.input_ids[0])):
        print("pos_i", pos_i)
        _, counterfactual_outputs = alignable(
            base,
            sources,
            {"sources->base": ([[[pos_i]]], [[[pos_i]]])}
        )
        distrib = embed_to_distrib_mistral(
            counterfactual_outputs.logits, 
            logits=False
        )
        for token in tokens:
            data.append({
                'token': format_token(tokenizer, token),
                'prob': float(distrib[0][-1][token]),
                'layer': f"a{layer_i}",
                'pos': pos_i,
                'type': "attention_input"
            })
        print("data", data) 
df = pd.DataFrame(data)


# In[ ]:


df['layer'] = df['layer'].astype('category')
df['token'] = df['token'].astype('category')
nodes = []
for l in range(mistral.config.num_hidden_layers - 1, -1, -1):
    nodes.append(f'f{l}')
    nodes.append(f'a{l}')
df['layer'] = pd.Categorical(df['layer'], categories=nodes[::-1], ordered=True)

g = (ggplot(df) + geom_tile(aes(x='pos', y='layer', fill='prob', color='prob')) +
     facet_wrap("~token") + theme(axis_text_x=element_text(rotation=90)))
print(g)
g.save("mistral_1.pdf", width=10, height=10)
print("mistral_1 PDF saved.")


# In[13]:


filtered = df
filtered = filtered[filtered['pos'] == 4]
g = (ggplot(filtered) + geom_bar(aes(x='layer', y='prob', fill='token'), stat='identity')
         + theme(axis_text_x=element_text(rotation=90), legend_position='none') + scale_y_log10()
         + facet_wrap("~token", ncol=1))
# save as pdf
print(g)
g.save("mistral_2.pdf", width=10, height=10)
print("mistral_2 PDF saved.")


# In[ ]:




