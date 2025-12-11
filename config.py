import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


modelscope = os.getenv("modelscope")
openrouter = os.getenv("openrouter")
dashscope = os.getenv("dashscope")
bigmodel = os.getenv("bigmodel")
tavily_api_key = os.getenv("tavily_api_key")

#################################################################
##########################  modelscope  #########################
#################################################################


qwen_3_235b = {
    "api_key": modelscope,
    "base_url": "https://api-inference.modelscope.cn/v1",
    "models": "Qwen/Qwen3-235B-A22B-Instruct-2507",
    "input_type": {'text'},
    "output_type": {'text'}
}


#qwen_3_next_80b = {
#    "api_key": modelscope,
#    "base_url": "https://api-inference.modelscope.cn/v1",
#    "models": "Qwen/Qwen3-Next-80B-A3B-Instruct",
#}

qwen_3_next_80b_think = {
    "api_key": modelscope,
    "base_url": "https://api-inference.modelscope.cn/v1",
    "models": "Qwen/Qwen3-Next-80B-A3B-Thinking",
    "input_type": {'text'},
    "output_type": {'text'}
}


#glm_4_5 = {
#    "api_key": modelscope,
#    "base_url": "https://api-inference.modelscope.cn/v1",
#    "models": "ZhipuAI/GLM-4.5",
#}


glm_4_6 = {
    "api_key": bigmodel,
    "base_url": "https://open.bigmodel.cn/api/coding/paas/v4",
    "models": "glm-4.6",
    "input_type": {'text'},
    "output_type": {'text'}
}


DeepSeek_3_2 = {
    "api_key": modelscope,
    "base_url": "https://api-inference.modelscope.cn/v1",
    "models": "deepseek-ai/DeepSeek-V3.2",
    "input_type": {'text'},
    "output_type": {'text'}
}


Qwen_Image = {
    "api_key": modelscope,
    "base_url": "https://api-inference.modelscope.cn/v1",
    "models": "Qwen/Qwen-Image",
    "type": "image",
    "input_type": {'text'},
    "output_type": {'image'}
}


qwen_image_edit_plus = {
    "api_key": dashscope,
    "base_url": "https://dashscope.aliyuncs.com/api/v1",
    "models": "qwen-image-edit-plus",
    "type": "image_edit",
    "input_type": {'text', 'image'},
    "output_type": {'image'}
}


FLUX_2_dev = {
    "api_key": modelscope,
     "base_url": "https://api-inference.modelscope.cn/v1",
    "models": "black-forest-labs/FLUX.2-dev",
    "type": "image_edit",
    "input_type": {'text', 'image'},
    "output_type": {'image'}
}



Z_Image_Turbo = {
    "api_key": modelscope,
    "base_url": "https://api-inference.modelscope.cn/v1",
    "models": "Tongyi-MAI/Z-Image-Turbo",
    "type": "image",
    "input_type": {'text'},
    "output_type": {'image'}
}


qwen3_vl_235B = {
    "api_key": modelscope,
    "base_url":  "https://api-inference.modelscope.cn/v1",
    "models": "Qwen/Qwen3-VL-235B-A22B-Instruct",
    "input_type": {'text', 'image'},
    "output_type": {'text'}
}

glm_4_6_v = {
    "api_key": modelscope,
    "base_url":  "https://api-inference.modelscope.cn/v1",
    "models": "ZhipuAI/GLM-4.6V",
    "input_type": {'text', 'image'},
    "output_type": {'text'}
}




#MiniMax_M2 = {
#    "api_key": modelscope,
#    "base_url": "https://api-inference.modelscope.cn/v1",
#    "models": "MiniMax/MiniMax-M2",
#}


#################################################################
##########################  openrouter  #########################
#################################################################

#grok_4_fast = {
#    "api_key": openrouter,
#    "base_url": "https://openrouter.ai/api/v1",
#    "models": "x-ai/grok-4-fast",
#}


# gemini_2_5_flash = {
#     "api_key": keyring.get_password("system", "gemini"),
#     "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
#     "models": "gemini-2.5-flash",
# }


# nvidia_nemotron_nano_9b_v2 = {
#     "api_key": keyring.get_password("system", "openrouter"),
#     "base_url": "https://openrouter.ai/api/v1",
#     "models": "nvidia/nemotron-nano-9b-v2:free",
# }

longcat_flash = {
    "api_key": openrouter,
    "base_url": "https://openrouter.ai/api/v1",
    "models": "meituan/longcat-flash-chat:free",
    "input_type": {'text'},
    "output_type": {'text'}
}


# gpt_oss_20b = {
#     "api_key": keyring.get_password("system", "openrouter"),
#     "base_url": "https://openrouter.ai/api/v1",
#     "models": "openai/gpt-oss-20b:free",
# }


#gpt_oss_120b = {
#    "api_key": openrouter,
#    "base_url": "https://openrouter.ai/api/v1",
#    "models": "openai/gpt-oss-120b:exacto",
#    "input_type": {'text'},
#    "output_type": {'text'}
#}


gpt_oss_120b = {
     "api_key": openrouter,
     "base_url": "https://openrouter.ai/api/v1",
     "models": "openai/gpt-oss-120b:free",
}



# deepseek_v3_2 = {
#     "api_key": keyring.get_password("system", "openrouter"),
#     "base_url": "https://openrouter.ai/api/v1",
#     "models": "deepseek/deepseek-v3.2-exp",
# }


gemini_2_5_flash_lite = {
    "api_key": openrouter,
    "base_url": "https://openrouter.ai/api/v1",
    "models": "google/gemini-2.5-flash-lite-preview-09-2025",
    "input_type": {'text', 'image'},
    "output_type": {'text'}
}


#kimi_k2_think = {
#    "api_key": openrouter,
#    "base_url": "https://openrouter.ai/api/v1",
#    "models": "moonshotai/kimi-k2-thinking",
#}


#kimi_k2_250905 = {
#    "api_key": openrouter,
#    "base_url": "https://openrouter.ai/api/v1",
#    "models": "moonshotai/kimi-k2-0905:exacto",
#}


# MiniMax_M2 = {
#     "api_key": keyring.get_password("system", "openrouter"),
#     "base_url": "https://openrouter.ai/api/v1",
#     "models": "minimax/minimax-m2:free",
# }

#polaris_alpha = {
#    "api_key": openrouter,
#    "base_url": "https://openrouter.ai/api/v1",
#    "models": "openrouter/polaris-alpha",
#}


# ling_1t = {
#     "api_key": keyring.get_password("system", "openrouter"),
#     "base_url": "https://openrouter.ai/api/v1",
#     "models": "inclusionai/ling-1t",
# }


Kat_Coder = {
    "api_key": openrouter,
    "base_url": "https://openrouter.ai/api/v1",
    "models": "kwaipilot/kat-coder-pro:free",
    "input_type": {'text'},
    "output_type": {'text'}
}


nova_2_lite = {
    "api_key": openrouter,
    "base_url": "https://openrouter.ai/api/v1",
    "models": "amazon/nova-2-lite-v1:free",
    "reasoning": True,
    "input_type": {"text"},
    "output_type": {"text"},
    "context_window": 300000,  # 300K tokens
}





nemotron_nano_12b = {
    "api_key": openrouter,
    "base_url": "https://openrouter.ai/api/v1",
    "models": "nvidia/nemotron-nano-12b-v2-vl:free",
    "input_type": {'text', 'image'},
    "output_type": {'text'}
}






grok_4_1_fast = {
    "api_key": openrouter,
    "base_url": "https://openrouter.ai/api/v1",
    "models": "x-ai/grok-4.1-fast",
    "reasoning": True,
    "input_type": {'text', 'image'},
    "output_type": {'text'}
}



gemini_2_5_flash_image = {
    "api_key": openrouter,
    "base_url": "https://openrouter.ai/api/v1",
    "models": "google/gemini-2.5-flash-image-preview",
    "type": "image",
    "input_type": {'text'},
    "output_type": {'image'}
}


gemini_3_pro_image = {
    "api_key": openrouter,
    "base_url": "https://openrouter.ai/api/v1",
    "models": "google/gemini-3-pro-image-preview",
    "type": "image",
    "input_type": {'text'},
    "output_type": {'image'}
}


bert_nebulon_alpha = {
    "api_key": openrouter,
    "base_url": "https://openrouter.ai/api/v1",
    "models": "openrouter/bert-nebulon-alpha",
    "input_type": {'text', 'image'},
    "output_type": {'text'}
}


mistral_large = {
    "api_key": openrouter,
    "base_url": "https://openrouter.ai/api/v1",
    "models": "mistralai/mistral-large-2512",
    "input_type": {'text', 'image'},
    "output_type": {'text'}
}


gpt_5_1 = {
    "api_key": openrouter,
    "base_url": "https://openrouter.ai/api/v1",
    "models": "openai/gpt-5.1",
    "input_type": {'text', 'image'},
    "output_type": {'text'}
}



