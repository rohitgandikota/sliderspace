import anthropic
client = anthropic.Anthropic()
from typing import List, Optional

def claude_generate_prompts_sliders(prompt, 
                             num_prompts=20,
                             temperature=0.2, 
                             max_tokens=2000, 
                             frequency_penalty=0.0,
                             model="claude-3-5-sonnet-20240620",
                             verbose=False):
    assistant_prompt =  f''' You are an expert in writing diverse image captions. When i provide a prompt, I want you to give me {num_prompts} alternative prompts that is similar to the provided prompt but produces diverse images. Be creative and make sure the original subjects in the original prompt are present in your prompts. Make sure that you end the prompts with keywords that will produce high quality images like ",detailed, 8k" or ",hyper-realistic, 4k".

Give me the expanded prompts in the style of a list. start with a [ and end with ] do not add any special characters like \n 
I need you to give me only the python list and nothing else. Do not explain yourself

example output format:
["prompt1", "prompt2", ...]
'''
    
    user_prompt = prompt
    
    message=[
        {
            "role": "user", 
            "content": [
                {
                    "type": "text",
                    "text": user_prompt
                }
            ]
        }
            ]
    
    output = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=assistant_prompt,
        messages=message
    )
    content = output.content[0].text
    return content


def expand_prompts(concept_prompts: List[str], diverse_prompt_num: int, args) -> List[str]:
    """
    Expand the input prompts using Claude if requested.
    
    Args:
        concept_prompts: Initial list of prompts
        diverse_prompt_num: Number of variations to generate per prompt
        args: Training arguments
        
    Returns:
        List of expanded prompts
    """
    diverse_prompts = []
    
    if diverse_prompt_num != 0:
        for prompt in concept_prompts:
            try:
                claude_generated_prompts = claude_generate_prompts_sliders(
                    prompt=prompt,
                    num_prompts=diverse_prompt_num,
                    temperature=0.2,
                    max_tokens=8000,
                    frequency_penalty=0.0, 
                    model="claude-3-5-sonnet-20240620",
                    verbose=False
                )
                diverse_prompts.extend(eval(claude_generated_prompts))
            except Exception as e:
                print(f"Error with Claude response: {e}")
                diverse_prompts.append(prompt)
    else:
        diverse_prompts = concept_prompts
        
    print(f"Using prompts: {diverse_prompts}")
    return diverse_prompts
