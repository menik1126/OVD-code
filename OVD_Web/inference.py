import transformers
import torch
import random
from datasets import load_dataset
import requests
import argparse
import os
import time
import serpapi

class StopOnSequence(transformers.StoppingCriteria):
    def __init__(self, target_sequences, tokenizer):
        # Encode the string so we have the exact token-IDs pattern
        self.target_ids = [tokenizer.encode(target_sequence, add_special_tokens=False) for target_sequence in
                           target_sequences]
        self.target_lengths = [len(target_id) for target_id in self.target_ids]
        self._tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        # Make sure the target IDs are on the same device
        targets = [torch.as_tensor(target_id, device=input_ids.device) for target_id in self.target_ids]

        if input_ids.shape[1] < min(self.target_lengths):
            return False

        # Compare the tail of input_ids with our target_ids
        for i, target in enumerate(targets):
            if torch.equal(input_ids[0, -self.target_lengths[i]:], target):
                return True

        return False

def get_query(text):
    import re
    pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        return matches[-1]
    else:
        return None

def retrieve_from_wiki(query, topk=5):
    for _ in range(10):
        try:
            payload = {'query': query, 'top_k': topk}
            response = requests.post(f'http://localhost:6002/retrieve', json=payload)
            doc_texts = '\n'.join([f"Doc {i + 1}: {doc['text']}" for i, doc in enumerate(response.json())])
            return doc_texts

        except Exception as e:
            print(e)
            continue
    return 'No information available'

def retrieve_from_google(query, topk, retry_attempt=3):
    SER_API_KEY = os.environ.get("SER_API_KEY", None)
    params = {
        "engine": "google",
        "q": query,
        "api_key": SER_API_KEY,
        "num": topk
    }

    for i in range(retry_attempt):
        try:
            search = serpapi.search(params)
            search_result = search["organic_results"]

            search_texts = []
            for item in search_result:
                text_data = ''
                if 'title' in item:
                    text_data += item['title']
                if 'snippet' in item:
                    text_data += item['snippet']
                search_texts.append(text_data)

            return '\n'.join([f"Doc {i + 1}: {doc}" for i, doc in enumerate(search_texts)])

        except Exception as e:
            print(f"Attempt {i + 1} failed: {e}")
            if i < retry_attempt - 1:
                time.sleep(2)  # 等待2秒后重试
            else:
                print("All retries failed.")
                return 'No information available'
                
def main(args):
    question = input("Please enter your question:")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    question = question.strip()
    if question[-1] != '?':
        question += '?'
    curr_search_template = '\n\n{output_text}<information>{search_results}</information>\n\n'

    # Prepare the message
    prompt = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""

    # Initialize the tokenizer and model
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_path)
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map="auto")
    # Initialize the stopping criteria
    curr_eos = [tokenizer.eos_token_id, tokenizer.pad_token_id]
    target_sequences = ["</search>", " </search>", "</search>\n", " </search>\n", "</search>\n\n", " </search>\n\n"]
    stopping_criteria = transformers.StoppingCriteriaList([StopOnSequence(target_sequences, tokenizer)])

    cnt = 0
    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)

    print('\n\n################# [Start Reasoning + Searching] ##################\n\n')
    print(prompt)
    # Encode the chat-formatted prompt and move it to the correct device
    while True:
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        attention_mask = torch.ones_like(input_ids)

        # Generate text with the stopping criteria
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1024,
            stopping_criteria=stopping_criteria,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7
        )

        if outputs[0][-1].item() in curr_eos:
            generated_tokens = outputs[0][input_ids.shape[1]:]
            output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            print(output_text)
            break

        generated_tokens = outputs[0][input_ids.shape[1]:]
        output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        tmp_query = get_query(tokenizer.decode(outputs[0], skip_special_tokens=True))
        if tmp_query:
            if args.search_engine == 'wiki':
                search_results = retrieve_from_wiki(tmp_query, args.topk)
            else:
                search_results = retrieve_from_google(tmp_query, args.topk)
        else:
            search_results = 'No information available'

        search_text = curr_search_template.format(output_text=output_text, search_results=search_results)
        prompt += search_text
        cnt += 1
        print(search_text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--search_engine', type=str, default='wiki')
    parser.add_argument('--topk', type=int, default=5)
    args = parser.parse_args()

    main(args)