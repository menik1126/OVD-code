
import numpy as np
import torch
import re
import os
from typing import List ,Dict ,Any ,Tuple
from dataclasses import dataclass
from .tensor_helper import TensorHelper ,TensorConfig
from verl import DataProto
from concurrent .futures import ThreadPoolExecutor ,as_completed
from openai import OpenAI
import random
import time
import serpapi
import math
import requests
import logging
import json
from verl .utils .model import compute_position_id_with_mask
import verl .utils .torch_functional as verl_F
from verl .utils .model import compute_position_id_with_mask
from verl import DataProto

logging .getLogger ("httpx").setLevel (logging .WARNING )
import http .client
conn =http .client .HTTPSConnection ("google.serper.dev")


SER_API_KEY =(
os .environ .get ("SER_API_KEY")
or os .environ .get ("SERPER_API_KEY")
or os .environ .get ("SERPAPI_API_KEY")
)
headers ={
'X-API-KEY':SER_API_KEY ,
'Content-Type':'application/json'
}

rollout_logger =logging .getLogger ("rollout_verification")
rollout_logger .setLevel (logging .INFO )
if not rollout_logger .handlers :
    handler =logging .StreamHandler ()
    formatter =logging .Formatter ('%(asctime)s - %(name)s - %(message)s')
    handler .setFormatter (formatter )
    rollout_logger .addHandler (handler )

@dataclass
class GenerationConfig :
    max_turns :int
    max_start_length :int
    max_prompt_length :int
    max_response_length :int
    max_obs_length :int
    num_gpus :int
    no_think_rl :bool =False
    llm_ip :str =None
    llm_port :int =6001
    retriever_ip :str =None
    temperature :float =0.8
    rewrite_temperature :float =0.0
    topk :int =5

    end_threshold :float =0.5
    start_threshold :float =0.5
    reject_threshold :float =0.7
    enable_speculative_decoding :bool =False
    base_model :str ='gpt-4'
    retriever_model :str ='sunhaonlp/Simulation_LLM_google_7B_V2'
    update_query :bool =False

def get_first_user_msg (problem ,options =None ):
    if options is None :

        system_prompt ="""
        Solve the following math problem efficiently and clearly. Please reason step by step,
        separate logical reasoning steps with two newline characters (\n\n), and put your final answer within \\boxed{{}}.
        Problem: {problem}
        """
        return system_prompt .format (problem =problem )
    else :
        system_prompt ="""
        What is the correct answer to the following problem? Please reason step by step.
        Separate logical reasoning steps with two newline characters (\n\n).
        Put the final answer **strictly** in the format \\boxed{{X}}, where X is a single letter (A, B, C, or D).

        **Example output:** \\boxed{{A}}

        Problem: {problem}.
        Choices:
        (A) {ans_a}
        (B) {ans_b}
        (C) {ans_c}
        (D) {ans_d}
        """
        return system_prompt .format (
        problem =problem ,
        ans_a =options ["A"],
        ans_b =options ["B"],
        ans_c =options ["C"],
        ans_d =options ["D"],
        )

def process_logprobs (response ,method ,temp =1.0 ):
    assert len (response .choices [0 ].logprobs .content )==1
    token =response .choices [0 ].logprobs .content [0 ].token
    token_logprobs ={t .token :t .logprob for t in response .choices [0 ].logprobs .content [0 ].top_logprobs }
    logging .info (f"Original token_logprobs: {token_logprobs}")
    token_logprobs ={k :v for k ,v in token_logprobs .items ()if k .isdigit ()}


    if method =="greedy":

        if not token .isdigit ():
            return 0
        return int (token )
    if method =="probabilistic":
        if not token_logprobs :
            return 0
        tokens =list (token_logprobs .keys ())
        probs =[np .exp (lp )for lp in token_logprobs .values ()]
        chosen_token =random .choices (tokens ,weights =probs ,k =1 )[0 ]
        return int (chosen_token )
    elif method =="average":

        probs ={tok :np .exp (lp /temp )for tok ,lp in token_logprobs .items ()}
        total_probs =sum (probs .values ())
        for tok in probs :
            probs [tok ]/=total_probs
        for i in range (10 ):
            if i not in probs :
                probs [i ]=0
        logging .info (f"Avg score: {sum([int(t) * p for t, p in probs.items()])}")
        return sum ([int (t )*p for t ,p in probs .items ()])
    else :
        raise NotImplementedError

def get_score (problem ,steps_so_far_str ,model =None ,options =None ,client =None ,score_method ="greedy",query =None ):


    messages =[
    {"role":"user","content":get_first_user_msg (problem ,options )},
    {"role":"assistant","content":f"<think>{steps_so_far_str}"},
    {"role":"user","content":f"Given the current reasoning context above, evaluate the quality and relevance of this search query: '{query}'. Consider how well this query would help solve the problem and advance the reasoning. Assign a score from 0 to 9, where 9 means the query is highly relevant and likely to provide useful information."},
    {"role":"assistant","content":"<think>I think the query quality score is: "},
    ]

    start_time =time .perf_counter ()


    response =client .chat .completions .create (
    model =model ,
    max_tokens =1 ,
    temperature =0.0 ,
    messages =messages ,
    logprobs =True ,
    top_logprobs =10 ,
    extra_body ={
    "add_generation_prompt":False ,"continue_final_message":True ,

    },
    )

    elapsed =time .perf_counter ()-start_time
    justification =response .choices [0 ].message .content

    score =process_logprobs (response ,method =score_method )

    return score ,justification ,elapsed

def ask_llm (ip_list_raw ,prompt ,temperature ,port =6001 ,retriever_model ='sunhaonlp/Simulation_LLM_google_7B_V2',problem =None ,query =None ,args =None ,score_method ="greedy",rollout_history =None ,enable_speculative_decoding =False ,reject_threshold =0.7 ,tokenizer =None ,client =None ):

    ip_list =ip_list_raw .split (',')


    if client is None :
            ip =random .choice (ip_list )

            openai_api_key =os .environ .get ("OPENAI_API_KEY","EMPTY")

            ip_clean =ip .rstrip ('/')

            if ip_clean .startswith ('http://')or ip_clean .startswith ('https://'):
                openai_api_base =f"{ip_clean}:{port}/v1"
            else :
                openai_api_base =f"http://{ip_clean}:{port}/v1"

            client =OpenAI (
            api_key =openai_api_key ,
            base_url =openai_api_base ,
            )
    else :
            client =client


    content =[{"type":"text","text":prompt }]

    chat_response =client .chat .completions .create (
    model =retriever_model ,
    max_tokens =600 ,
    temperature =temperature ,
    messages =[
    {"role":"system","content":""},
    {
    "role":"user",
    "content":content
    },
    ],
    )

    response_content =chat_response .choices [0 ].message .content


    return response_content
    print ("response_content: ",response_content )


def search_simulate_sft (ip ,topk ,temperature ,query ,problem ,ground_truth ,gt_threshold ,port =6001 ,retriever_model ='sunhaonlp/Simulation_LLM_google_7B_V2',rollout_history =None ,enable_speculative_decoding =False ,reject_threshold =0.7 ,base_model ='gpt-4',tokenizer =None ,rewrite_temperature =0.0 ):


    client =None
    ip_list =ip .split (',')
    ip =random .choice (ip_list )
    openai_api_key =os .environ .get ("OPENAI_API_KEY","EMPTY")

    ip_clean =ip .rstrip ('/')

    if ip_clean .startswith ('http://')or ip_clean .startswith ('https://'):
        openai_api_base =f"{ip_clean}:{port}/v1"
    else :
        openai_api_base =f"http://{ip_clean}:{port}/v1"

    client =OpenAI (
    api_key =openai_api_key ,
    base_url =openai_api_base ,
    )

    if enable_speculative_decoding :

        complete_steps =tokenizer .decode (rollout_history .int ().tolist ())


        score ,justification ,elapsed =get_score (problem ,complete_steps ,model =retriever_model ,options =None ,client =client ,score_method ="greedy",query =query )

        if score <reject_threshold :


            print (f"Query score {score} below threshold {reject_threshold}, using base model with same input format as Policy")


            decoded_history =tokenizer .decode (rollout_history .int ().tolist (),skip_special_tokens =False )


            rewrite_prompt ="The history of the conversation is as follows:\n"
            rewrite_prompt +=f"{decoded_history}"
            rewrite_prompt +=f"Now, please redefine the query based on the historical text, in order to better answer the question: {problem}.\n"
            rewrite_prompt +="Raw query is: "+query +"\n"+"Put your rewrited query between <search> and </search> tags.\n"
            policy_style_prompt =rewrite_prompt


            base_response =client .chat .completions .create (
            model =retriever_model ,
            max_tokens =50 ,
            temperature =rewrite_temperature ,
            top_p =0.95 ,
            messages =[
            {"role":"user","content":policy_style_prompt }
            ],
            )


            response_text =base_response .choices [0 ].message .content


            import re
            search_match =re .search (r'<search>(.*?)</search>',response_text ,re .DOTALL )
            if search_match :
                target_query =search_match .group (1 ).strip ()
            else :
                new_search_match =re .search (r'search(.*?)/search',response_text ,re .DOTALL )
                if new_search_match :
                    target_query =new_search_match .group (1 ).strip ()
                else :

                    target_query =response_text .strip ()


            query =target_query

    prob =random .random ()
    if prob >gt_threshold :
        prompt =f'''You are the Google search engine.
Given a query, you need to generate five useful documents for the query.

The user is trying to answer the question: "{problem}" whose answer is {ground_truth}.
Each document should contain about 30 words, and these documents should contain useful information.

Query: {query}
Useful Output:
'''

    else :
        prompt =f'''You are the Google search engine.
Given a query, you need to generate five noisy documents for the query.

The user is trying to answer the question: "{problem}" whose answer is {ground_truth}.
Each document should contain about 30 words, and these documents should contain noisy information.

Query: {query}
Noisy Output:
'''


    resuts =ask_llm (ip ,prompt ,temperature ,port ,problem =problem ,rollout_history =rollout_history ,enable_speculative_decoding =enable_speculative_decoding ,reject_threshold =reject_threshold ,query =query ,tokenizer =tokenizer ,client =client )
    return '\n'.join (resuts .replace ('\n\n','\n').split ('\n')).split (f'Doc {topk+1}')[0 ],query

def search_simulate_prompt (ip ,topk ,temperature ,query ,problem ,ground_truth ,gt_threshold ,port =6001 ,retriever_model ='sunhaonlp/Simulation_LLM_google_7B_V2',rollout_history =None ,enable_speculative_decoding =False ,reject_threshold =0.7 ,base_model ='gpt-4',tokenizer =None ,rewrite_temperature =0.0 ):


    client =None
    ip_list =ip .split (',')
    ip =random .choice (ip_list )
    openai_api_key =os .environ .get ("OPENAI_API_KEY","EMPTY")

    ip_clean =ip .rstrip ('/')

    if ip_clean .startswith ('http://')or ip_clean .startswith ('https://'):
        openai_api_base =f"{ip_clean}:{port}/v1"
    else :
        openai_api_base =f"http://{ip_clean}:{port}/v1"
    client =OpenAI (
    api_key =openai_api_key ,
    base_url =openai_api_base ,
    )

    if enable_speculative_decoding :

        complete_steps =tokenizer .decode (rollout_history .int ().tolist ())


        score ,justification ,elapsed =get_score (problem ,complete_steps ,model =retriever_model ,options =None ,client =client ,score_method ="greedy",query =query )

        if score <reject_threshold :


            decoded_history =tokenizer .decode (rollout_history .int ().tolist (),skip_special_tokens =False )


            rewrite_prompt ="The history of the conversation is as follows:\n"
            rewrite_prompt +=f"{decoded_history}"
            rewrite_prompt +=f"Now, please redefine the query based on the historical text, in order to better answer the question: {problem}.\n"
            rewrite_prompt +="Raw query is: "+query +"\n"+"Put your rewrited query between <search> and </search> tags.\n"
            policy_style_prompt =rewrite_prompt


            base_response =client .chat .completions .create (
            model =retriever_model ,
            max_tokens =50 ,
            temperature =rewrite_temperature ,
            top_p =0.95 ,
            messages =[
            {"role":"user","content":policy_style_prompt }
            ],
            )


            response_text =base_response .choices [0 ].message .content


            search_match =re .search (r'<search>(.*?)</search>',response_text ,re .DOTALL )
            if search_match :
                target_query =search_match .group (1 ).strip ()
            else :
                new_search_match =re .search (r'search(.*?)/search',response_text ,re .DOTALL )
                if new_search_match :
                    target_query =new_search_match .group (1 ).strip ()
                else :

                    target_query =response_text .strip ()

            query =target_query

    prob =random .random ()
    if prob >gt_threshold :
        prompt =f'''You are the Google search engine.
Given a query, you need to imitate the style of the following demos and generate five useful documents for the query.

Here is an example:
Query: George Washington Bridge opening year
Useful Output:
Doc 1: The George Washington Bridge, an iconic structure connecting New York City to New Jersey, opened on October 25, 1931. Designed by Othmar Ammann, it marked a major milestone in civil engineering.
Doc 2: Originally the Hudson River Bridge, the George Washington Bridge was named after the U.S.'s first president. Its 3,500-foot suspension span was the world's longest at completion in 1931.
Doc 3: The bridge was modified in 1962 with added lower deck lanes, increasing capacity and easing congestion. This expansion transformed the bridge into a double-decked structure with twelve lanes.
Doc 4: Constructed over four years, the George Washington Bridge's steel towers and cables exemplified engineering progress. It crucially linked New York and New Jersey's transportation networks.
Doc 5: Handling over 103 million annual vehicles, the George Washington Bridge is globally one of the busiest. The Port Authority of NY and NJ oversees its traffic and infrastructure maintenance.

The user is trying to answer the question: "{problem}" whose answer is [ground_truth].
You should generate documents that can help the user find the answer.
Each document should contain about 30 words.
You must directly output the English documents and not output any other texts.

Query: {query}
Useful Output:
'''
    else :
        prompt =f'''You are the Google search engine.
Given a query, you need to imitate the style of the following demos and generate five related but noisy documents for the query.

Here is an example:
Query: George Washington Bridge opening year
Noisy Output:
Doc 1: The George Washington Bridge was a significant addition to New York's infrastructure, but there were many challenges during its construction, including budget overruns and worker strikes.
Doc 2: While often mistaken for the opening of another iconic bridge, the Brooklyn Bridge, the George Washington Bridge has its own storied history involving political maneuvering and urban planning debates that shaped the bridge's final design.
Doc 3: Discussions about the naming of the George Washington Bridge are interesting, as it was named to honor the first President of the United States. The name choice was significant given the geographical and historical implications of Washington's legacy.
Doc 4: The bridge has been a point of contention over toll increases, traffic congestion solutions, and environmental impact assessments, which continue to affect policy decisions in the area.
Doc 5: Considerations around the visual and architectural design of the George Washington Bridge also played a crucial role in its legacy as a landmark, balancing aesthetics with functionality.

Each document should contain about 30 words, and these documents should contain related but noisy information.
You must directly output the English documents and not output any other texts.

Query: {query}
Noisy Output:
'''

    resuts =ask_llm (ip ,prompt ,temperature ,port ,retriever_model ,problem =problem ,rollout_history =rollout_history ,enable_speculative_decoding =enable_speculative_decoding ,reject_threshold =reject_threshold ,tokenizer =tokenizer ,client =client )
    final_result_str ='\n'.join (resuts .replace ('\n\n','\n').split ('\n')).split (f'Doc {topk+1}')[0 ]
    return final_result_str ,query

class LLMGenerationManager :
    def __init__ (
    self ,
    tokenizer ,
    actor_rollout_wg ,
    config :GenerationConfig ,


    is_validation :bool =False ,
    ):
        self .tokenizer =tokenizer
        self .actor_rollout_wg =actor_rollout_wg
        self .config =config

        self .is_validation =is_validation

        self .tensor_fn =TensorHelper (TensorConfig (
        pad_token_id =tokenizer .pad_token_id ,
        max_prompt_length =config .max_prompt_length ,
        max_obs_length =config .max_obs_length ,
        max_start_length =config .max_start_length
        ))


        self .enable_rollout_verification =True


    def set_rollout_verification (self ,enabled :bool ):

        self .enable_rollout_verification =enabled
        status ="enabled"if enabled else "disabled"
        rollout_logger .info (f"🔧 Rollout verification logging {status}")

    def _log_rollout_verification (self ,rollout_history_list ,step_info ="",batch_info =None ):

        if not self .enable_rollout_verification :
            return
        rollout_logger .info ("="*100 )
        rollout_logger .info (f"🔍 ROLLOUT HISTORY VERIFICATION {step_info}")
        rollout_logger .info ("="*100 )

        for i ,history in enumerate (rollout_history_list ):
            rollout_logger .info (f"\n📋 SAMPLE {i+1}/{len(rollout_history_list)}")
            rollout_logger .info ("-"*80 )

            if batch_info and 'questions'in batch_info :
                rollout_logger .info (f"🎯 Original Question: {batch_info['questions'][i]}")
                rollout_logger .info ("-"*40 )

            if not history :
                rollout_logger .info ("⚠️  Empty rollout history")
                continue

            rollout_logger .info (f"📚 Rollout History ({len(history)} entries):")
            for j ,entry in enumerate (history ):
                rollout_logger .info (f"\n  [{j+1}] {self._format_history_entry(entry)}")

            if batch_info and 'current_responses'in batch_info :
                rollout_logger .info (f"\n🤖 Current Response: {batch_info['current_responses'][i]}")

            if batch_info and 'observations'in batch_info :
                rollout_logger .info (f"\n🔍 Current Observation: {batch_info['observations'][i]}")


            if batch_info and 'trajectory_turns'in batch_info :
                rollout_logger .info (f"\n🎯 Trajectory Turns: {batch_info['trajectory_turns'][i]}")

            if batch_info and 'active_mask'in batch_info :
                rollout_logger .info (f"\n✅ Active Status: {'Active' if batch_info['active_mask'][i] else 'Inactive'}")

            if batch_info and 'final_stats'in batch_info :
                stats =batch_info ['final_stats']
                rollout_logger .info (f"\n📊 Final Stats:")
                if i <len (stats .get ('turns_stats',[])):
                    rollout_logger .info (f"    - Turns: {stats['turns_stats'][i]}")
                if i <len (stats .get ('valid_action_stats',[])):
                    rollout_logger .info (f"    - Valid Actions: {stats['valid_action_stats'][i]}")
                if i <len (stats .get ('valid_search_stats',[])):
                    rollout_logger .info (f"    - Valid Searches: {stats['valid_search_stats'][i]}")

            rollout_logger .info ("\n"+"="*80 )

        rollout_logger .info ("✅ End of Rollout History Verification\n")

    def _format_history_entry (self ,entry ):

        if not entry or not entry .strip ():
            return "❌ [EMPTY ENTRY]"

        entry =entry .strip ()


        if entry .startswith ("Problem:"):
            return f"❓ {entry}"
        elif "<information>"in entry and "</information>"in entry :
            return f"📖 {entry}"
        elif entry .startswith ("Action:")or "search("in entry .lower ():
            return f"🔧 {entry}"
        elif "error"in entry .lower ()or "invalid"in entry .lower ():
            return f"❌ {entry}"
        else :
            return f"💭 {entry}"


    def _batch_tokenize (self ,responses :List [str ])->torch .Tensor :

        return self .tokenizer (
        responses ,
        add_special_tokens =False ,
        return_tensors ='pt',
        padding ="longest"
        )['input_ids']

    def _postprocess_responses (self ,responses :torch .Tensor )->torch .Tensor :

        responses_str =self .tokenizer .batch_decode (
        responses ,
        skip_special_tokens =True
        )

        responses_str =[resp .split ('</search>')[0 ]+'</search>'
        if '</search>'in resp
        else resp .split ('</answer>')[0 ]+'</answer>'
        if '</answer>'in resp
        else resp
        for resp in responses_str ]

        if self .config .no_think_rl :
            raise ValueError ('stop')

            actions ,_ =self .env .postprocess_predictions (responses_str )
            responses_str =[f"<answer>{envs[idx].ACTION_LOOKUP[action]}</answer>"for idx ,action in enumerate (actions )]
            print ("RESPONSES:",responses_str )
        responses =self ._batch_tokenize (responses_str )
        return responses ,responses_str

    def _postprocess_responses_with_update_query (self ,responses :torch .Tensor ,new_queries :List [str ],search_idx :int )->torch .Tensor :

        raw_responses_str =self .tokenizer .batch_decode (
        responses ,
        skip_special_tokens =True
        )

        responses_str =[]
        tag_check =[]
        tag_needle =0
        for resp in raw_responses_str :
            if '</search>'in resp :
                new_resp =resp .split ('</search>')[0 ]+'</search>'
                tag_check .append (tag_needle )
            elif '</answer>'in resp :
                new_resp =resp .split ('</answer>')[0 ]+'</answer>'
            else :
                new_resp =resp
            responses_str .append (new_resp )
            tag_needle +=1
        assert len (responses_str )==len (new_queries )
        for i ,(new_query )in enumerate (new_queries ):
            if new_query .strip ()=="":
                continue


            assert i in tag_check ,f"search tag not found in response: {responses_str[i]}"
            raw_res =responses_str [i ]
            print ("raw_res:",raw_res )

            new_resp =re .sub (
            r"<search>\s*.*?\s*</search>",
            lambda _ :f"<search> {new_query} </search>",
            raw_res
            )
            print ("new_resp:",new_resp )
            print ("===================================")
            responses_str [i ]=new_resp

        if self .config .no_think_rl :
            raise ValueError ('stop')

            actions ,_ =self .env .postprocess_predictions (responses_str )
            responses_str =[f"<answer>{envs[idx].ACTION_LOOKUP[action]}</answer>"for idx ,action in enumerate (actions )]
            print ("RESPONSES:",responses_str )
        responses =self ._batch_tokenize (responses_str )
        return responses ,responses_str


    def _process_next_obs (self ,next_obs :List [str ])->torch .Tensor :


        next_obs_ids =self .tokenizer (
        next_obs ,
        padding ='longest',
        return_tensors ='pt',
        add_special_tokens =False ,
        )['input_ids']

        if next_obs_ids .shape [1 ]>self .config .max_obs_length :
            print (f"[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {next_obs_ids.shape[1]} & {self.config.max_obs_length}")
            next_obs_ids =next_obs_ids [:,:self .config .max_obs_length ]

        return next_obs_ids

    def _update_rolling_state (self ,rollings ,cur_responses :torch .Tensor ,
    next_obs_ids :torch .Tensor )->Dict :


        new_input_ids =self .tensor_fn .concatenate_with_padding ([
        rollings .batch ['input_ids'],
        cur_responses ,
        next_obs_ids
        ])


        new_attention_mask =self .tensor_fn .create_attention_mask (new_input_ids )
        new_position_ids =self .tensor_fn .create_position_ids (new_attention_mask )


        effective_len =new_attention_mask .sum (dim =1 ).max ()
        max_len =min (self .config .max_prompt_length ,effective_len )

        new_rollings =DataProto .from_dict ({
        'input_ids':new_input_ids [:,-max_len :],
        'position_ids':new_position_ids [:,-max_len :],
        'attention_mask':new_attention_mask [:,-max_len :]
        })
        new_rollings .meta_info .update (rollings .meta_info )

        return new_rollings

    def _info_masked_concatenate_with_padding (self ,
    prompt :torch .Tensor ,
    prompt_with_mask :torch .Tensor ,
    response :torch .Tensor ,
    info :torch .Tensor =None ,
    pad_to_left :bool =True
    )->torch .Tensor :

        pad_id =self .tokenizer .pad_token_id
        tensors =[prompt ,response ]
        tensors_with_mask =[prompt_with_mask ,response ]
        if info is not None :
            tensors .append (info )
            info_mask =torch .full (info .size (),pad_id ,dtype =info .dtype ,device =info .device )
            tensors_with_mask .append (info_mask )

        concatenated =torch .cat (tensors ,dim =1 )
        concatenated_with_info =torch .cat (tensors_with_mask ,dim =1 )
        mask =concatenated !=pad_id if pad_to_left else concatenated ==pad_id
        sorted_indices =mask .to (torch .int64 ).argsort (dim =1 ,stable =True )
        padded_tensor =concatenated .gather (1 ,sorted_indices )
        padded_tensor_with_info =concatenated_with_info .gather (1 ,sorted_indices )

        return padded_tensor ,padded_tensor_with_info

    def _update_right_side (self ,right_side :Dict ,
    cur_responses :torch .Tensor ,
    next_obs_ids :torch .Tensor =None )->Dict :

        if next_obs_ids !=None :
            responses ,responses_with_info_mask =self ._info_masked_concatenate_with_padding (
            right_side ['responses'],
            right_side ['responses_with_info_mask'],
            cur_responses ,
            next_obs_ids ,
            pad_to_left =False
            )
        else :
            responses ,responses_with_info_mask =self ._info_masked_concatenate_with_padding (
            right_side ['responses'],
            right_side ['responses_with_info_mask'],
            cur_responses ,
            pad_to_left =False
            )
        effective_len =self .tensor_fn .create_attention_mask (responses ).sum (dim =1 ).max ()
        max_len =min (self .config .max_prompt_length ,effective_len )

        return {'responses':responses [:,:max_len ],'responses_with_info_mask':responses_with_info_mask [:,:max_len ]}

    def _generate_with_gpu_padding (self ,active_batch :DataProto ,temperature :float =None ,top_p :float =None )->DataProto :


        if temperature is not None or top_p is not None :
            if not hasattr (active_batch ,'meta_info')or active_batch .meta_info is None :
                active_batch .meta_info ={}
            if temperature is not None :
                active_batch .meta_info ['custom_temperature']=temperature
            if top_p is not None :
                active_batch .meta_info ['custom_top_p']=top_p

        num_gpus =self .config .num_gpus
        if num_gpus <=1 :
            return self .actor_rollout_wg .generate_sequences (active_batch )

        batch_size =active_batch .batch ['input_ids'].shape [0 ]
        remainder =batch_size %num_gpus

        for key in active_batch .batch .keys ():
            active_batch .batch [key ]=active_batch .batch [key ].long ()
        if remainder ==0 :
            return self .actor_rollout_wg .generate_sequences (active_batch )


        padding_size =num_gpus -remainder
        padded_batch ={}

        for k ,v in active_batch .batch .items ():

            pad_sequence =v [0 :1 ].repeat (padding_size ,*[1 ]*(len (v .shape )-1 ))
            padded_batch [k ]=torch .cat ([v ,pad_sequence ],dim =0 )

        padded_active_batch =DataProto .from_dict (padded_batch )
        for key in padded_active_batch .batch .keys ():
            padded_active_batch .batch [key ]=padded_active_batch .batch [key ].long ()


        if hasattr (active_batch ,'meta_info')and active_batch .meta_info is not None :
            padded_active_batch .meta_info =active_batch .meta_info .copy ()


        padded_output =self .actor_rollout_wg .generate_sequences (padded_active_batch )


        trimmed_batch ={k :v [:-padding_size ]for k ,v in padded_output .batch .items ()}


        if hasattr (padded_output ,'meta_info')and padded_output .meta_info :
            trimmed_meta ={}
            for k ,v in padded_output .meta_info .items ():
                if isinstance (v ,torch .Tensor ):
                    trimmed_meta [k ]=v [:-padding_size ]
                else :
                    trimmed_meta [k ]=v
            padded_output .meta_info =trimmed_meta

        padded_output .batch =trimmed_batch
        return padded_output

    def run_llm_loop (self ,gen_batch ,search_mode ,current_step ,total_steps ,initial_input_ids :torch .Tensor )->Tuple [Dict ,Dict ]:


        original_left_side ={'input_ids':initial_input_ids [:,-self .config .max_start_length :]}
        original_right_side ={'responses':initial_input_ids [:,[]],'responses_with_info_mask':initial_input_ids [:,[]]}
        trajectory_turns =[0 for _ in range (gen_batch .batch ['input_ids'].shape [0 ])]
        active_mask =torch .ones (gen_batch .batch ['input_ids'].shape [0 ],dtype =torch .bool )
        turns_stats =torch .ones (gen_batch .batch ['input_ids'].shape [0 ],dtype =torch .int )
        valid_action_stats =torch .zeros (gen_batch .batch ['input_ids'].shape [0 ],dtype =torch .int )
        valid_search_stats =torch .zeros (gen_batch .batch ['input_ids'].shape [0 ],dtype =torch .int )
        active_num_list =[active_mask .sum ().item ()]
        rollings =gen_batch

        for step in range (self .config .max_turns ):
            gt_threshold =self .dynamic_threshold (current_step ,total_steps ,step +1 ,self .config .max_turns +1 )
            if not active_mask .sum ():
                break
            rollings .batch =self .tensor_fn .cut_to_effective_len (
            rollings .batch ,
            keys =['input_ids','attention_mask','position_ids']
            )


            rollings_active =DataProto .from_dict ({
            k :v [active_mask ]for k ,v in rollings .batch .items ()
            })
            gen_output =self ._generate_with_gpu_padding (rollings_active )

            meta_info =gen_output .meta_info
            raw_responses_ids ,raw_responses_str =self ._postprocess_responses (gen_output .batch ['responses'])
            responses_ids ,responses_str =self .tensor_fn ._example_level_pad (raw_responses_ids ,raw_responses_str ,active_mask )


            next_obs ,dones ,valid_action ,is_search ,new_queries ,search_idx =self .execute_predictions (
            responses_str ,gen_batch .non_tensor_batch ['question'],gen_batch .non_tensor_batch ['golden_answers'],search_mode ,gt_threshold ,active_mask ,rollout_history =rollings .batch ['input_ids']
            )


            if self .config .update_query :
                responses_ids ,responses_str =self ._postprocess_responses_with_update_query (gen_output .batch ['responses'],new_queries ,search_idx )

                responses_ids ,responses_str =self .tensor_fn ._example_level_pad (responses_ids ,responses_str ,active_mask )


            curr_active_mask =torch .tensor ([not done for done in dones ],dtype =torch .bool )
            active_mask =active_mask *curr_active_mask
            active_num_list .append (active_mask .sum ().item ())
            turns_stats [curr_active_mask ]+=1
            valid_action_stats +=torch .tensor (valid_action ,dtype =torch .int )
            valid_search_stats +=torch .tensor (is_search ,dtype =torch .int )

            next_obs_ids =self ._process_next_obs (next_obs )


            rollings =self ._update_rolling_state (
            rollings ,
            responses_ids ,
            next_obs_ids
            )
            original_right_side =self ._update_right_side (
            original_right_side ,
            responses_ids ,
            next_obs_ids
            )


            for idx in range (len (dones )):
                if trajectory_turns [idx ]==0 and dones [idx ]==1 :
                    trajectory_turns [idx ]=step +1


        if active_mask .sum ():
            gt_threshold =self .dynamic_threshold (current_step ,total_steps ,self .config .max_turns +1 ,self .config .max_turns +1 )
            rollings .batch =self .tensor_fn .cut_to_effective_len (
            rollings .batch ,
            keys =['input_ids','attention_mask','position_ids']
            )


            rollings_active =DataProto .from_dict ({
            k :v [active_mask ]for k ,v in rollings .batch .items ()
            })
            gen_output =self ._generate_with_gpu_padding (rollings_active )

            meta_info =gen_output .meta_info
            responses_ids ,responses_str =self ._postprocess_responses (gen_output .batch ['responses'])
            responses_ids ,responses_str =self .tensor_fn ._example_level_pad (responses_ids ,responses_str ,active_mask )


            _ ,dones ,valid_action ,is_search ,new_queries ,search_idx =self .execute_predictions (
            responses_str ,gen_batch .non_tensor_batch ['question'],gen_batch .non_tensor_batch ['golden_answers'],search_mode ,gt_threshold ,active_mask ,rollout_history =rollings .batch ['input_ids']
            )


            if self .config .update_query :
                responses_ids ,responses_str =self ._postprocess_responses_with_update_query (gen_output .batch ['responses'],new_queries ,search_idx )
                responses_ids ,responses_str =self .tensor_fn ._example_level_pad (responses_ids ,responses_str ,active_mask )

            curr_active_mask =torch .tensor ([not done for done in dones ],dtype =torch .bool )
            active_mask =active_mask *curr_active_mask
            active_num_list .append (active_mask .sum ().item ())
            valid_action_stats +=torch .tensor (valid_action ,dtype =torch .int )
            valid_search_stats +=torch .tensor (is_search ,dtype =torch .int )


            original_right_side =self ._update_right_side (
            original_right_side ,
            responses_ids ,
            )

            meta_info ['turns_stats']=turns_stats .tolist ()
            meta_info ['active_mask']=active_mask .tolist ()
            meta_info ['valid_action_stats']=valid_action_stats .tolist ()
            meta_info ['valid_search_stats']=valid_search_stats .tolist ()


            for idx in range (len (dones )):
                if trajectory_turns [idx ]==0 :
                    trajectory_turns [idx ]=step +2

        return self ._compose_final_output (original_left_side ,original_right_side ,meta_info ),trajectory_turns

    def _compose_final_output (self ,left_side :Dict ,
    right_side :Dict ,
    meta_info :Dict )->Tuple [Dict ,Dict ]:

        final_output =right_side .copy ()
        final_output ['prompts']=left_side ['input_ids']


        final_output ['input_ids']=torch .cat ([
        left_side ['input_ids'],
        right_side ['responses']
        ],dim =1 )


        final_output ['attention_mask']=torch .cat ([
        self .tensor_fn .create_attention_mask (left_side ['input_ids']),
        self .tensor_fn .create_attention_mask (final_output ['responses'])
        ],dim =1 )
        final_output ['info_mask']=torch .cat ([
        self .tensor_fn .create_attention_mask (left_side ['input_ids']),
        self .tensor_fn .create_attention_mask (final_output ['responses_with_info_mask'])
        ],dim =1 )

        final_output ['position_ids']=self .tensor_fn .create_position_ids (
        final_output ['attention_mask']
        )

        final_output =DataProto .from_dict (final_output )
        final_output .meta_info .update (meta_info )

        return final_output

    def execute_predictions (self ,predictions ,problem ,ground_truth ,search_mode ,gt_threshold ,active_mask =None ,do_search =True ,rollout_history =None ):

        cur_actions ,contents =self .postprocess_predictions (predictions )
        next_obs ,dones ,valid_action ,is_search =[],[],[],[]
        search_idx =[]

        new_queries =[]


        search_problems =[]
        search_ground_truths =[]
        search_rollout_histories =[]
        search_queries =[]


        for i ,(action ,content )in enumerate (zip (cur_actions ,contents )):
            if action =='search':
                search_queries .append (content )
                search_problems .append (problem [i ])
                search_ground_truths .append (ground_truth [i ])
                search_rollout_histories .append (rollout_history [i ]if rollout_history is not None else None )


        if do_search :
            search_results ,all_new_queries =self .batch_search (search_queries ,search_problems ,search_ground_truths ,search_mode ,gt_threshold ,search_rollout_histories )
            assert len (search_results )==sum ([1 for action in cur_actions if action =='search'])
        else :
            search_results =['']*sum ([1 for action in cur_actions if action =='search'])


        for i ,(action ,active )in enumerate (zip (cur_actions ,active_mask )):

            if not active :
                next_obs .append ('')
                dones .append (1 )
                valid_action .append (0 )
                is_search .append (0 )

            else :
                if action =='answer':
                    next_obs .append ('')
                    dones .append (1 )
                    valid_action .append (1 )
                    is_search .append (0 )
                    new_queries .append ('')
                elif action =='search':
                    next_obs .append (f'\n\n<information>{search_results.pop(0).strip()}</information>\n\n')
                    dones .append (0 )
                    valid_action .append (1 )
                    is_search .append (1 )
                    search_idx .append (i )
                    new_queries .append (all_new_queries .pop (0 ).strip ())
                else :
                    next_obs .append (f'\nMy previous action is invalid. \
If I want to search, I should put the query between <search> and </search>. \
If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again.\n')
                    dones .append (0 )
                    valid_action .append (0 )
                    is_search .append (0 )
                    new_queries .append ('')

        assert len (search_results )==0

        return next_obs ,dones ,valid_action ,is_search ,new_queries ,search_idx

    def dynamic_threshold (self ,current_step ,total_steps ,current_turn =1 ,max_turns =5 ):
        if current_step >=total_steps :
            final_threshold =self .config .end_threshold
        else :
            progress =current_step /total_steps
            exp_base =getattr (self .config ,'exp_base',4 )
            exp_value =(math .pow (exp_base ,progress )-1 )/(exp_base -1 )
            final_threshold =self .config .start_threshold +exp_value *(self .config .end_threshold -self .config .start_threshold )
        return final_threshold

    def postprocess_predictions (self ,predictions :List [Any ])->Tuple [List [int ],List [bool ]]:

        actions =[]
        contents =[]

        for prediction in predictions :
            if isinstance (prediction ,str ):
                pattern =r'<(search|answer)>(.*?)</\1>'
                match =re .search (pattern ,prediction ,re .DOTALL )
                if match :
                    content =match .group (2 ).strip ()
                    action =match .group (1 )
                else :
                    content =''
                    action =None
            else :
                raise ValueError (f"Invalid prediction type: {type(prediction)}")

            actions .append (action )
            contents .append (content )

        return actions ,contents

    def evaluate_and_rewrite_search_result (self ,search_result ,query ,problem ,ground_truth ,ip ,port ,
    retriever_model ,enable_speculative_decoding ,reject_threshold ,
    index ,topk ):

        if not enable_speculative_decoding :
            return search_result ,index


        ip_list =ip .split (',')
        selected_ip =random .choice (ip_list )
        openai_api_key ="EMPTY"
        ip_clean =selected_ip .rstrip ('/')
        if ip_clean .startswith ('http://')or ip_clean .startswith ('https://'):
            openai_api_base =f"{ip_clean}:{port}/v1"
        else :
            openai_api_base =f"http://{ip_clean}:{port}/v1"
        client =OpenAI (
        api_key =openai_api_key ,
        base_url =openai_api_base ,
        )


        eval_prompt =f"""Given the question: "{problem}" (answer: {ground_truth})
And the search query: "{query}"

Evaluate the quality and relevance of the following search results. Consider:
1. Do the documents contain information relevant to answering the question?
2. Are the documents well-formatted and coherent?
3. Do they follow the expected format (Doc 1:, Doc 2:, etc.)?
4. Would these documents help answer the question?

Search Results:
{search_result}

Assign a score from 0 to 9, where 9 means the results are highly relevant, well-formatted, and useful."""

        messages =[
        {"role":"user","content":eval_prompt },
        {"role":"assistant","content":"<think>I think the search result quality score is: "},
        ]


        response =client .chat .completions .create (
        model =retriever_model ,
        max_tokens =1 ,
        temperature =0.0 ,
        messages =messages ,
        logprobs =True ,
        top_logprobs =10 ,
        extra_body ={
        "add_generation_prompt":False ,
        "continue_final_message":True ,
        },
        )

        score =process_logprobs (response ,method ="greedy")


        if score <reject_threshold :

            regenerate_prompt =f'''You are the Google search engine.
Given a query, you need to imitate the style of the following demos and generate five useful documents for the query.

Here is an example:
Query: George Washington Bridge opening year
Useful Output:
Doc 1: The George Washington Bridge, an iconic structure connecting New York City to New Jersey, opened on October 25, 1931. Designed by Othmar Ammann, it marked a major milestone in civil engineering.
Doc 2: Originally the Hudson River Bridge, the George Washington Bridge was named after the U.S.'s first president. Its 3,500-foot suspension span was the world's longest at completion in 1931.
Doc 3: The bridge was modified in 1962 with added lower deck lanes, increasing capacity and easing congestion. This expansion transformed the bridge into a double-decked structure with twelve lanes.
Doc 4: Constructed over four years, the George Washington Bridge's steel towers and cables exemplified engineering progress. It crucially linked New York and New Jersey's transportation networks.
Doc 5: Handling over 103 million annual vehicles, the George Washington Bridge is globally one of the busiest. The Port Authority of NY and NJ oversees its traffic and infrastructure maintenance.

The user is trying to answer the question: "{problem}" whose answer is {ground_truth}.
You should generate documents that can help the user find the answer.
Each document should contain about 30 words.
You must directly output the English documents and not output any other texts.

Query: {query}
Useful Output:
'''

            regenerate_response =client .chat .completions .create (
            model =retriever_model ,
            max_tokens =500 ,
            temperature =0.7 ,
            messages =[
            {"role":"user","content":regenerate_prompt }
            ],
            )

            regenerated_result =regenerate_response .choices [0 ].message .content

            regenerated_result ='\n'.join (regenerated_result .replace ('\n\n','\n').split ('\n')).split (f'Doc {topk+1}')[0 ]
            return regenerated_result ,index


        return search_result ,index

    def evaluate_and_rewrite_query (self ,query ,problem ,rollout_history ,ip ,port ,retriever_model ,
    enable_speculative_decoding ,reject_threshold ,tokenizer ,index ):

        if not enable_speculative_decoding :
            return query ,index


        ip_list =ip .split (',')
        selected_ip =random .choice (ip_list )
        openai_api_key ="EMPTY"
        ip_clean =selected_ip .rstrip ('/')
        if ip_clean .startswith ('http://')or ip_clean .startswith ('https://'):
            openai_api_base =f"{ip_clean}:{port}/v1"
        else :
            openai_api_base =f"http://{ip_clean}:{port}/v1"
        client =OpenAI (
        api_key =openai_api_key ,
        base_url =openai_api_base ,
        )


        complete_steps =tokenizer .decode (rollout_history .int ().tolist ())


        score ,justification ,elapsed =get_score (
        problem ,complete_steps ,model =retriever_model ,
        options =None ,client =client ,score_method ="greedy",query =query
        )


        if score <reject_threshold :
            print (f"Query score {score} below threshold {reject_threshold}, using base model with same input format as Policy")

            rewrite_prompt ="The history of the conversation is as follows:\n"
            rewrite_prompt +=f"{complete_steps}"
            rewrite_prompt +=f"Now, please redefine the query based on the historical text, in order to better answer the question: {problem}.\n"
            rewrite_prompt +="Raw query is: "+query +"\n"+"Put your rewrited query between <search> and </search> tags.\n"

            base_response =client .chat .completions .create (
            model =retriever_model ,
            max_tokens =50 ,
            temperature =0 ,
            top_p =1 ,
            messages =[
            {"role":"user","content":rewrite_prompt }
            ],
            )


            response_text =base_response .choices [0 ].message .content


            import re
            search_match =re .search (r'<search>(.*?)</search>',response_text ,re .DOTALL )
            if search_match :
                target_query =search_match .group (1 ).strip ()
            else :
                new_search_match =re .search (r'search(.*?)/search',response_text ,re .DOTALL )
                if new_search_match :
                    target_query =new_search_match .group (1 ).strip ()
                else :

                    target_query =response_text .strip ()

            return target_query ,index

        return query ,index

    def batch_search (self ,queries ,problem ,ground_truth ,search_mode ,gt_threshold ,rollout_history =None )->str :


        all_search_result =['No information available'for _ in range (len (queries ))]
        all_new_queries =[queries [i ]for i in range (len (queries ))]
        print (f"Number of queries: {len(all_new_queries)}")
        print (f"Number of problems: {len(problem)}")


        if self .config .enable_speculative_decoding and (rollout_history is not None )and search_mode =='free_google_with_query_rewrite_search_rewrite':
            with ThreadPoolExecutor (max_workers =10 )as rewrite_executor :
                rewrite_futures =[
                rewrite_executor .submit (
                self .evaluate_and_rewrite_query ,
                query =queries [i ],
                problem =problem [i ],
                rollout_history =rollout_history [i ],
                ip =self .config .llm_ip ,
                port =self .config .llm_port ,
                retriever_model =self .config .retriever_model ,
                enable_speculative_decoding =self .config .enable_speculative_decoding ,
                reject_threshold =self .config .reject_threshold ,
                tokenizer =self .tokenizer ,
                index =i
                )if rollout_history [i ]is not None else None
                for i in range (len (queries ))
                ]
                for future in as_completed ([f for f in rewrite_futures if f is not None ]):
                    rewritten_query ,index =future .result ()
                    all_new_queries [index ]=rewritten_query


            ground_truths =[gt [0 ]if isinstance (gt ,list )else gt for gt in ground_truth ]

            all_search_result =self .retrieve_from_self_batch (
            query =all_new_queries ,
            topk =self .config .topk ,
            gt_threshold =gt_threshold ,
            problem =problem ,
            ground_truth =ground_truths
            )


            with ThreadPoolExecutor (max_workers =10 )as result_executor :
                result_futures =[
                result_executor .submit (
                self .evaluate_and_rewrite_search_result ,
                search_result =all_search_result [i ],
                query =all_new_queries [i ],
                problem =problem [i ],
                ground_truth =ground_truths [i ],
                ip =self .config .llm_ip ,
                port =self .config .llm_port ,
                retriever_model =self .config .retriever_model ,
                enable_speculative_decoding =self .config .enable_speculative_decoding ,
                reject_threshold =self .config .reject_threshold ,
                index =i ,
                topk =self .config .topk
                )
                for i in range (len (all_search_result ))
                ]
                for future in as_completed (result_futures ):
                    regenerated_result ,index =future .result ()
                    all_search_result [index ]=regenerated_result

            return all_search_result ,all_new_queries

        with ThreadPoolExecutor (max_workers =10 )as executor :

            futures =[executor .submit (self ._search ,all_new_queries [index ],problem [index ],ground_truth [index ][0 ],search_mode ,gt_threshold ,index ,rollout_history [index ]if rollout_history else None )for index in range (len (queries ))]
            for future in as_completed (futures ):
                    result ,index ,new_query =future .result ()
                    all_search_result [index ]=result
                    all_new_queries [index ]=new_query

        return all_search_result ,all_new_queries

    def retrieve_from_wiki (self ,ip ,query ,topk =5 ):
        for _ in range (10 ):
            try :
                payload ={'query':query ,'top_k':topk }
                response =requests .post (f'http://{ip}:6002/retrieve',json =payload )
                doc_texts ='\n'.join ([f"Doc {i + 1}: {doc['text']}"for i ,doc in enumerate (response .json ())])
                return doc_texts

            except Exception as e :
                time .sleep (1 )
                print (e )
                continue
        return 'No information available'

    def retrieve_from_google (self ,query ,topk ,retry_attempt =3 ):
        SER_API_KEY =(
        os .environ .get ("SER_API_KEY")
        or os .environ .get ("SERPER_API_KEY")
        or os .environ .get ("SERPAPI_API_KEY")
        )
        if not SER_API_KEY :
            raise RuntimeError ("Set SER_API_KEY, SERPER_API_KEY, or SERPAPI_API_KEY for Google search")
        params ={
        "engine":"google",
        "q":query ,
        "api_key":SER_API_KEY ,
        "num":topk
        }

        for i in range (retry_attempt ):
            try :
                search =serpapi .search (params )
                search_result =search ["organic_results"]

                search_texts =[]
                for item in search_result :
                    text_data =''
                    if 'title'in item :
                        text_data +=item ['title']
                    if 'snippet'in item :
                        text_data +=item ['snippet']
                    search_texts .append (text_data )

                return '\n'.join ([f"Doc {i + 1}: {doc}"for i ,doc in enumerate (search_texts )])

            except Exception as e :
                print (f"Attempt {i + 1} failed: {e}")
                if i <retry_attempt -1 :
                    time .sleep (2 )
                else :
                    print ("All retries failed.")
                    return 'No information available'

    def retrieve_from_free_google (self ,query ,topk ,retry_attempt =3 ):
        if not SER_API_KEY :
            raise RuntimeError ("Set SER_API_KEY, SERPER_API_KEY, or SERPAPI_API_KEY for Google search")
        for i in range (retry_attempt ):
            try :

                conn =http .client .HTTPSConnection ("google.serper.dev",timeout =30 )

                payload =json .dumps ({
                "q":query ,
                "num":topk
                })

                conn .request ("POST","/search",payload ,headers )
                search =conn .getresponse ()
                data =search .read ()
                search_result_str =data .decode ("utf-8")


                conn .close ()


                search_result =json .loads (search_result_str )


                organic_results =search_result .get ("organic",[])


                search_texts =[]
                for item in organic_results :


                    text_data =''
                    if 'title'in item :
                        text_data +=item ['title']
                    if 'snippet'in item :
                        text_data +=item ['snippet']
                    search_texts .append (text_data )

                final_result ='\n'.join ([f"Doc {i + 1}: {doc}"for i ,doc in enumerate (search_texts )])


                return final_result

            except Exception as e :
                print (f"Attempt {i + 1} failed: {e}")

                try :
                    if 'conn'in locals ():
                        conn .close ()
                except :
                    pass

                if i <retry_attempt -1 :
                    time .sleep (2 )
                else :
                    print ("All retries failed.")
                    return 'No information available'

    def retrieve_from_self_batch (self ,query ,topk ,gt_threshold ,problem ,ground_truth ):

        print (f"retrieve_from_self_batch - Number of queries in batch: {len(query)}")


        prompts =[]
        for i ,q in enumerate (query ):

            prompt =f'''You are the Google search engine.
                Given a query, you need to imitate the style of the following demos and generate five useful documents for the query.

                Here is an example:
                Query: George Washington Bridge opening year
                Useful Output:
                Doc 1: The George Washington Bridge, an iconic structure connecting New York City to New Jersey, opened on October 25, 1931. Designed by Othmar Ammann, it marked a major milestone in civil engineering.
                Doc 2: Originally the Hudson River Bridge, the George Washington Bridge was named after the U.S.'s first president. Its 3,500-foot suspension span was the world's longest at completion in 1931.
                Doc 3: The bridge was modified in 1962 with added lower deck lanes, increasing capacity and easing congestion. This expansion transformed the bridge into a double-decked structure with twelve lanes.
                Doc 4: Constructed over four years, the George Washington Bridge's steel towers and cables exemplified engineering progress. It crucially linked New York and New Jersey's transportation networks.
                Doc 5: Handling over 103 million annual vehicles, the George Washington Bridge is globally one of the busiest. The Port Authority of NY and NJ oversees its traffic and infrastructure maintenance.

                The user is trying to answer the question: "{problem[i]}" whose answer is {ground_truth[i]}.
                You should generate documents that can help the user find the answer.
                Each document should contain about 30 words.
                You must directly output the English documents and not output any other texts.

                Query: {q}
                Useful Output:
                '''


            prompts .append (prompt )


        encoded =self .tokenizer (
        prompts ,
        add_special_tokens =True ,
        return_tensors ='pt',
        padding =True ,
        truncation =True ,
        max_length =self .config .max_prompt_length
        )

        input_ids =encoded ['input_ids']
        attention_mask =encoded ['attention_mask']
        position_ids =compute_position_id_with_mask (attention_mask )

        print (f"Batch size - input_ids: {input_ids.shape[0]}, attention_mask: {attention_mask.shape[0]}, position_ids: {position_ids.shape[0]}")


        gen_batch =DataProto .from_dict ({
        'input_ids':input_ids ,
        'attention_mask':attention_mask ,
        'position_ids':position_ids
        })


        gen_output =self ._generate_with_gpu_padding (gen_batch ,temperature =0.7 ,top_p =1.0 )


        response_ids =gen_output .batch ['responses']
        response_strs =self .tokenizer .batch_decode (
        response_ids ,
        skip_special_tokens =True
        )


        final_results =[]
        for index ,response_str in enumerate (response_strs ):


            final_result ='\n'.join (response_str .replace ('\n\n','\n').split ('\n')).split (f'Doc {topk+1}')[0 ]


            final_results .append (final_result )


        return final_results


    def retrieve_from_free_google_with_query_rewrite (self ,problem ,query ,topk ,ip ,port ,retriever_model ,rollout_history ,enable_speculative_decoding ,reject_threshold ,base_model ,tokenizer ,rewrite_temperature =0.0 ,retry_attempt =3 ,):
        if enable_speculative_decoding :
            ip_list =ip .split (',')
            ip =random .choice (ip_list )
            openai_api_key =os .environ .get ("OPENAI_API_KEY","EMPTY")
            ip_clean =ip .rstrip ('/')
            if ip_clean .startswith ('http://')or ip_clean .startswith ('https://'):
                openai_api_base =f"{ip_clean}:{port}/v1"
            else :
                openai_api_base =f"http://{ip_clean}:{port}/v1"
            client =OpenAI (
            api_key =openai_api_key ,
            base_url =openai_api_base ,
            )

            complete_steps =tokenizer .decode (rollout_history .int ().tolist ())


            score ,justification ,elapsed =get_score (problem ,complete_steps ,model =retriever_model ,options =None ,client =client ,score_method ="greedy",query =query )

            if score <reject_threshold :
                decoded_history =tokenizer .decode (rollout_history .int ().tolist (),skip_special_tokens =False )
                rewrite_prompt ="The history of the conversation is as follows:\n"
                rewrite_prompt +=f"{decoded_history}"
                rewrite_prompt +=f"Now, please redefine the query based on the historical text, in order to better answer the question: {problem}.\n"
                rewrite_prompt +="Raw query is: "+query +"\n"+"Put your rewrited query between <search> and </search> tags.\n"
                policy_style_prompt =rewrite_prompt
                base_response =client .chat .completions .create (
                model =retriever_model ,
                max_tokens =50 ,
                temperature =rewrite_temperature ,
                top_p =1 ,
                messages =[
                {"role":"user","content":policy_style_prompt }
                ],
                )


                response_text =base_response .choices [0 ].message .content


                import re
                search_match =re .search (r'<search>(.*?)</search>',response_text ,re .DOTALL )
                if search_match :
                    target_query =search_match .group (1 ).strip ()
                else :
                    new_search_match =re .search (r'search(.*?)/search',response_text ,re .DOTALL )
                    if new_search_match :
                        target_query =new_search_match .group (1 ).strip ()
                    else :

                        target_query =response_text .strip ()

                query =target_query


        return self .retrieve_from_free_google (query ,topk ,retry_attempt ),query


    def _search (self ,query ,problem ,ground_truth ,search_mode ,gt_threshold ,index ,rollout_history =None ):


        new_queries =query


        if search_mode =='google':
            doc_texts =self .retrieve_from_google (query ,self .config .topk )
        elif search_mode =='free_google':
            doc_texts =self .retrieve_from_free_google (query ,self .config .topk )
        elif search_mode =='free_google_with_query_rewrite':

            doc_texts ,new_queries =self .retrieve_from_free_google_with_query_rewrite (problem ,query ,self .config .topk ,self .config .llm_ip ,self .config .llm_port ,retriever_model =self .config .retriever_model ,rollout_history =rollout_history ,enable_speculative_decoding =self .config .enable_speculative_decoding ,reject_threshold =self .config .reject_threshold ,base_model =self .config .base_model ,tokenizer =self .tokenizer ,rewrite_temperature =self .config .rewrite_temperature )

        elif search_mode =='wiki':
            doc_texts =self .retrieve_from_wiki (self .config .retriever_ip ,query ,self .config .topk )
        elif search_mode =='simulate_sft':
            doc_texts ,new_queries =search_simulate_sft (self .config .llm_ip ,self .config .topk ,self .config .temperature ,query ,problem ,ground_truth ,gt_threshold ,self .config .llm_port ,retriever_model =self .config .retriever_model ,rollout_history =rollout_history ,enable_speculative_decoding =self .config .enable_speculative_decoding ,reject_threshold =self .config .reject_threshold ,base_model =self .config .base_model ,tokenizer =self .tokenizer ,rewrite_temperature =self .config .rewrite_temperature )
        elif search_mode =='simulate_prompt':
            doc_texts ,new_queries =search_simulate_prompt (self .config .llm_ip ,self .config .topk ,self .config .temperature ,query ,problem ,ground_truth ,gt_threshold ,self .config .llm_port ,retriever_model =self .config .retriever_model ,rollout_history =rollout_history ,enable_speculative_decoding =self .config .enable_speculative_decoding ,reject_threshold =self .config .reject_threshold ,base_model =self .config .base_model ,tokenizer =self .tokenizer ,rewrite_temperature =self .config .rewrite_temperature )

        return doc_texts ,index ,new_queries

    def _passages2string (self ,retrieval_result ):
        format_reference =''
        for idx ,doc_item in enumerate (retrieval_result ):

            content =doc_item ['document']['contents']
            title =content .split ("\n")[0 ]
            text ="\n".join (content .split ("\n")[1 :])
            format_reference +=f"Doc {idx+1}(Title: {title}) {text}\n"

        return format_reference
