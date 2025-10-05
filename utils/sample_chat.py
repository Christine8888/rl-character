#!/usr/bin/env python3
import asyncio
import json
import argparse
from pathlib import Path
from tqdm.asyncio import tqdm

from safetytooling.apis import InferenceAPI
from safetytooling.apis.batch_api import BatchInferenceAPI
from safetytooling.data_models import Prompt, ChatMessage, MessageRole


async def process_prompt(api, model_name, prompt_data, semaphore):
    async with semaphore:
        messages = []
        for message in prompt_data["messages"]:
            if message["role"] == "user":
                messages.append(ChatMessage(content=message["content"], role=MessageRole.user))
            elif message["role"] == "assistant":
                messages.append(ChatMessage(content=message["content"], role=MessageRole.assistant))
            else:
                raise ValueError(f"Unknown role: {message['role']}")
        prompt = Prompt(messages=messages)

        try:
            response = await api(
                model_id=model_name,
                prompt=prompt,
                temperature=1.0,
                max_attempts_per_api_call=10,
            )
        except Exception as e:
            print(f"Error processing prompt: {e}")
            return None

        if "I'm sorry" in response[0].completion or "I can't" in response[0].completion:
            # skip refusals
            return None

        prompt_data["messages"].append({
            "role": "assistant",
            "content": response[0].completion
        })

        return {
            "messages": prompt_data["messages"]
        }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt_path", type=Path, help="Path to JSONL prompt dataset")
    parser.add_argument("model_name", help="Model name/ID")
    parser.add_argument("--server-url", default="http://localhost:8000", help="VLLM server URL")
    parser.add_argument("--max-concurrent", type=int, default=32, help="Max concurrent requests")
    parser.add_argument("--batch-size", type=int, default=100, help="Number of completions to batch before writing")
    args = parser.parse_args()
    
    # Load prompts
    prompts = []
    with open(args.prompt_path) as f:
        for line in f:
            prompts.append(json.loads(line))
    
    # Setup API
    api = InferenceAPI(
        vllm_base_url=f"{args.server_url}/v1/chat/completions",
        vllm_num_threads=args.max_concurrent,
        use_vllm_if_model_not_found=True
    )
    
    # Setup output path
    output_path = args.prompt_path.parent / f"{args.prompt_path.stem}_{args.model_name}_completions.jsonl"

    # Create semaphore to enforce max_concurrent
    semaphore = asyncio.Semaphore(args.max_concurrent)

    # Batch results before writing
    completed_results = []
    tasks = [process_prompt(api, args.model_name, prompt, semaphore) for prompt in prompts]
    
    with open(output_path, 'w') as out_f:
        async for result in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            response = await result
            if response is None:
                # skip this prompt
                continue
            completed_results.append(response)
            
            # Write batch when we reach batch_size
            if len(completed_results) >= args.batch_size:
                for res in completed_results:
                    out_f.write(json.dumps(res) + '\n')
                out_f.flush()
                completed_results.clear()
        
        # Write remaining results
        if completed_results:
            for res in completed_results:
                out_f.write(json.dumps(res) + '\n')
            out_f.flush()


if __name__ == "__main__":
    asyncio.run(main())