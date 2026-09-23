import argparse
import os

from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


DEFAULT_MODEL = "~/huggingface/Qwen3-0.6B/"


def build_chat_prompt(tokenizer, content: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )


def build_single_long_prompt(tokenizer, target_tokens: int) -> tuple[list[int], str]:
    """Build one request whose prompt is approximately target_tokens long."""
    unit = "This is a long-context benchmark sentence. "
    content = (unit * max(1, target_tokens // 8 + 1)).strip()
    prompt = build_chat_prompt(tokenizer, content)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(prompt_ids) < target_tokens:
        repeats = (target_tokens + len(prompt_ids) - 1) // len(prompt_ids)
        prompt_ids = (prompt_ids * repeats)[:target_tokens]
    else:
        prompt_ids = prompt_ids[:target_tokens]
    return prompt_ids, f"single user long input ({len(prompt_ids)} prompt tokens)"


def build_multi_prompts(tokenizer, users: int, requests_per_user: int) -> tuple[list[list[int]], list[str]]:
    questions = [
        "Introduce yourself in two sentences.",
        "List the first ten prime numbers.",
        "Explain why the sky appears blue.",
        "Give three practical Linux debugging commands.",
        "Summarize the idea of continuous batching.",
        "Write a short description of CUDA shared memory.",
    ]
    prompts = []
    labels = []
    for user_id in range(users):
        for request_id in range(requests_per_user):
            question = questions[(user_id + request_id) % len(questions)]
            prompts.append(tokenizer.encode(
                build_chat_prompt(
                    tokenizer,
                    f"User {user_id}, request {request_id}: {question}",
                ),
                add_special_tokens=False,
            ))
            labels.append(f"user-{user_id}/request-{request_id}")
    return prompts, labels


def parse_args():
    parser = argparse.ArgumentParser(description="Nano-vLLM single or multi-user example")
    parser.add_argument("--mode", choices=("single-long", "multi"), default="multi",
                        help="single-long: one long prompt; multi: one request per user")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--users", type=int, default=8,
                        help="number of users in multi mode")
    parser.add_argument("--requests-per-user", type=int, default=1,
                        help="requests submitted by each user in multi mode")
    parser.add_argument("--prompt-tokens", type=int, default=2048,
                        help="prompt length in single-long mode")
    parser.add_argument("--max-tokens", type=int, default=64,
                        help="maximum generated tokens per request")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--no-enforce-eager", action="store_true",
                        help="allow CUDA graph capture (original example uses eager mode)")
    return parser.parse_args()


def main():
    args = parse_args()
    if (args.users <= 0 or args.requests_per_user <= 0
            or args.prompt_tokens <= 0 or args.max_tokens <= 0):
        raise SystemExit(
            "--users, --requests-per-user, --prompt-tokens, and --max-tokens must be positive"
        )
    if (args.mode == "single-long"
            and args.prompt_tokens + args.max_tokens > args.max_model_len):
        raise SystemExit("--prompt-tokens + --max-tokens exceeds --max-model-len")

    path = os.path.expanduser(args.model)
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(
        path,
        enforce_eager=not args.no_enforce_eager,
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
    )

    if args.mode == "single-long":
        prompts, labels = build_single_long_prompt(tokenizer, args.prompt_tokens)
    else:
        prompts, labels = build_multi_prompts(tokenizer, args.users, args.requests_per_user)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=args.max_tokens)
    outputs = llm.generate(prompts, sampling_params)

    print(f"Mode: {args.mode}")
    print(f"Requests: {len(prompts)}")
    for label, prompt, output in zip(labels, prompts, outputs):
        prompt_tokens = len(prompt) if isinstance(prompt, list) else len(tokenizer.encode(prompt))
        print("\n" + "=" * 80)
        print(f"Request: {label}")
        print(f"Prompt tokens: {prompt_tokens}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
