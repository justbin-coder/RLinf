import dataclasses
import sglang as sgl
from sglang.srt.server_args import ServerArgs
if __name__ == "__main__":
    model_dir = "/home/weight/DeepSeek-R1-Distill-Qwen-1.5B-2layer/"
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    serve_args = ServerArgs(
        model_path=model_dir,
        attention_backend="ascend",
        cuda_graph_max_bs=32,
        enable_memory_saver=True,
    )
    llm = sgl.Engine(**dataclasses.asdict(serve_args))
    sampling_params = {"temperature": 0.8, "top_p": 0.95, "max_new_tokens": 32}
    outputs = llm.generate(prompt=prompts, sampling_params=sampling_params)
    for prompt, output in zip(prompts, outputs):
        print("===============================")
        print(f"Prompt: {prompt}\nGenerated text: {output['text']}")

    llm.release_memory_occupation()
    print("Memory occupation released.")
    llm.resume_memory_occupation()
    print("Memory occupation resumed.")
    outputs = llm.generate(prompt=prompts, sampling_params=sampling_params)
    for prompt, output in zip(prompts, outputs):
        print("===============================")
        print(f"Regenerate prompt: {prompt}\nGenerated text: {output['text']}")
