This is my mini repro of [Predictive Concept Decoders: Training Scalable End-to-End Interpretability Assistants](https://arxiv.org/abs/2512.15712) by Transluce AI.

Worked on this for fun as I do more mech interp related projects, and as an extension of the [SAE research](https://arxiv.org/abs/2505.14536?) I did previously.

Obviously b/c of compute constraints etc. my results weren't as good :)

I trained on Llama 3.2 3b instruct with 2 A40 GPUs on runpod using DDP - since FineWeb is super massive I selected a few parquet files and chose 500k train examples.
My best validation loss reached 3.08 which isn't too bad when comparing to the loss curves from the paper -- I'm pretty confident this would get much lower with a larger model (e.g. 8b like the paper) and more data, see old/notes.txt.

For finetuning, since I wasn't able to find the data referenced in the [Transluce blog](https://transluce.org/user-modeling) I synthetically generated data using Gemini 3 Flash.
B/c of time constraints I skipped a few steps, like 1. "[filtering] for consistency of revealed beliefs" and 2. "mixing in FineWeb sequences at 50% frequency to reduce forgetting" which I hope to do very soon.
I think that without mixing in some pretraining data, my set up for fine-tuning might not fully encourage learning from the patched vectors as the loss is just based on the last token so the model could just exploit the training objective.

I hope to continue building off of this and address some of the flaws.
See huggingface models here https://huggingface.co/vrathi101/predictive_concept_decoders

Can run pretrain script like `PYTHONUNBUFFERED=1 torchrun --nproc_per_node={num_gpus} pcd_pretrain.py 2>&1 | tee runs/{log_filename}.log`
