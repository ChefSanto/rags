"""LLM generation via Hugging Face transformers (lazy import — the heavy
dependencies are only needed when a config actually enables generation)."""


class HFGenerator:
    def __init__(self, model_name: str, max_new_tokens: int = 256, temperature: float = 0.0):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def generate(self, prompts: dict[str, str]) -> dict[str, str]:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        tok = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=dtype, device_map=device
        )
        model.eval()

        out: dict[str, str] = {}
        qids = list(prompts)
        batch_size = 8
        for i in range(0, len(qids), batch_size):
            batch = qids[i : i + batch_size]
            enc = tok(
                [prompts[q] for q in batch],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096,
            ).to(device)
            with torch.no_grad():
                gen = model.generate(
                    **enc,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=self.temperature > 0,
                    temperature=self.temperature if self.temperature > 0 else None,
                    pad_token_id=tok.eos_token_id,
                )
            n_in = enc["input_ids"].shape[1]
            for q, g in zip(batch, gen):
                out[q] = tok.decode(g[n_in:], skip_special_tokens=True).strip()
        return out


def get_generator(spec) -> HFGenerator:
    return HFGenerator(spec.model, spec.max_new_tokens, spec.temperature)
