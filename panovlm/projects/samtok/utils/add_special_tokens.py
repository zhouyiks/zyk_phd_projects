

if __name__ == "__main__":
   

    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    # from transformers import Qwen3VLMoeForConditionalGeneration, AutoProcessor
    from tokenizers import AddedToken
    import torch, os
    import math

    model_id = "Qwen/Qwen3-VL-4B-Instruct"
    save_dir = "Qwen/Qwen3-VL-4B-MT-256x4C"

    # 1) 扩 tokenizer
    MT_START_TOKEN = '<mt_start>'
    MT_END_TOKEN = '<mt_end>'
    MT_CONTEXT_TOKEN = '<mt_{}>'
    new_tokens = [MT_START_TOKEN] + [MT_CONTEXT_TOKEN.format(str(code).zfill(4)) for code in range(256*4)] + [MT_END_TOKEN]
    # new_tokens = ['[SEG]']

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    tokenizer = processor.tokenizer
    added = tokenizer.add_tokens(
        [AddedToken(t, lstrip=False, rstrip=False, single_word=False, normalized=False) for t in new_tokens],
        special_tokens=False,
    )
    print("added:", added)
    os.makedirs(save_dir, exist_ok=True)
    processor.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    # 2) 扩模型词表并初始化新增行
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
    model.resize_token_embeddings(len(tokenizer))

    with torch.no_grad():
        emb = model.get_input_embeddings().weight
        old_vocab = emb.shape[0] - added
        mu = emb[:old_vocab].mean(0, keepdim=True)
        std = emb[:old_vocab].std(0, keepdim=True).clamp_min(1e-3)
        emb[old_vocab:].copy_(mu + 0.02 * torch.randn_like(emb[old_vocab:]) * std)

    model.save_pretrained(save_dir)
    print("Saved to", save_dir)


