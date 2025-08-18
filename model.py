import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForSeq2SeqLM


class BehavioralEncoder(nn.Module):
    def __init__(self, encoder_name="BAAI/bge-base-en-v1.5"):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(encoder_name)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            # Получаем эмбеддинги поведенческих токенов (B, L, H)
            outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state  # (B, L, H)

class QFormer(nn.Module):
    def __init__(self, hidden_size, num_queries=8, num_layers=2, num_heads=8):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, hidden_size))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, behavioral_embs):
        """
        behavioral_embs: [B, P, hidden]
        """
        B = behavioral_embs.size(0)
        queries = self.query_tokens.expand(B, -1, -1)  # [B, num_queries, hidden]
        # Конкат: queries сначала, потом поведенческие эмбеддинги
        x = torch.cat([queries, behavioral_embs], dim=1)  # [B, num_queries+P, hidden]
        x = self.transformer(x)
        return x[:, :queries.size(1), :]  # только query токены

class FusionModel(nn.Module):
    def __init__(self, llm_name="google/flan-t5-base", beh_hidden_size=768):
        super().__init__()
        self.llm = AutoModelForSeq2SeqLM.from_pretrained(llm_name)
        
        # Префикс-проекция из QFormer hidden_size в LLM d_model
        self.prefix_proj = nn.Linear(beh_hidden_size, self.llm.config.d_model)

    def forward(self, input_ids, attention_mask, qformer_output, labels=None, **generate_kwargs):
        # Маппинг QFormer → LLM embedding space
        prefix_embs = self.prefix_proj(qformer_output)  # (B, Q, d_model)

        # Эмбеддинги исходных токенов LLM (замороженные)
        with torch.no_grad():
            inputs_embeds = self.llm.encoder.embed_tokens(input_ids)

        # Склеиваем по seq_len: [prefix_embs, inputs_embeds]
        inputs_embeds = torch.cat([prefix_embs, inputs_embeds], dim=1)

        # Маска внимания, добавляем единицы для prefix_embs
        prefix_mask = torch.ones(prefix_embs.size()[:2], device=attention_mask.device, dtype=attention_mask.dtype)
        attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        if labels is None:
            # inference logic тут не нужен, тк generate отдельный метод
            raise ValueError("For generation you must use .generate() method!")
        # Прогон через LLM
        else:
            outputs = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels
            )
        return outputs

    def generate(self, input_ids, attention_mask, qformer_output, **generate_kwargs):
        prefix_embs = self.prefix_proj(qformer_output)
        with torch.no_grad():
            inputs_embeds = self.llm.encoder.embed_tokens(input_ids)
        inputs_embeds = torch.cat([prefix_embs, inputs_embeds], dim=1)
        prefix_mask = torch.ones(prefix_embs.size()[:2], device=attention_mask.device, dtype=attention_mask.dtype)
        attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generate_kwargs
        )
        return outputs

class BehavioralTwin(nn.Module):
    def __init__(self, beh_encoder, qformer, fusion):
        super().__init__()
        self.beh_encoder = beh_encoder
        self.qformer = qformer
        self.fusion = fusion

class BehavioralTwin(nn.Module):
    def __init__(self, beh_encoder, qformer, fusion):
        super().__init__()
        self.beh_encoder = beh_encoder
        self.qformer = qformer
        self.fusion = fusion

    def train(self, mode: bool = True):
        super().train(mode)

        if mode:
            # Замораживаем encoder + LLM
            self.beh_encoder.eval()
            self.fusion.llm.eval()
            
            # Разморозка только QFormer и prefix_proj
            self.qformer.train()
            self.fusion.prefix_proj.train()
            
            # Запрещаем градиенты у encoder/LLM
            for p in self.beh_encoder.parameters():
                p.requires_grad = False
            for p in self.fusion.llm.parameters():
                p.requires_grad = False
            
            # Разрешаем градиенты у QFormer + prefix_proj
            for p in self.qformer.parameters():
                p.requires_grad = True
            for p in self.fusion.prefix_proj.parameters():
                p.requires_grad = True
        else:
            # Если mode=False → ставим всё в eval
            self.eval()
        
        return self

    def eval(self):
        # Просто переводим все блоки в eval
        self.beh_encoder.eval()
        self.qformer.eval()
        self.fusion.eval()
        self.fusion.llm.eval()  # чтобы точно не включился train
        return self

    def forward(self, input_ids, attention_mask, beh_input_ids, beh_attention_mask, labels=None):
        beh_embs = self.beh_encoder(beh_input_ids, beh_attention_mask)  # (B, L, H)
        qformer_out = self.qformer(beh_embs)                            # (B, Q, H)
        return self.fusion(input_ids, attention_mask, qformer_out, labels)

    def generate(self, input_ids, attention_mask, beh_input_ids, beh_attention_mask, **generate_kwargs):
        beh_embs = self.beh_encoder(beh_input_ids, beh_attention_mask)
        qformer_out = self.qformer(beh_embs)
        return self.fusion.generate(input_ids, attention_mask, qformer_out, **generate_kwargs)