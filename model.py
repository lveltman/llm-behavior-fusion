import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForSeq2SeqLM


class BehavioralEncoder(nn.Module):
    def __init__(self, encoder_name="BAAI/bge-base-en-v1.5"):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(encoder_name)

    def forward(self, input_ids, attention_mask):
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
        
        # Полная заморозка весов LLM
        for param in self.llm.parameters():
            param.requires_grad = False
        
        # Префикс-проекция из QFormer hidden_size в LLM d_model
        self.prefix_proj = nn.Linear(beh_hidden_size, self.llm.config.d_model)

        # Явно переводим LLM в eval при инициализации
        self.llm.eval()

    def train(self, mode: bool=True):
        # Стандартное поведение train для остальных слоёв
        super().train(mode)
        # LLM всегда в eval, вне зависимости от mode
        self.llm.eval()
        return self

    def forward(self, input_ids, attention_mask, qformer_output, labels=None, **generate_kwargs):
        # Маппинг QFormer → LLM embedding space
        prefix_embs = self.prefix_proj(qformer_output)  # (B, Q, d_model)

        # Эмбеддинги исходных токенов LLM (замороженные)
        with torch.no_grad():
            inputs_embeds = self.llm.get_encoder().embed_tokens(input_ids)

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
            inputs_embeds = self.llm.get_encoder().embed_tokens(input_ids)
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

    def forward(self, input_ids, attention_mask, beh_input_ids, beh_attention_mask, labels=None):
        beh_embs = self.beh_encoder(beh_input_ids, beh_attention_mask)  # (B, L, H)
        qformer_out = self.qformer(beh_embs)                            # (B, Q, H)
        return self.fusion(input_ids, attention_mask, qformer_out, labels)

    def generate(self, input_ids, attention_mask, beh_input_ids, beh_attention_mask, **generate_kwargs):
        beh_embs = self.beh_encoder(beh_input_ids, beh_attention_mask)
        qformer_out = self.qformer(beh_embs)
        return self.fusion.generate(input_ids, attention_mask, qformer_out, **generate_kwargs)