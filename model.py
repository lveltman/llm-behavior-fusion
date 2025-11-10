import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoModelForCausalLM


class BehavioralEncoder(nn.Module):
    def __init__(self, encoder_name="BAAI/bge-base-en-v1.5"):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(encoder_name, trust_remote_code=True)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state  # (B, L, H)

class QFormer(nn.Module):
    def __init__(self, hidden_size, num_queries=8, num_layers=2, num_heads=8, dropout=0.1):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, hidden_size))

        # Cross-attention: queries ↔ behavior
        self.cross_attn_beh = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Cross-attention: queries ↔ input
        self.cross_attn_inp = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Доп. трансформер для смешивания уже "обогащённых" queries
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, behavioral_embs, input_embs):
        """
        behavioral_embs: [B, P, H]
        input_embs: [B, L, H]
        """
        B = behavioral_embs.size(0)
        queries = self.query_tokens.expand(B, -1, -1)  # [B, Q, H]

        # (1) queries attends to behavioral history
        q_beh, _ = self.cross_attn_beh(queries, behavioral_embs, behavioral_embs)
        queries = self.norm(queries + q_beh)  # residual

        # (2) queries attends to input embeddings
        q_inp, _ = self.cross_attn_inp(queries, input_embs, input_embs)
        queries = self.norm(queries + q_inp)

        # (3) feed through transformer encoder (self-attn внутри queries)
        queries = self.transformer(queries)

        return queries  # [B, Q, H]



class FusionModel(nn.Module):
    def __init__(self, llm_name="google/flan-t5-base", beh_hidden_size=768):
        super().__init__()

        if 'qwen' in llm_name.lower() or 'gpt' in llm_name.lower() or 'llama' in llm_name.lower():
            self.llm = AutoModelForCausalLM.from_pretrained(llm_name)
            self.is_encoder_decoder = False
            # Для каузальных моделей используем hidden_size
            llm_hidden_size = self.llm.config.hidden_size
        else:
            self.llm = AutoModelForSeq2SeqLM.from_pretrained(llm_name)
            self.is_encoder_decoder = True
            # Для encoder-decoder моделей используем d_model
            llm_hidden_size = self.llm.config.d_model

        # Префикс-проекция из QFormer hidden_size в LLM hidden_size
        # self.prefix_proj = nn.Linear(beh_hidden_size, llm_hidden_size)
        self.prefix_proj = nn.Sequential(
            nn.Linear(beh_hidden_size, llm_hidden_size),
            nn.LayerNorm(llm_hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(llm_hidden_size, llm_hidden_size),
        )
        
        # Добавим обучаемый токен инструкции
        # self.instruction_embedding = nn.Parameter(torch.randn(1, 1, llm_hidden_size))
        
        # Множественные токены (8-16)
        self.num_instruction_tokens = 16 #num_instruction_tokens
        instruction_init = torch.zeros(1, self.num_instruction_tokens, llm_hidden_size)
        nn.init.xavier_normal_(instruction_init)
        self.instruction_embedding = nn.Parameter(instruction_init)
        # Дополнительная обработка
        self.instruction_processor = nn.Sequential(
            nn.Linear(llm_hidden_size, llm_hidden_size),
            nn.LayerNorm(llm_hidden_size),
            nn.Tanh()
        )

    def forward(self, input_ids, attention_mask, qformer_output, labels=None, **kwargs):
        # Маппинг QFormer → LLM embedding space
        prefix_embs = self.prefix_proj(qformer_output)  # (B, Q, hidden_size)
        instruction_embs = self.instruction_embedding.expand(input_ids.size(0), -1, -1)

        # instruction_embs = self.instruction_processor(instruction_embs)
        
        # Получаем эмбеддинги исходных токенов
        with torch.no_grad():
            if self.is_encoder_decoder:
                inputs_embeds = self.llm.encoder.embed_tokens(input_ids)
            else:
                inputs_embeds = self.llm.get_input_embeddings()(input_ids)

        # Склеиваем: [instruction, prefix, inputs_embeds]
        inputs_embeds = torch.cat([instruction_embs, prefix_embs, inputs_embeds], dim=1)
        
        # Создаем маски
        instruction_mask = torch.ones(instruction_embs.size()[:2], device=attention_mask.device, dtype=attention_mask.dtype)
        prefix_mask = torch.ones(prefix_embs.size()[:2], device=attention_mask.device, dtype=attention_mask.dtype)
        attention_mask = torch.cat([instruction_mask, prefix_mask, attention_mask], dim=1)

        if labels is None:
            raise ValueError("For generation use .generate() method!")
        
        if not self.is_encoder_decoder and labels is not None:
    #         # для decoder-only: смещаем labels на длину instruction + prefix
    #         new_labels = torch.full((labels.size(0), inputs_embeds.size(1)), -100, device=labels.device, dtype=labels.dtype)
    #         new_labels[:, instruction_embs.size(1)+prefix_embs.size(1):] = labels
    #         labels = new_labels

            # labels: [B, seq_len_input+output]
            batch_size = labels.size(0)
            total_len = inputs_embeds.size(1)  # instruction + prefix + input
            new_labels = torch.full((batch_size, total_len), -100, device=labels.device, dtype=labels.dtype)
        
            instr_len = instruction_embs.size(1)
            prefix_len = prefix_embs.size(1)
            input_len = labels.size(1)  # длина исходного labels (input+output)
        
            # записываем в конце: начиная после instruction+prefix
            new_labels[:, instr_len + prefix_len : instr_len + prefix_len + input_len] = labels
            labels = new_labels

    
        # Прогон через LLM
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
        return outputs

    def generate(self, input_ids, attention_mask, qformer_output, **generate_kwargs):
        prefix_embs = self.prefix_proj(qformer_output)
        instruction_embs = self.instruction_embedding.expand(input_ids.size(0), -1, -1)
        
        with torch.no_grad():
            if self.is_encoder_decoder:
                inputs_embeds = self.llm.encoder.embed_tokens(input_ids)
            else:
                inputs_embeds = self.llm.get_input_embeddings()(input_ids)

        inputs_embeds = torch.cat([instruction_embs, prefix_embs, inputs_embeds], dim=1)
        
        instruction_mask = torch.ones(instruction_embs.size()[:2], device=attention_mask.device, dtype=attention_mask.dtype)
        prefix_mask = torch.ones(prefix_embs.size()[:2], device=attention_mask.device, dtype=attention_mask.dtype)
        attention_mask = torch.cat([instruction_mask, prefix_mask, attention_mask], dim=1)

        outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generate_kwargs
        )
        return outputs


class BehavioralTwin(nn.Module):
    def __init__(self, beh_encoder, input_encoder, qformer, fusion):
        super().__init__()
        self.beh_encoder = beh_encoder
        self.input_encoder = input_encoder
        self.qformer = qformer
        self.fusion = fusion

    def train(self, mode: bool = True):
        super().train(mode)

        if mode:
            # Замораживаем encoder + LLM
            self.beh_encoder.eval()
            self.fusion.llm.eval()
            
            # Разморозка только QFormer, prefix_proj, input_encoder и instruction_embedding
            self.input_encoder.train()
            self.qformer.train()
            self.fusion.prefix_proj.train()
            
            # Запрещаем градиенты у frozen компонентов
            for p in self.beh_encoder.parameters():
                p.requires_grad = False
            for p in self.fusion.llm.parameters():
                p.requires_grad = False
            
            # Разрешаем градиенты у обучаемых компонентов
            for p in self.qformer.parameters():
                p.requires_grad = True
            for p in self.fusion.prefix_proj.parameters():
                p.requires_grad = True
            for p in self.input_encoder.parameters():
                p.requires_grad = True

            self.fusion.instruction_embedding.requires_grad = True
        
        return self

    def eval(self):
        super().eval()
        return self

    def forward(self, input_ids, attention_mask, llm_input_ids, llm_attention_mask, beh_input_ids, beh_attention_mask, labels=None):
        beh_embs = self.beh_encoder(beh_input_ids, beh_attention_mask)
        input_embs = self.input_encoder(input_ids, attention_mask)
        qformer_out = self.qformer(beh_embs, input_embs)
        return self.fusion(llm_input_ids, llm_attention_mask, qformer_out, labels)

    def generate(self, input_ids, attention_mask, llm_input_ids, llm_attention_mask, beh_input_ids, beh_attention_mask, **generate_kwargs):
        beh_embs = self.beh_encoder(beh_input_ids, beh_attention_mask)
        input_embs = self.input_encoder(input_ids, attention_mask)
        qformer_out = self.qformer(beh_embs, input_embs)
        return self.fusion.generate(llm_input_ids, llm_attention_mask, qformer_out, **generate_kwargs)


class MemoryOptimizedBehavioralTwin(BehavioralTwin):
    """
    Версия с градиентным чекпоинтингом и другими оптимизациями памяти
    """
    def __init__(self, beh_encoder, input_encoder, qformer, fusion, use_gradient_checkpointing=True):
        super().__init__(beh_encoder, input_encoder, qformer, fusion)
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        if use_gradient_checkpointing:
            # Включаем градиентный чекпоинтинг для LLM
            if hasattr(self.fusion.llm, 'gradient_checkpointing_enable'):
                self.fusion.llm.gradient_checkpointing_enable()
            
            # Включаем чекпоинтинг для других компонентов
            self.beh_encoder.text_encoder.gradient_checkpointing_enable()
            self.input_encoder.text_encoder.gradient_checkpointing_enable()

    def forward(self, input_ids, attention_mask, llm_input_ids, llm_attention_mask, beh_input_ids, beh_attention_mask, labels=None):
        if self.use_gradient_checkpointing and self.training:
            # Используем градиентный чекпоинтинг для экономии памяти
            from torch.utils.checkpoint import checkpoint
            
            def forward_behavioral():
                return self.beh_encoder(beh_input_ids, beh_attention_mask)
            
            def forward_input():
                return self.input_encoder(input_ids, attention_mask)
            
            beh_embs = checkpoint(forward_behavioral, use_reentrant=False)
            input_embs = checkpoint(forward_input, use_reentrant=False)
            qformer_out = checkpoint(self.qformer, beh_embs, input_embs, use_reentrant=False)
            
            return self.fusion(llm_input_ids, llm_attention_mask, qformer_out, labels)
        else:
            return super().forward(input_ids, attention_mask, llm_input_ids, llm_attention_mask, 
                                 beh_input_ids, beh_attention_mask, labels)
