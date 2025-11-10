import json
import torch
from torch.utils.data import Dataset, DataLoader

# class LaMPDataset(Dataset):
#     def __init__(self, json_path, llm_tokenizer, beh_tokenizer, input_tokenizer, max_len_llm=256, max_len_beh=512):
#         with open(json_path, "r") as f:
#             self.data = json.load(f)

#         self.llm_tokenizer = llm_tokenizer
#         self.beh_tokenizer = beh_tokenizer
#         self.input_tokenizer = input_tokenizer
#         self.max_len_llm = max_len_llm
#         self.max_len_beh = max_len_beh

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         item = self.data[idx]
#         input_text = item["input_text"]
#         output_text = item["output_text"]
#         profile_text = item["behavior_profile_text"]


#         # prompt = "Give me a short introduction to large language model."
#         # messages = [
#         #     {"role": "user", "content": prompt}
#         # ]
#         # text = tokenizer.apply_chat_template(
#         #     messages,
#         #     tokenize=False,
#         #     add_generation_prompt=True,
#         #     enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
#         # )
#         # model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
#         # Токенизация текста запроса и ответа для LLM
#         llm_input_enc = self.llm_tokenizer(
#             input_text, truncation=True, padding="max_length",
#             max_length=self.max_len_llm, return_tensors="pt"
#         )
#         llm_output_enc = self.llm_tokenizer(
#             output_text, truncation=True, padding="max_length",
#             max_length=self.max_len_llm, return_tensors="pt"
#         )

#         input_enc = self.input_tokenizer(
#             input_text, truncation=True, padding="max_length",
#             max_length=self.max_len_llm, return_tensors="pt"
#         )
        
#         profile_text = " ".join(profile_text) if isinstance(profile_text, list) else profile_text

#         beh_enc = self.beh_tokenizer(
#             profile_text, truncation=True, padding="max_length",
#             max_length=self.max_len_beh, return_tensors="pt"
#         )
            
#         return {
#             "input_ids": input_enc["input_ids"].squeeze(0),
#             "attention_mask": input_enc["attention_mask"].squeeze(0),
#             "llm_input_ids": llm_input_enc["input_ids"].squeeze(0),
#             "llm_attention_mask": llm_input_enc["attention_mask"].squeeze(0),
#             "labels": llm_output_enc["input_ids"].squeeze(0),
#             "labels_attention_mask": llm_output_enc["attention_mask"].squeeze(0),
#             "beh_input_ids": beh_enc["input_ids"].squeeze(0),
#             "beh_attention_mask": beh_enc["attention_mask"].squeeze(0)
#         }
        

class LaMPDataset(Dataset):
    def __init__(self, json_path, llm_tokenizer, beh_tokenizer, input_tokenizer, max_len_llm=256, max_len_beh=512, llm_type="encoder-decoder"):
        """
        llm_type: "encoder-decoder" или "decoder-only"
        """
        with open(json_path, "r") as f:
            self.data = json.load(f)

        self.llm_tokenizer = llm_tokenizer
        self.beh_tokenizer = beh_tokenizer
        self.input_tokenizer = input_tokenizer
        self.max_len_llm = max_len_llm
        self.max_len_beh = max_len_beh
        self.llm_type = llm_type

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = item["input_text"]
        output_text = item["output_text"]
        profile_text = item["behavior_profile_text"]

        profile_text = " ".join(profile_text) if isinstance(profile_text, list) else profile_text
    
        beh_enc = self.beh_tokenizer(
            profile_text, truncation=True, padding="max_length",
            max_length=self.max_len_beh, return_tensors="pt"
        )

        input_enc = self.input_tokenizer(
            input_text, truncation=True, padding="max_length",
            max_length=self.max_len_beh, return_tensors="pt"
        )

        llm_input_enc = self.llm_tokenizer(
            input_text, truncation=True, padding="max_length",
            max_length=self.max_len_llm, return_tensors="pt"
        )
        llm_output_enc = self.llm_tokenizer(
            output_text, truncation=True, padding="max_length",
            max_length=self.max_len_llm, return_tensors="pt"
        )
        
        if self.llm_type == "encoder-decoder":
            # input → encoder, labels → decoder
            labels = llm_output_enc["input_ids"]
            labels_attention_mask = llm_output_enc["attention_mask"]
        else:
            # decoder-only: объединяем input + output, labels = copy of input_ids
            # объединяем input_ids: [input_text + output_text]
            input_ids = torch.cat(
                [llm_input_enc["input_ids"], llm_output_enc["input_ids"][:, 1:]], dim=1
            )
            attention_mask = torch.cat(
                [llm_input_enc["attention_mask"], llm_output_enc["attention_mask"][:, 1:]], dim=1
            )
        
            # labels = копия input_ids
            labels = input_ids.clone()
            # input часть игнорируем
            labels[:, :llm_input_enc["input_ids"].size(1)] = -100
        
            llm_input_enc = {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }
            labels_attention_mask = attention_mask

        return {
            "input_ids": input_enc["input_ids"].squeeze(0),
            "attention_mask": input_enc["attention_mask"].squeeze(0),
            "llm_input_ids": llm_input_enc["input_ids"].squeeze(0),
            "llm_attention_mask": llm_input_enc["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
            "labels_attention_mask": labels_attention_mask.squeeze(0),
            "beh_input_ids": beh_enc["input_ids"].squeeze(0),
            "beh_attention_mask": beh_enc["attention_mask"].squeeze(0),
        }

def collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])

    llm_input_ids = torch.stack([item["llm_input_ids"] for item in batch])
    llm_attention_mask = torch.stack([item["llm_attention_mask"] for item in batch])

    labels = torch.stack([item["labels"] for item in batch])
    labels_attention_mask = torch.stack([item["labels_attention_mask"] for item in batch])
    beh_input_ids = torch.stack([item["beh_input_ids"] for item in batch])
    beh_attention_mask = torch.stack([item["beh_attention_mask"] for item in batch])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "llm_input_ids": llm_input_ids,
        "llm_attention_mask": llm_attention_mask,
        "labels": labels,
        "labels_attention_mask": labels_attention_mask,
        "beh_input_ids": beh_input_ids,
        "beh_attention_mask": beh_attention_mask,
    }