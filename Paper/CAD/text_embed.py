import torch.nn as nn
import torch
from transformers import BertTokenizer, BertModel
from pathlib import Path
import os
from huggingface_hub import hf_hub_download
from typing import Optional

MODEL_NAME_DICT = {"bert_large_uncased": "google-bert/bert-large-uncased"}


def prepare_cross_attention_mask_batch(mask, cad_seq_len=271):
    if mask.shape[0] > 1:
        length = mask.shape[1]
        batch_size = mask.shape[0]
        mask = mask.reshape(batch_size, 1, length)
    mask = torch.tile(mask, (1, cad_seq_len, 1))  # (512) -> (271, 512)
    mask = torch.where(
        mask, -torch.inf, 0
    )  # Changing the [True,False] format to [0,-inf] format

    return mask


class TextEmbedder(nn.Module):
    def __init__(
            self,
            model_name: str,
            max_seq_len: int,
            cache_dir: Optional[str] = None,
            tokenizer: Optional[BertTokenizer] = None
    ):
        """
        Args:
            model_name: Name of the model (e.g., "bert_large_uncased")
            max_seq_len: Maximum sequence length for text
            cache_dir: Directory to cache model files
            tokenizer: Pre-loaded tokenizer (optional)
        """
        super(TextEmbedder, self).__init__()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_seq_len = max_seq_len
        self.model_name = MODEL_NAME_DICT.get(model_name, "bert_large_uncased")

        # 使用传入的分词器或创建新的
        if tokenizer:
            self.tokenizer = tokenizer
            print("Using provided tokenizer")
        else:
            self.tokenizer = self._load_tokenizer(cache_dir)

        # 加载BERT模型
        self.model = self._load_model(cache_dir).to(device)

        # 冻结模型参数
        for param in self.model.parameters():
            param.requires_grad = False

    def _load_tokenizer(self, cache_dir: Optional[str] = None) -> BertTokenizer:
        """加载分词器，增加错误处理和自动下载功能"""
        try:
            # 确保缓存目录存在
            if cache_dir:
                cache_path = Path(cache_dir)
                if not cache_path.exists():
                    cache_path.mkdir(parents=True, exist_ok=True)
                    print(f"Created cache directory: {cache_path}")

                # 检查必需文件是否存在
                required_files = ["vocab.txt", "tokenizer.json", "tokenizer_config.json"]
                missing_files = [f for f in required_files if not (cache_path / f).exists()]

                if missing_files:
                    print(f"Missing files in cache: {missing_files}")
                    # 尝试自动下载缺失文件
                    try:
                        for file in missing_files:
                            hf_hub_download(
                                repo_id=self.model_name,
                                filename=file,
                                cache_dir=cache_dir
                            )
                        print("Missing files downloaded successfully")
                    except Exception as e:
                        print(f"Failed to download missing files: {e}")
                        cache_dir = None

            # 尝试从缓存目录加载
            if cache_dir:
                print(f"Loading tokenizer from cache: {cache_dir}")
                return BertTokenizer.from_pretrained(cache_dir)
        except Exception as e:
            print(f"Error loading tokenizer from cache: {e}")

        # 回退到默认加载方式
        print(f"Loading tokenizer from model name: {self.model_name}")
        return BertTokenizer.from_pretrained(self.model_name)

    def _load_model(self, cache_dir: Optional[str] = None) -> BertModel:
        """加载BERT模型，增加错误处理"""
        try:
            if cache_dir:
                print(f"Loading model from cache: {cache_dir}")
                return BertModel.from_pretrained(cache_dir)
        except Exception as e:
            print(f"Error loading model from cache: {e}")

        # 回退到默认加载方式
        print(f"Loading model from model name: {self.model_name}")
        return BertModel.from_pretrained(
            self.model_name,
            cache_dir=cache_dir,
            max_position_embeddings=self.max_seq_len
        )

    def get_embedding(self, texts: list[str]):
        """获取文本嵌入"""
        if isinstance(texts, str):
            texts = [texts]

        with torch.no_grad():
            input_ids = self.tokenizer(
                texts,
                return_tensors="pt",
                max_length=self.max_seq_len,
                truncation=True,
                padding=True,
            ).to("cuda")

            all_output = self.model(**input_ids)
            embedding = all_output[0]
            key_padding_mask = (input_ids["attention_mask"] == 0)

        return embedding, key_padding_mask

    @staticmethod
    def from_config(config: dict, tokenizer: Optional[BertTokenizer] = None):
        """从配置创建TextEmbedder实例"""
        return TextEmbedder(
            model_name=config['model_name'],
            max_seq_len=config['max_seq_len'],
            cache_dir=config.get('cache_dir'),
            tokenizer=tokenizer
        )