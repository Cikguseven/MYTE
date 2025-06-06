import torch
import logging
import os
import argparse
import sys
from transformers import set_seed
import json

from tokenizers import processors
from tokenizers import Tokenizer
from tokenizers.implementations import SentencePieceBPETokenizer, SentencePieceUnigramTokenizer
from tokenizers.models import Unigram, BPE
from tokenizers.trainers import UnigramTrainer, BpeTrainer
from tokenizers.normalizers import BertNormalizer
from tokenizers.pre_tokenizers import BertPreTokenizer
from tokenizers.processors import TemplateProcessing

from transformers import XLMRobertaTokenizerFast


set_seed(20)
logging.info(torch.cuda.is_available())


def get_save_path(save_dir, type, languages, vocab_size, alpha, lowercase):
    save_path = os.path.join(save_dir, type, '-'.join(languages),f"alpha-{alpha}_N-{vocab_size}" )
    if lowercase:
        save_path += '-lc'
    return save_path


def convert_tokenizer(save_path, type, just_trained=True):

    if not just_trained and type.startswith('sp'):
        # TODO: the following paths are only to update arguments in old tokenizers, can be deleted when all tokenizers are converted.
        tokenizer = Tokenizer.from_file(os.path.join(save_path, "tokenizer.json"))

        tokenizer.post_processor = processors.RobertaProcessing(
            sep=("</s>", tokenizer.model.token_to_id("</s>")),
            cls=("<s>", tokenizer.model.token_to_id("<s>")),
        )

        tokenizer.save(os.path.join(save_path, "tokenizer.json"))

        with open(sys.argv[0], 'r') as cur_file:
            cur_running = cur_file.readlines()
        with open(os.path.join(save_path,'conversion_script.py'),'w') as log_file:
            log_file.writelines(cur_running)
        with open(os.path.join(save_path,'conversion_args.txt'),'w') as log_file:
                log_file.writelines(sys.argv[1:])

    # convert to XLMRoberta Tokenizer
    hf_tokenizer = XLMRobertaTokenizerFast.from_pretrained(save_path, max_len=512)
    hf_tokenizer.save_pretrained(save_path)


def save_tokenizer(tokenizer, out_path):
    """ Function saving vocab and arguments to the """

    os.makedirs(out_path, exist_ok=True)

    logging.info(f"Saving tokenizer at {out_path}")
    tokenizer.save(os.path.join(out_path, "tokenizer.json"))

    with open(sys.argv[0], 'r') as cur_file:
        cur_running = cur_file.readlines()
    with open(os.path.join(out_path,'script.py'),'w') as log_file:
        log_file.writelines(cur_running)
    with open(os.path.join(out_path,'args.txt'),'w') as log_file:
        log_file.writelines(sys.argv[1:])

    # Saving vocab
    with open(os.path.join(out_path,'vocab.json'), "w", encoding='utf-8') as outfile:
        json.dump(dict(sorted(tokenizer.get_vocab().items(), key=lambda item: item[1])),
                  outfile, indent=2, ensure_ascii=False)


def main(args):
    alpha = args.alpha
    vocab_size = args.vocab_size
    languages = args.languages
    data_paths = args.data_list
    lowercase = not args.cased

    type = args.type
    out_dir= args.out_dir

    save_path = get_save_path(out_dir, type, languages, vocab_size, alpha, lowercase)

    if args.train_tokenizer:
        non_sp_special_tokens = ["[UNK]","[CLS]","[SEP]","[PAD]","[MASK]"]
        sp_special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]

        if type == "unigram":
            # TODO: check where to put special tokens
            tokenizer = Tokenizer(Unigram(None))
            trainer = UnigramTrainer(vocab_size=vocab_size, special_tokens=non_sp_special_tokens)
        elif type == "bpe":
            tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
            trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=non_sp_special_tokens)
        elif type == "sp-unigram":
            tokenizer = SentencePieceUnigramTokenizer()
        elif type == "sp-bpe":
            tokenizer = SentencePieceBPETokenizer(unk_token="<unk>")
        else:
            raise ValueError(f"Unknown tokenizer type: {type}.")

        if not type.startswith("sp"):
            tokenizer.normalizer = BertNormalizer(lowercase=lowercase)
            tokenizer.pre_tokenizer = BertPreTokenizer()
            tokenizer.post_processor = TemplateProcessing(
                single="[CLS] $A [SEP]",
                pair="[CLS] $A [SEP] $B:1 [SEP]:1",
                special_tokens=[
                    ("[CLS]", non_sp_special_tokens.index("[CLS]")),
                    ("[SEP]", non_sp_special_tokens.index("[SEP]")),
                ],
            )
        else:
            tokenizer.post_processor = processors.RobertaProcessing(
                sep=("</s>", sp_special_tokens.index("</s>")),
                cls=("<s>", sp_special_tokens.index("<s>")),
            )

        # Customize training
        logging.info(f"Training tokenizer on:\n{data_paths}")
        if not type.startswith("sp"):
            tokenizer.train(
                data_paths,
                trainer
            )
        else:
            tokenizer.train(data_paths, vocab_size=vocab_size, special_tokens=sp_special_tokens)

        save_tokenizer(tokenizer, save_path)

        logging.info("Done creating tokenizer")

    if args.convert_tokenizer:

        convert_tokenizer(save_path, type, just_trained=args.train_tokenizer)
        logging.info("Done converting tokenizer")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--data_list', nargs='+', help='<Required> Set flag', required=True)
    parser.add_argument('-o', '--out_dir', type=str, required=True)
    parser.add_argument('-a', '--alpha', type=str, required=True, help='Balancing coeficient alpha.')
    parser.add_argument('-l', '--languages', nargs='+', required=True, help='List of languages the tokenizer was trained on.')
    parser.add_argument('-v', '--vocab_size', type=int, required=True)
    parser.add_argument('-t', '--type', type=str, required=False, default="unigram")
    parser.add_argument('-c', '--cased', type=bool, default=False)
    parser.add_argument('-tt', '--train_tokenizer', type=lambda x: (str(x).lower() == 'true'), default=True,
                        help="Whether to train a tokenizer.")
    parser.add_argument('-ct', '--convert_tokenizer', type=lambda x: (str(x).lower() == 'true'), default=True,
                        help="Whether to convert tokenizer, to XLMRoberta type")
    args = parser.parse_args()
    main(args)