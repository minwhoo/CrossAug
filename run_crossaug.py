import time
import argparse

import torch
import jsonlines
from tqdm import trange, tqdm
from nltk import word_tokenize
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def find_substitution_map(sent1, sent2):
    """Find overlapping words in the given two sentences"""
    words1 = word_tokenize(sent1)
    words2 = word_tokenize(sent2)
    start_idx = 0
    while words1[start_idx] == words2[start_idx]:
        start_idx += 1
        if start_idx == len(words1) or start_idx == len(words2):
            return None

    end_idx = -1
    while words1[end_idx] == words2[end_idx]:
        end_idx -= 1

    if end_idx == -1:
        words_overlap1 = words1[start_idx:]
        words_overlap2 = words2[start_idx:]
    else:
        words_overlap1 = words1[start_idx:end_idx+1]
        words_overlap2 = words2[start_idx:end_idx+1]

    if 0 < len(words_overlap1) <= 3 and 0 < len(words_overlap2) <= 3:
        return words_overlap1, words_overlap2
    else:
        return None


def substitute_sent(sent, orig_words, replacing_words):
    """Find and substitute word phrases from given sentence"""
    sent_words = word_tokenize(sent)
    j = 0
    match_start_idx = None
    match_end_idx = None
    matches = []
    for i in range(len(sent_words)):
        if sent_words[i] == orig_words[j]:
            if j == 0:
                match_start_idx = i
            j += 1
        else:
            j = 0
            match_start_idx = None
            match_end_idx = None
        if j == len(orig_words):
            match_end_idx = i
            matches.append((match_start_idx, match_end_idx))
            j = 0
            match_start_idx = None
            match_end_idx = None
    if len(matches) == 1:
        i, j = matches[0]
        return ' '.join(sent_words[:i] + replacing_words + sent_words[j+1:])
    else:
        return None


def generate_negative_claims(data, batch_size):
    """Generate negative (refuted) claims using fine-tuned negative claim generation model"""
    model_name = 'minwhoo/bart-base-negative-claim-generation'
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to('cuda' if torch.cuda.is_available() else 'cpu')

    for i in trange(0, len(data), batch_size):
        sents = [d['claim'] for d in data[i:i+batch_size]]
        batch = tokenizer(sents, padding=True, truncation=True, return_tensors="pt")
        out = model.generate(batch['input_ids'].to(model.device), num_beams=5)
        refuted_sents = tokenizer.batch_decode(out, skip_special_tokens=True)
        for j, refuted in enumerate(refuted_sents):
            data[i + j]['claim_refuted'] = refuted
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", type=str)
    parser.add_argument("--out_file", type=str)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    print(f"Reading from path: {args.in_file}")
    with jsonlines.open(args.in_file, mode='r') as reader:
        data = [obj for obj in reader]
    print(f"Data loaded! Data size: {len(data):,}")

    print('Generate negative claims')
    start_time = time.time()
    data = generate_negative_claims(data, args.batch_size)
    print(f"time took: {time.time() - start_time}")

    print('Modify evidence using lexical search-based substitution')
    failed_cnt = 0
    start_time = time.time()
    for d in tqdm(data):
        try:
            span_pair = find_substitution_map(d['claim'], d['claim_refuted'])
        except:
            failed_cnt += 1
        else:
            if span_pair is not None:
                orig_span, replace_span  = span_pair
                evid_refuted = substitute_sent(d['evidence'], orig_span, replace_span)
                if evid_refuted is not None:
                    d['evidence_refuted'] = evid_refuted
    print(f"time took: {time.time() - start_time}")

    print('Augment data')
    augmented_data = []
    for d in data:
        augmented_data.append({
            'gold_label': d['gold_label'],
            'evidence': d['evidence'],
            'claim': d['claim'],
            'id': len(augmented_data),
            'weight': 0.0,
        })
        if d['gold_label'] == 'SUPPORTS':
            augmented_data.append({
                    'gold_label': 'REFUTES',
                    'evidence': d['evidence'],
                    'claim': d['claim_refuted'],
                    'id': len(augmented_data),
                    'weight': 0.0,
                })
            if 'evidence_refuted' in d:
                augmented_data.append({
                        'gold_label': 'REFUTES',
                        'evidence': d['evidence_refuted'],
                        'claim': d['claim'],
                        'id': len(augmented_data),
                        'weight': 0.0,
                    })
                augmented_data.append({
                        'gold_label': 'SUPPORTS',
                        'evidence': d['evidence_refuted'],
                        'claim': d['claim_refuted'],
                        'id': len(augmented_data),
                        'weight': 0.0,
                    })

    print(f"Saving to path: {args.out_file}")
    with jsonlines.open(args.out_file, mode='w') as writer:
        writer.write_all(augmented_data)
    print(f"Data saved! Data size: {len(augmented_data):,}")


if __name__ == "__main__":
    main()
