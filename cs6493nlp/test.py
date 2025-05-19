from collections import defaultdict
from tqdm import tqdm
import nltk
# from nltk.tokenize import word_tokenize


# def convert_idx(text, tokens):
#     current = 0
#     spans = []
#     for token in tokens:
#         current = text.find(token, current)
#         if current < 0:
#             print("Token {} cannot be found".format(token))
#             raise Exception()
#         spans.append((current, current + len(token)))
#         current += len(token)

#     return spans

# def word_tokenize(tokens):
#     return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]



# def process_txt_files(src_file, tgt_file, ans_file):
#     counter = defaultdict(lambda: 0)  # Initialize word frequency counter
#     examples = []  # Initialize list to store processed examples
#     total = 0  # Initialize total example count
    
#     # Open and read source file containing article content
#     with open(src_file, "r") as src_f:
#         # Open and read target file containing questions
#         with open(tgt_file, "r") as tgt_f:
#             with open(ans_file, "r") as ans_f:
#             # Read lines from both source and target files simultaneously
#                 for src_line, tgt_line, ans_line in zip(src_f, tgt_f, ans_f):
#                 # Preprocess context: convert to lowercase and tokenize
#                     context = src_line.replace(
#                     "''", '" ').replace("``", '" ').strip().lower()
#                     context_tokens = word_tokenize(context)
#                     spans = convert_idx(context,context_tokens)
                    
#                     # Preprocess question: convert to lowercase and tokenize
#                     ques = tgt_line.replace(
#                     "''", '" ').replace("``", '" ').strip().lower()
#                     ques_tokens = word_tokenize(ques)
                    
#                     # Update word frequencies in the counter
#                     for token in ques_tokens:
#                         counter[token] += 1
                    
#                     # Initialize lists to store answer indices and texts
#                     y1s, y2s = [], []
#                     answer_texts = []

#                     answer_text = ans_file.replace(
#                     "''", '" ').replace("``", '" ').strip().lower()
#                     answer_start = context.find(answer_texts)
#                     answer_end = answer_start + len(answer_text)
#                     answer_texts.append(answer_text)
#                     answer_span = []

#                     for token in context_tokens:
#                         counter[token] += len
#                     # Create example dictionary containing processed information
#                     example = {"context_tokens": context_tokens, "ques_tokens": ques_tokens,
#                             "y1s": y1s, "y2s": y2s, "answers": answer_texts}
#                     examples.append(example)  # Add example to list
    
#     return examples, counter  # Return processed examples and word frequency counter

# from collections import defaultdict
# from tqdm import tqdm
# import nltk
# from nltk.tokenize import word_tokenize


def convert_idx(text, tokens):
    current = 0
    spans = []
    for token in tokens:
        current = text.find(token, current)
        if current < 0:
            print("Token {} cannot be found".format(token))
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)

    return spans

def word_tokenize(tokens):
    return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]


def process_txt_files(src_file, tgt_file, ans_file):
    counter = defaultdict(lambda: 0)
    examples = []
    total = 0
    
    with open(src_file, "r") as src_f, open(tgt_file, "r") as tgt_f, open(ans_file, "r") as ans_f:
        for src_line, tgt_line, ans_line in zip(src_f, tgt_f, ans_f):
            context = src_line.replace("''", '" ').replace("``", '" ').strip().lower()
            context_tokens = word_tokenize(context)
            spans = convert_idx(context, context_tokens)
            
            ques = tgt_line.replace("''", '" ').replace("``", '" ').strip().lower()
            ques_tokens = word_tokenize(ques)
            
            for token in ques_tokens:
                counter[token] += 1
            
            y1s, y2s = [], []
            answer_texts = []
            
            for answer_text in ans_line.strip().split(';'):
                answer_text = answer_text.strip().lower()
                answer_start = context.find(answer_text)
                if answer_start < 0:
                    print("Answer '{}' not found in context".format(answer_text))
                    continue
                answer_end = answer_start + len(answer_text)
                answer_texts.append(answer_text)
                answer_span = []
                for idx, span in enumerate(spans):
                    if not (answer_end <= span[0] or answer_start >= span[1]):
                        answer_span.append(idx)
                if answer_span:
                    y1, y2 = answer_span[0], answer_span[-1]
                    y1s.append(y1)
                    y2s.append(y2)
                else:
                    print("Answer '{}' not found in context spans".format(answer_text))
            
            example = {"context_tokens": context_tokens, "ques_tokens": ques_tokens,
                       "y1s": y1s, "y2s": y2s, "answers": answer_texts}
            examples.append(example)
    
    return examples, counter

src_file = 'test_src.txt'
tgt_file = 'test_tgt.txt'
ans_file = 'test_ans.txt'
examples, counter = process_txt_files(src_file, tgt_file, ans_file)
print(examples, counter)

# srcfile = 'test_src.txt'
# tgtfile = 'test_tgt.txt'
# example , counter = process_txt_files(srcfile,tgtfile)
# print(example,counter)