import torch
from transformers import DistilBertForQuestionAnswering as Model
from transformers import DistilBertTokenizer as Tokenizer

def get_segments(question_ids, context_ids):
    max_len = 512
    quest_len = len(question_ids)
    segments = []
    index = 0
    while(index<len(question_ids) + len(context_ids)):
        segments.append(question_ids + context_ids[index:index + 512 - quest_len - 1])
        index+=256 - quest_len - 1
    return segments


def run(question, context, model, tokenizer):
    question_ids = tokenizer.encode(question)
    context_ids  = tokenizer.encode(context)
    print((len(question_ids), len(context_ids)))
    print(len(question_ids + context_ids))
    segments = get_segments(question_ids, context_ids)

    full_answer = ""
    for index, input_ids in enumerate(segments):
        output = model(torch.tensor([input_ids]))
        start_scores = output.start_logits
        end_scores = output.end_logits
        answer_start = torch.argmax(start_scores)
        answer_end = torch.argmax(end_scores)
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        print((round(index/len(segments),2), tokens[answer_start:answer_end + 1]))
        answer = ' '.join(tokens[answer_start:answer_end + 1]).replace(" ##", "")
        if answer != '[CLS]':
            full_answer += answer + " / "
    print(full_answer)
    context_ids  = tokenizer.encode(full_answer)
    input_ids = question_ids + context_ids
    output = model(torch.tensor([input_ids]))
    start_scores = output.start_logits
    end_scores = output.end_logits
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    #print((round(index / len(segments), 2), tokens[answer_start:answer_end + 1]))
    print("final answer: " + ' '.join(tokens[answer_start:answer_end + 1]).replace(" ##", ""))






import wikipedia as wiki
import pprint as pp

print("load model")
model_name = 'distilbert-base-uncased-distilled-squad'
tokenizer = Tokenizer.from_pretrained(model_name)
model = Model.from_pretrained(model_name, return_dict=True)
print("model loaded")

question = 'Where in India located?'

results = wiki.search(question)
print("Wikipedia search results for our question:\n")
pp.pprint(results)

page = wiki.page(results[0])
text = page.content
#print(text)
print(f"\nThe {results[0]} Wikipedia article contains {len(text)} characters.")

run(question, text, model, tokenizer)

