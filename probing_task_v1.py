import torch
from transformers import DistilBertForQuestionAnswering as Model
from transformers import DistilBertTokenizer as Tokenizer

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


class AttentionSentimentClassifier(nn.Module):

    def __init__(self):
        super(AttentionSentimentClassifier, self).__init__()
        self.fc = nn.Linear(400, 47)
        self.classify = nn.Sigmoid()

    def forward(self, input):
        return self.classify(self.fc(input))

from datasets import load_dataset
train_dataset_hg = load_dataset('trec', split='train')
test_dataset_hg  = load_dataset('trec', split='test')
print((train_dataset_hg[50]['label-fine'], train_dataset_hg[50]['text']))
#prepare_dataset
train_dataset = []
for data in train_dataset_hg:
    train_dataset.append((data['text'], data['label-fine']))
test_dataset = []
for data in test_dataset_hg:
    test_dataset.append([data['text'], data['label-fine']])

print(train_dataset[50])
print(test_dataset [50])
print(len(train_dataset))

def QA_type_probing(question, context, model, tokenizer):
    question_ids = tokenizer.encode(question)
    context_ids  = tokenizer.encode(context)
    input_ids = question_ids + context_ids
    output = model(torch.tensor([input_ids]), output_hidden_states=True, return_dict=True)
    #start_scores = output.start_logits
    #end_scores = output.end_logits
    #answer_start = torch.argmax(start_scores)
    #answer_end = torch.argmax(end_scores)
    #tokens = tokenizer.convert_ids_to_tokens(input_ids)
    #answer = ' '.join(tokens[answer_start:answer_end + 1]).replace(" ##", "")
    #print(answer)
    print(output.hidden_states.shape)


def run(question, context, model, tokenizer):
    question_ids = tokenizer.encode(question)
    context_ids  = tokenizer.encode(context)
    input_ids = question_ids + context_ids
    output = model(torch.tensor([input_ids]), output_hidden_states=True, return_dict=True)
    start_scores = output.start_logits
    end_scores = output.end_logits
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    answer = ' '.join(tokens[answer_start:answer_end + 1]).replace(" ##", "")
    print(answer)
    print(output.hidden_states)

if __name__ == '__main__':

    print("load model")
    model_name = 'distilbert-base-uncased-distilled-squad'
    tokenizer = Tokenizer.from_pretrained(model_name)
    model = Model.from_pretrained(model_name, return_dict=True)
    print("model loaded")


    squad_question = "What is a common punishment in the UK and Ireland?"

    squad_context = "Currently detention is one of the most common punishments in schools in the United States, the UK, Ireland, " \
              "Singapore and other countries. It requires the pupil to remain in school at a given time in the school day " \
              "(such as lunch, recess or after school); or even to attend school on a non-school day, e.g. Saturday detention " \
              "held at some schools. During detention,students normally have to sit in a classroom and do work, write lines or" \
              " a punishment essay, or sit quietly."

    QA_type_probing(squad_question, squad_context, model, tokenizer)

