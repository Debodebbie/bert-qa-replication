import torch
from transformers import BertForQuestionAnswering as Model
from transformers import BertTokenizer as Tokenizer

import torch.nn as nn
import torch.optim as optim

class AttentionSentimentClassifier(nn.Module):
    def __init__(self):
        super(AttentionSentimentClassifier, self).__init__()
        self.fc = nn.Linear(1*768, 47)
        self.classify = nn.Sigmoid()

    def forward(self, input):
        return self.classify(self.fc(input.flatten(1)))


from datasets import load_dataset
train_dataset_hg = load_dataset('trec', split='train')
test_dataset_hg  = load_dataset('trec', split='test')


def QUES_probing_classifier_training(train_set, model, tokenizer):

    # The intention of pooled_output and sequence_output are different.
    # Since, the embeddings from the BERT model at the output layer are known to be contextual embeddings,
    # the output of the 1st token, i.e, [CLS] token would have captured sufficient context.
    # Hence, the authors of BERT paper found it sufficient to use only the output from the 1st token for few tasks
    # such as classification.
    # They call this output from the single token (i.e, 1st token) as pooled_output

    criterion = nn.CrossEntropyLoss()
    mlp_classifier = []
    optimizer      = []
    for layer_wise_classifiers in range(13):
        mlp_classifier.append(AttentionSentimentClassifier())
        optimizer.append(optim.Adam(mlp_classifier[len(mlp_classifier)-1].parameters()))

    print("training started")
    for epoch in range(3):
        act_loss = [0.0]*13
        for index, (inputs, labels) in enumerate (train_set):
            output = model(torch.tensor([inputs]), output_hidden_states=True, return_dict=True)
            #Train all probing classifiers for each one generated output of the NLP model
            for layer in range(13):
                optimizer[layer].zero_grad()
                pooled_output = output.hidden_states[layer][:, :1, :]
                y_pred = mlp_classifier[layer](pooled_output)
                labels = torch.tensor([labels])
                loss = criterion(y_pred, labels)
                loss.backward(retain_graph=True)
                optimizer[layer].step()
                act_loss[layer] += loss.item()

            if index % 25 == 24:
                print('[%d, %5d] loss:' %(epoch, index) + ", " + str([round(i / 25.0,3) for i in act_loss]))
                act_loss = [0.0]*13

        for layer in range(13):
            path = './bert_classifier/nn_model_epoch' + str(epoch) + '_layer' + str(layer) + '.pth'
            torch.save(mlp_classifier[layer].state_dict(), path)
        print("model saved")
        print("training finished")


def QUES_probing_classifier_testing(test_set, model, tokenizer):

    criterion = nn.CrossEntropyLoss()
    mlp_classifier = []

    load_from_epoch = 1
    print("load classifier from training epoch " + str(load_from_epoch))
    for layer in range(13):
        mlp_classifier.append(AttentionSentimentClassifier())
        path = './bert_classifier/nn_model_epoch' + str(load_from_epoch) + '_layer' + str(layer) + '.pth'
        mlp_classifier[layer].load_state_dict(torch.load(path))

    print("evaluation started")
    act_loss = [0.0]*13
    for index, (inputs, labels) in enumerate (test_set):

        output = model(torch.tensor([inputs]), output_hidden_states=True, return_dict=True)

        #Train all probing classifiers for a calculated output of BERT
        for layer in range(13):
            mlp_classifier[layer].eval()
            pooled_output = output.hidden_states[layer][:, :1, :]
            y_pred = mlp_classifier[layer](pooled_output)
            labels = torch.tensor([labels])
            loss = criterion(y_pred, labels)
            loss.backward(retain_graph=True)
            act_loss[layer] += loss.item()

        if index % 25 == 24:
            print('[%5d] loss:' %(index) + ", " + str([round(i / index, 3) for i in act_loss]))

    print("training finished")

if __name__ == '__main__':

    print("load model")
    # = 'distilbert-base-uncased-distilled-squad'
    model_name = 'csarron/bert-base-uncased-squad-v1'
    tokenizer = Tokenizer.from_pretrained(model_name)
    model = Model.from_pretrained(model_name)#, return_dict=True)
    print("model loaded")

    print("prepare_dataset")
    train_dataset = []
    for data in train_dataset_hg:
        input_ids = tokenizer.encode(data['text'])
        train_dataset.append((input_ids, data['label-fine']))
    test_dataset = []
    for data in test_dataset_hg:
        input_ids = tokenizer.encode(data['text'])
        test_dataset.append([input_ids, data['label-fine']])
    print(len(train_dataset))
    print(len(test_dataset))
    #QUES_probing_classifier_training (train_dataset, model, tokenizer)
    QUES_probing_classifier_testing  (test_dataset, model, tokenizer)


#Results
#After epoch0
#[  499] loss:, [3.518, 3.446, 3.375, 3.254, 3.198, 3.22, 3.233]
#After epoch1
#[  499] loss:, [3.518, 3.396, 3.326, 3.209, 3.119, 3.151, 3.277]
#After epoch2
#[  499] loss:, [3.519, 3.369, 3.29, 3.173, 3.101, 3.139, 3.247]

# As visible:
# In this case adding more epochs reduces the summed loss and increases the contrast between the values,
# but it keeps the relevant layer wise dependency relatively constant, hence we considered one epoch training
# for the MLP-classifier as sufficient for further experiments.
#