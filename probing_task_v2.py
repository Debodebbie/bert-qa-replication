import torch
from transformers import DistilBertForQuestionAnswering as Model
from transformers import DistilBertTokenizer as Tokenizer

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
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


def QA_type_probing(train_loader, test_loader, model, tokenizer):
    '''
    question_ids = tokenizer.encode(question)
    input_ids = question_ids
    output = model(torch.tensor([input_ids]), output_hidden_states=True, return_dict=True)
    print(len(output.hidden_states))
    print(output.hidden_states[layer].shape)
    # The intention of pooled_output and sequence_output are different.
    # Since, the embeddings from the BERT model at the output layer are known to be contextual embeddings,
    # the output of the 1st token, i.e, [CLS] token would have captured sufficient context.
    # Hence, the authors of BERT paper found it sufficient to use only the output from the 1st token for few tasks
    # such as classification.
    # They call this output from the single token (i.e, 1st token) as pooled_output
    pooled_output = output.hidden_states[layer][:,:1,:]
    print(pooled_output.shape)

    output = mlp_classifier(pooled_output)
    #print(output)
    '''

    criterion = nn.CrossEntropyLoss()
    mlp_classifier = []
    optimizer      = []
    for layer_wise_classifiers in range(7):
        mlp_classifier.append(AttentionSentimentClassifier())
        optimizer.append(optim.Adam(mlp_classifier[len(mlp_classifier)-1].parameters()))


    print("training started")
    for epoch in range(5):
        act_loss = [0.0]*7
        for index, (inputs, labels) in enumerate (train_loader):

            output = model(torch.tensor([inputs]), output_hidden_states=True, return_dict=True)

            #Train all probing classifiers for a calculated output of BERT
            for layer in range(7):
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
                act_loss = [0.0]*7
                #break

        for layer in range(7):
            path = './classifier/nn_model_epoch' + str(epoch) + '_layer' + str(layer) + '.pth'
            torch.save(mlp_classifier[layer].state_dict(), path)
        print("model saved")
        #break
        print("training finished")

        #12:45


if __name__ == '__main__':

    print("load model")
    model_name = 'distilbert-base-uncased-distilled-squad'
    tokenizer = Tokenizer.from_pretrained(model_name)
    model = Model.from_pretrained(model_name, return_dict=True)
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

    #train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    #test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=16, num_workers=4)

    QA_type_probing(train_dataset, test_dataset, model, tokenizer)


