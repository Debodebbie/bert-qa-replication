import torch
#from transformers import DistilBertForQuestionAnswering as Model
#from transformers import DistilBertTokenizer as Tokenizer
#from transformers import RobertaForQuestionAnswering as Model
#from transformers import RobertaTokenizer as Tokenizer
from transformers import T5ForConditionalGeneration as Model
from transformers import T5Tokenizer as Tokenizer

import torch.nn as nn
import torch.optim as optim

class AttentionSentimentClassifier(nn.Module):
    def __init__(self):
        super(AttentionSentimentClassifier, self).__init__()
        # second layer --comment out, if only single layer is relevant
        self.fc2 = nn.Linear(1*768, 1*768)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(1 * 768, 47)
        self.classify = nn.Sigmoid()

    def forward(self, input):
        return self.classify(self.fc1(self.relu(self.fc2(input.flatten(1)))))


# download and prepare the dataset
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

    '''   
    load_from_epoch = 0
    print("load classifier from training epoch " + str(load_from_epoch))
    for layer in range(7):
        mlp_classifier.append(AttentionSentimentClassifier())
        path = './probing_token_num/nn_model_epoch_4_' + str(load_from_epoch) + '_layer' + str(layer) + '.pth'
        mlp_classifier[layer].load_state_dict(torch.load(path))
        optimizer.append(optim.Adam(mlp_classifier[len(mlp_classifier) - 1].parameters()))
    '''

    print("training started")
    for epoch in range(0,2):
        act_loss = [0.0]*13
        for index, (inputs, labels) in enumerate(train_set):

            output = model.forward(input_ids=inputs['input_ids'], decoder_input_ids=inputs.input_ids,
                                   attention_mask=inputs['attention_mask'], output_hidden_states=True)

            #Train all probing classifiers for each one generated output of the NLP model
            for layer in range(13):
                optimizer[layer].zero_grad()
                pooled_output = output.encoder_hidden_states[layer][:, :1, :]
                y_pred = mlp_classifier[layer](pooled_output)
                labels = torch.tensor([labels])
                loss = criterion(y_pred, labels)
                loss.backward(retain_graph=True)
                optimizer[layer].step()
                act_loss[layer] += loss.item()

            if index % 25 == 24:
                print('[%d, %5d] loss:' %(epoch, index) + ", " + str([round(i / 25.0,3) for i in act_loss]))
                act_loss = [0.0]*13
                #Use this break only for the first run to test the correct path specifications etc.
                break
        for layer in range(13):
            path = './probing_T5_QUES/nn_model_epoch_' + str(epoch) + '_layer' + str(layer) + '.pth'
            torch.save(mlp_classifier[layer].state_dict(), path)
        print("model saved")
        print("training finished")



if __name__ == '__main__':

    # load the model
    print("load model")
    model_name = 'valhalla/t5-base-qa-qg-hl'
    tokenizer = Tokenizer.from_pretrained(model_name)
    model = Model.from_pretrained(model_name, return_dict=True)
    print("model loaded")

    # prepare the dataset
    print("prepare_dataset")
    train_dataset = []
    for data in train_dataset_hg:
        input_ids = tokenizer(data['text'], return_tensors="pt")
        train_dataset.append((input_ids, data['label-fine']))
    test_dataset = []
    for data in test_dataset_hg:
        input_ids = tokenizer(data['text'], return_tensors="pt")
        test_dataset.append([input_ids, data['label-fine']])
    print(len(train_dataset))
    print(len(test_dataset))
    # train the model
    QUES_probing_classifier_training (train_dataset, model, tokenizer)

#
# As descript:
# In this case adding more epochs reduces the summed loss and increases the contrast between the values,
# but it keeps the relevant layer wise dependency relatively constant, hence we considered one epoch training
# for the MLP-classifier as sufficient for further experiments.
#