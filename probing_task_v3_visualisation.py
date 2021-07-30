import matplotlib.pyplot as plt
import numpy as np

'''
#distilBERT epoch dependency
array = [[3.518, 3.446, 3.375, 3.254, 3.198, 3.22, 3.233],
         [3.518, 3.396, 3.326, 3.209, 3.119, 3.151, 3.277],
         [3.519, 3.369, 3.29, 3.173, 3.101, 3.139, 3.247],
         [3.517, 3.349, 3.285, 3.142, 3.091, 3.134, 3.123]]
'''

'''
#BERT-base-uncased, epoch dependency/probing task
array = [[3.52, 3.479, 3.415, 3.384, 3.317, 3.279, 3.275, 3.235, 3.263, 3.247, 3.205, 3.192, 3.231],
        [3.52, 3.446, 3.363, 3.332, 3.261, 3.239, 3.215, 3.198, 3.205, 3.2, 3.174, 3.166, 3.187]]
'''

'''
#distilBERT token dependency
array = [[3.378, 3.352, 3.263, 3.19, 3.077, 3.138, 3.184],
         [3.394, 3.31, 3.257, 3.158, 3.052, 3.112, 3.125],
         [3.303, 3.266, 3.219, 3.195, 3.123, 3.299, 3.397],
         [3.242, 3.253, 3.213, 3.174, 3.089, 3.266, 3.231]]
'''

'''
# distilBERT two layered classifier #Layers 1,2,3 of Felix
# (default lr maybe wrong at his device)
array = [[3.521, 3.501, 3.572, 3.49, 3.275, 3.356, 3.417],
        [3.512, 3.537, 3.457, 3.423, 3.229, 3.465, 3.721],
        [3.51, 3.488, 3.509, 3.397, 3.22, 3.455, 3.726],
        [3.508, 3.488, 3.741, 3.375, 3.195, 3.452, 3.718]]

#distilBERT two layered classifier
array = [[3.521, 3.501, 3.572, 3.49, 3.275, 3.356, 3.417],
        [3.5, 3.48, 3.496, 3.397, 3.211, 3.322, 3.391],
        [3.498, 3.472, 3.474, 3.378, 3.188, 3.31, 3.388],
        [3.498, 3.46, 3.459, 3.372, 3.17, 3.304, 3.395]]
'''


'''
#Compare DistilBERT, BERT and RoBERTa
array = [[3.518, 3.446, 3.446, 3.375, 3.375, 3.254, 3.254, 3.198, 3.198, 3.22, 3.22, 3.233, 3.233],
        [3.518, 3.396, 3.396, 3.326, 3.326, 3.209, 3.209, 3.119, 3.119, 3.151, 3.151, 3.277, 3.277],
        [3.52, 3.479, 3.415, 3.384, 3.317, 3.279, 3.275, 3.235, 3.263, 3.247, 3.205, 3.192, 3.231],
        [3.52, 3.446, 3.363, 3.332, 3.261, 3.239, 3.215, 3.198, 3.205, 3.2, 3.174, 3.166, 3.187],
         #RoBERTa
         [3.52, 3.453, 3.517, 3.519, 3.519, 3.518, 3.525, 3.516, 3.504, 3.517, 3.572, 3.524, 3.517],
         [3.513, 3.39, 3.504, 3.512, 3.512, 3.505, 3.514, 3.462, 3.382, 3.506, 3.593, 3.525, 3.453]]
         #T5
         #[3.856, 3.858, 3.89, 3.858, 3.858, 3.858, 3.925, 3.927, 3.917, 3.94, 3.915, 3.956, 3.313],
         #[3.856, 3.858, 3.858, 3.858, 3.858, 3.858, 3.925, 3.927, 3.917, 3.94, 3.915, 3.956, 3.272]]

#Compare DistilBERT, BERT and RoBERTa
array = [[3.52, 3.479, 3.415, 3.384, 3.317, 3.279, 3.275, 3.235, 3.263, 3.247, 3.205, 3.192, 3.231],
        [3.52, 3.446, 3.363, 3.332, 3.261, 3.239, 3.215, 3.198, 3.205, 3.2, 3.174, 3.166, 3.187],
         #T5
         [3.856, 3.858, 3.89, 3.858, 3.858, 3.858, 3.925, 3.927, 3.917, 3.94, 3.915, 3.956, 3.313],
         [3.856, 3.858, 3.858, 3.858, 3.858, 3.858, 3.925, 3.927, 3.917, 3.94, 3.915, 3.956, 3.272]]
'''
'''
'''
#Probing-task plot for the reproduction section
array = [[0.779, 0.816, 0.809, 0.827, 0.825, 0.820]]
#top0_moresteps	micro_avg: 0.779, macro_avg: 0.779, edges-coref-ontonotes_mcc: 0.557, edges-coref-ontonotes_acc: 0.777, edges-coref-ontonotes_precision: 0.778, edges-coref-ontonotes_recall: 0.780, edges-coref-ontonotes_f1: 0.779
#top2_moresteps	micro_avg: 0.816, macro_avg: 0.816, edges-coref-ontonotes_mcc: 0.631, edges-coref-ontonotes_acc: 0.810, edges-coref-ontonotes_precision: 0.814, edges-coref-ontonotes_recall: 0.818, edges-coref-ontonotes_f1: 0.816
#top4_moresteps	micro_avg: 0.809, macro_avg: 0.809, edges-coref-ontonotes_mcc: 0.616, edges-coref-ontonotes_acc: 0.793, edges-coref-ontonotes_precision: 0.807, edges-coref-ontonotes_recall: 0.811, edges-coref-ontonotes_f1: 0.809
#top6_moresteps	micro_avg: 0.827, macro_avg: 0.827, edges-coref-ontonotes_mcc: 0.652, edges-coref-ontonotes_acc: 0.812, edges-coref-ontonotes_precision: 0.824, edges-coref-ontonotes_recall: 0.829, edges-coref-ontonotes_f1: 0.827
#top8_moresteps	micro_avg: 0.825, macro_avg: 0.825, edges-coref-ontonotes_mcc: 0.648, edges-coref-ontonotes_acc: 0.811, edges-coref-ontonotes_precision: 0.820, edges-coref-ontonotes_recall: 0.830, edges-coref-ontonotes_f1: 0.825
#top10_moresteps	micro_avg: 0.820, macro_avg: 0.820, edges-coref-ontonotes_mcc: 0.639, edges-coref-ontonotes_acc: 0.806, edges-coref-ontonotes_precision: 0.816, edges-coref-ontonotes_recall: 0.824, edges-coref-ontonotes_f1: 0.820


'''
#Probing task QUES and REL
#BERT-base-uncased, epoch dependency/probing task
array = [[3.52, 3.479, 3.415, 3.384, 3.317, 3.279, 3.275, 3.235, 3.263, 3.247, 3.205, 3.192, 3.231],
        [3.52, 3.446, 3.363, 3.332, 3.261, 3.239, 3.215, 3.198, 3.205, 3.2, 3.174, 3.166, 3.187],
        [2.816, 2.785, 2.729, 2.73, 2.647, 2.643, 2.634, 2.607, 2.618, 2.588, 2.576, 2.581, 2.61],
        [2.814, 2.756, 2.676, 2.674, 2.595, 2.591, 2.576, 2.553, 2.561, 2.521, 2.538, 2.547, 2.552]]
'''

def normalize_2d(matrix):
    '''
    norm1 = np.linalg.norm(matrix[:2])
    print(matrix[2:4])
    norm2 = np.linalg.norm(matrix[2:4])
    norm3 = np.linalg.norm(matrix[4:6])
    #norm4 = np.linalg.norm(matrix[6:])
    matrix[:2] = matrix[:2]/norm1  # normalized matrix
    matrix[2:4] = matrix[2:4]/norm2
    matrix[4:6] = matrix[4:6] / norm3
    #matrix[6:] = matrix[6:] / norm4
    '''
    #for index, slice in enumerate(matrix):
    #    norm = np.linalg.norm(slice)
    #    matrix[index] = slice/norm  # normalized matrix

    norm = np.linalg.norm(matrix)
    matrix = matrix/norm  # normalized matrix
    return matrix

n_array = normalize_2d(array)
print(n_array)

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.imshow(n_array, cmap = 'Blues')
#plt.ylabel('Training epoch / token number')
#plt.ylabel('Training epoch and model type')
#plt.xticks([0,1,2,3,4,5], [0,2,4,6,8,10])
#plt.yticks([0,1,2,3], ["1.a","2.a","1.b","2.b"])
#plt.yticks([0,1,2,3,4,5], ["1.a","2.a","1.b","2.b","1.c","2.c"])
#plt.yticks([0,1,2,3],[1,2,3,4])
plt.yticks([],[])
#plt.yticks([0,1,2,3],["1/2","2/2","1/4","2/4"])  #--
#plt.ylabel('Training epoch and probing task')    #--
#a2 = ax.twinx()
#a2.set_yticks([2,4])
#a2.invert_yaxis()
#a2.set_ylabel('Number of input representation tokens')
plt.title('COREF Bert-base')
#plt.title('Comparing the probing results of distilBERT (a), BERT (b) and RoBERTa (c)')
#plt.title('Comparing the probing results of BERT (a) and T5 (b)')
#plt.title('Probing results using a two layered mlp probing classifier')
#plt.title('Probing results using multiple representation tokens')
#plt.title('QUES probing in dependence on the training epochs')
plt.xlabel('Layer')
#txt = ax.text(6, 0.5, "a: QUES probing task on the 'trac' dataset", ha='center', va='center', wrap=True)
#txt._get_wrap_line_width = lambda: 20
#txt2 = ax.text(6, 2.5, "b: REL probing task on the 'sem_eval_2010_task_8' dataset", ha='center', va='center', wrap=True)
#txt2._get_wrap_line_width = lambda: 20
#cbar = fig.colorbar(cax)

#fig.savefig('plots/disitlBERT_epoch_influence.png')
#fig.savefig('plots/BERT.png')
#fig.savefig('plots/disitlBERT_token_influence.png')
#fig.savefig('plots/disitlBERT_mlplayers_influence.png')
#fig.savefig('plots/architecture_distilBERT_BERT_RoBERTa.png')
#fig.savefig('plots/architecture_BERT_T5.png')
#fig.savefig('plots/reproduction.png')
#fig.savefig('plots/probing.png')