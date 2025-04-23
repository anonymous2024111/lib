import sys
sys.path.append('./eva/accuracy/agnn')
from mypyg.agnn_pyg import AGNN, train
from mypyg.mdataset import *

def test(data, epoches, layers, hidden):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        #start_time = time.time()
        inputInfo = MGCN_dataset(data).to(device)

        model = AGNN(inputInfo.num_features, hidden, inputInfo.num_classes, layers).to(device)
    
        train(inputInfo.edge_index, inputInfo.x, inputInfo.y, inputInfo.train_mask, inputInfo.val_mask,model, epoches)

    
if __name__ == "__main__":
    dataset = ['pubmed']

    test('/home/shijinliang/module/git-flashsprase-ae2/dataset/pubmed.npz', 300, 5, 100)