import torch
from sklearn.metrics import accuracy_score
import dgl
import torch.nn.functional as F


def predict(graph, model, batch_size, device):
    graph.ndata["h"] = graph.ndata["feat"]
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
    data_loader = dgl.dataloading.DataLoader(
        graph,
        torch.arange(graph.number_of_nodes()).to(device),
        sampler,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        device=device,
        num_workers=0,
        use_uva=True
    )
    
    for l, layer in enumerate(model.layers):
        y = torch.zeros(
            graph.num_nodes(),
            model.n_hidden if l != len(model.layers) - 1 else model.n_classes,
            device='cpu'
        )
        for input_nodes, output_nodes, blocks in data_loader:
            block = blocks[0]
            x = block.srcdata['h']
            h = layer(block, x)
            if l != len(model.layers) - 1:
                h = F.relu(h)
                h = model.dropout(h)

            y[output_nodes] = h.to('cpu')

        graph.ndata["h"] = y

    del graph.ndata['h']
    return y

def test_loop(graph,model,device,test_idx):
    predict_batch_size = 4096

    with torch.no_grad():
        pred = predict(graph, model.to(device), predict_batch_size, device)
        pred = pred[test_idx]
        _, predicted = torch.max(pred, 1)
        label = graph.ndata["label"][test_idx]
        accuracy = accuracy_score(label,predicted)
        accuracy = round(accuracy.item(), 3)

    print("Test accuracy:", accuracy)


