import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------- 1. Subset + convert networkx DiGraph -> PyG ----------
def subset_graph(G, n_nodes=100):
    """Induced subgraph on the top-n nodes by weighted degree (keeps it dense)."""
    deg = dict(G.degree(weight="weight"))
    keep = sorted(deg, key=deg.get, reverse=True)[:n_nodes]
    H = G.subgraph(keep).copy()
    H.remove_nodes_from(list(nx_isolates(H)))
    return H


def nx_isolates(H):
    import networkx as nx
    return list(nx.isolates(H))


def to_pyg(G):
    """
    Convert a networkx DiGraph to a PyG Data object. 
    This allows us to use the PyTorch Geometric library for graph neural networks. 
    The function extracts node features, edge indices, and edge attributes from the networkx graph and constructs a PyG Data object.
    """
    nodes = list(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    src, dst, w = zip(*[(idx[u], idx[v], d.get("weight", 1.0)) for u, v, d in G.edges(data=True)])

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.log1p(torch.tensor(w, dtype=torch.float))  # tame heavy-tailed weights

    # stance features: [has_trump, has_harris, unlabeled]
    # Current only using the stance features but this can be extended to add other node features
    feats = []
    for n in nodes:
        t = G.nodes[n].get("Trump") is not None
        h = G.nodes[n].get("Harris") is not None
        feats.append([float(t), float(h), float(not (t or h))])
    x = torch.tensor(feats, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=len(nodes))


# ---------- 2. Split ----------
def split_data(data):
    """Split the graph into train/val/test for link prediction."""
    split = RandomLinkSplit(
        num_val=0.2,
        num_test=0.2,
        is_undirected=False,            # directed graph: (u->v) != (v->u)
        disjoint_train_ratio=0.3,       # 30% of train edges used only as supervision, not message passing, this prevents information leakage
        add_negative_train_samples=False,  # train negatives are resampled each epoch
        neg_sampling_ratio=1.0,         # 1 negative per positive for val/test
    )
    return split(data)


# ---------- 3. Model ----------
class LinkPredictor(nn.Module):
    """
    Current model for initial test is a 2-layer GCN with asymmetric decoder for directed graphs.
    The decoder is important because it respects the direction of edges, which is crucial for directed 
    graphs like reposts, likes, quotes, and replies.
    """
    def __init__(self, num_nodes, in_dim, emb_dim=64, hidden=64):
        super().__init__()
        self.emb = nn.Embedding(num_nodes, emb_dim)  # learnable ID embedding (transductive)
        self.conv1 = GCNConv(emb_dim + in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        # asymmetric decoder -> respects edge direction
        self.decoder = nn.Sequential(
            nn.Linear(hidden * 3, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )
        """Initialize the link predictor."""

    def encode(self, x, edge_index, edge_weight=None):
        """encodes node features + embeddings into latent space."""
        h = torch.cat([self.emb.weight, x], dim=1)
        h = F.relu(self.conv1(h, edge_index, edge_weight))
        h = F.dropout(h, p=0.3, training=self.training)
        return self.conv2(h, edge_index, edge_weight)

    def decode(self, z, edge_label_index):
        """decodes latent node embeddings into edge probabilities."""
        zu, zv = z[edge_label_index[0]], z[edge_label_index[1]]
        return self.decoder(torch.cat([zu, zv, zu * zv], dim=1)).squeeze(-1)


# ---------- 4. Train / eval ----------
def train_step(model, opt, data):
    """Train the model for one step."""
    model.train()
    opt.zero_grad()
    z = model.encode(data.x, data.edge_index, data.edge_attr)

    pos = data.edge_label_index
    # avoid sampling any known edge as a negative
    known = torch.cat([data.edge_index, pos], dim=1)
    neg = negative_sampling(known, num_nodes=data.num_nodes, num_neg_samples=pos.size(1))

    idx = torch.cat([pos, neg], dim=1)
    y = torch.cat([torch.ones(pos.size(1)), torch.zeros(neg.size(1))]).to(pos.device)
    # compute loss and backprop
    loss = F.binary_cross_entropy_with_logits(model.decode(z, idx), y)
    loss.backward()
    opt.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data):
    """
    Evaluate the model on a validation/test set.
    Returns:  roc_auc_score which is the area under the ROC curve which dictates the capacity of the model to separate real links 
    from non-existent links, and average_precision_score which focuses on whether predicted links are actually correct especially 
    in imbalanced datasets where the number of non-links is much larger than the number of links.
    """
    model.eval()
    z = model.encode(data.x, data.edge_index, data.edge_attr)
    prob = torch.sigmoid(model.decode(z, data.edge_label_index)).cpu().numpy()
    y = data.edge_label.cpu().numpy()
    return roc_auc_score(y, prob), average_precision_score(y, prob)


def run_link_prediction(G, n_nodes=2000, epochs=200, lr=0.01, patience=20, name=""):
    H = subset_graph(G, n_nodes)
    data = to_pyg(H)
    print(f"[{name}] subgraph: {data.num_nodes} nodes, {data.edge_index.size(1)} edges")
    # Print size of training set
    print(f"[{name}] training set: {int(data.num_edges * 0.6)} edges")
    # Print the size of validation and test sets
    print(f"[{name}] validation set: {int(data.num_edges * 0.2)} edges, test set: {int(data.num_edges * 0.2)} edges")

    train_d, val_d, test_d = split_data(data)
    train_d, val_d, test_d = train_d.to(device), val_d.to(device), test_d.to(device)

    model = LinkPredictor(data.num_nodes, data.x.size(1)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    best_val, best_state, bad = 0, None, 0

    # training loop
    for epoch in range(1, epochs + 1):
        loss = train_step(model, opt, train_d)
        val_auc, val_ap = evaluate(model, val_d)
        if val_auc > best_val:
            best_val, bad = val_auc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
        if epoch % 10 == 0:
            print(f"  epoch {epoch:3d} | loss {loss:.4f} | val AUC {val_auc:.4f} AP {val_ap:.4f}")

    model.load_state_dict(best_state)
    test_auc, test_ap = evaluate(model, test_d)


    print(f"[{name}] TEST AUC {test_auc:.4f} | AP {test_ap:.4f}\n")
    return {"graph": name, "val_auc": best_val, "test_auc": test_auc, "test_ap": test_ap}
