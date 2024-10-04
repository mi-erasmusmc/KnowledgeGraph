import random

import torch
from torch.utils.data import Dataset
import networkx as nx


class MammalsEmbeddingDataset(Dataset):
    def __init__(
        self,
        mammals: nx.DiGraph,
    ):
        super().__init__()
        self.mammals = mammals
        self.edges_list = list(mammals.edges())

    # This Dataset object has a sample for each edge in the graph.
    def __len__(self) -> int:
        return len(self.edges_list)

    def __getitem__(self, idx: int):
        # For each existing edge in the graph we choose 10 fake or negative edges, which we build
        # from the idx-th existing edge. So, first we grab this edge from the graph.
        rel = self.edges_list[idx]

        # Next, we take our source node rel[0] and see which nodes in the graph are not a child of
        # this node.
        negative_target_nodes = list(
            self.mammals.nodes() - nx.descendants(self.mammals, rel[0]) - {rel[0]}
        )

        # Then, we sample at most 5 of these negative target nodes...
        # NOTE: this type of sampling is a straightforward, but inefficient method. If your dataset
        #       is larger, consider using more efficient sampling methods, as this will otherwise
        #       form a bottleneck. See (closed) Issue 59 on GitHub for some more information.
        negative_target_sample_size = min(5, len(negative_target_nodes))
        negative_target_nodes_sample = random.sample(
            negative_target_nodes, negative_target_sample_size
        )

        # and add these to a tensor which will be used as input for our embedding model.
        edges = torch.tensor([rel] + [[rel[0], neg] for neg in negative_target_nodes_sample])

        # Next, we do the same with our target node rel[1], but now where we sample from nodes
        # which aren't a parent of it.
        negative_source_nodes = list(
            self.mammals.nodes() - nx.ancestors(self.mammals, rel[1]) - {rel[1]}
        )

        # We sample from these negative source nodes until we have a total of 10 negative edges...
        negative_source_sample_size = 10 - negative_target_sample_size
        negative_source_nodes_sample = random.sample(
            negative_source_nodes, negative_source_sample_size
        )

        # and add these to the tensor that we created above.
        edges = torch.cat(
            tensors=(edges, torch.tensor([[neg, rel[1]] for neg in negative_source_nodes_sample])),
            dim=0,
        )

        # Lastly, we create a tensor containing the labels of the edges, indicating whether it's a
        # True or a False edge.
        edge_label_targets = torch.cat(tensors=[torch.ones(1).bool(), torch.zeros(10).bool()])

        return edges, edge_label_targets
