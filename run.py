
import json
import os
import networkx as nx
import polars as pl

##########################
mammals_json_file = os.path.join("data", "opehr_concepts.csv")

df = pl.read_csv(mammals_json_file)

# Find unique nodes by combining ancestors and descendants
# Use Polars' unique function to get unique values from both columns efficiently
unique_ancestors = df['ancestor_concept_id'].unique()
unique_descendants = df['descendant_concept_id'].unique()

# Concatenate and get unique values once again to ensure full coverage
all_unique_nodes = pl.concat([unique_ancestors, unique_descendants]).unique()

# Use list comprehension with enumerate to create the mapping efficiently
node_to_index = {node: idx for idx, node in enumerate(all_unique_nodes)}

# Transform edges to use these indices
# Instead of converting to a list first, iterate directly over Polars Series
indexed_edges = [(node_to_index[ancestor], node_to_index[descendant])
                 for ancestor, descendant in zip(df['ancestor_concept_id'], df['descendant_concept_id'])]

# Initialize a NetworkX graph and add the transformed edges
G = nx.DiGraph()
G.add_edges_from(indexed_edges)

# Save node-to-index mapping as a CSV using Polars
mapping_df = pl.DataFrame({
    'original_id': list(node_to_index.keys()),
    'node_index': list(node_to_index.values())
})

# Save to 'output/node_mapping.csv'
output_path = 'output/node_mapping.csv'
os.makedirs('output', exist_ok=True)  # Ensure the output directory exists
mapping_df.write_csv(output_path)


# Create a dataset containing the graph from which we can sample

from MammalsEmbeddingDataset import MammalsEmbeddingDataset
from torch.utils.data import DataLoader

# Now, we construct the dataset.
dataset = MammalsEmbeddingDataset(mammals=G,)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# Initialize the Poincare ball on which the embeddings will be trained

from hypll.manifolds.poincare_ball import Curvature, PoincareBall

poincare_ball = PoincareBall(Curvature(1.0))

# Define the Poincare embedding model

from PoincareEmbedding import PoincareEmbedding

# We want to embed every node into a 2-dimensional Poincare ball.
model = PoincareEmbedding(
    num_embeddings=len(G.nodes()),
    embedding_dim=3,
    manifold=poincare_ball,
)

# model.to(device=).... # mps???

# Define the Poincare embedding loss function

import torch

# This function is given in equation (5) of the Poincare Embeddings paper.
def poincare_embeddings_loss(dists: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logits = dists.neg().exp()
    numerator = torch.where(condition=targets, input=logits, other=0).sum(dim=-1)
    denominator = logits.sum(dim=-1)
    loss = (numerator / denominator).log().mean().neg()
    return loss

# Perform a few “burn-in” training epochs with reduced learning rate

from hypll.optim import RiemannianSGD, RiemannianAdam

# The learning rate of 0.3 is dived by 10 during burn-in.
optimizer = RiemannianAdam(
    params=model.parameters(),
    lr=0.3 / 10,
)

# Perform training as we would usually.
for epoch in range(10):
    print(f"Lets go!")

    average_loss = 0
    for idx, (edges, edge_label_targets) in enumerate(dataloader):
        optimizer.zero_grad()

        # edges.todevice()...

        dists = model(edges)
        loss = poincare_embeddings_loss(dists=dists, targets=edge_label_targets)
        loss.backward()
        optimizer.step()

        average_loss += loss

    average_loss /= len(dataloader)
    print(f"Burn-in epoch {epoch} loss: {average_loss}")

# Train the embeddings

# Now we use the actual learning rate 0.3.
optimizer = RiemannianAdam(
    params=model.parameters(),
    lr=0.3,
)

for epoch in range(50):
    average_loss = 0
    for idx, (edges, edge_label_targets) in enumerate(dataloader):
        optimizer.zero_grad()

        dists = model(edges)
        loss = poincare_embeddings_loss(dists=dists, targets=edge_label_targets)
        loss.backward()
        optimizer.step()

        average_loss += loss

    average_loss /= len(dataloader)
    print(f"Epoch {epoch} loss: {average_loss}")
