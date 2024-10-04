import hypll.nn as hnn

from hypll.manifolds.poincare_ball import PoincareBall
import torch

class PoincareEmbedding(hnn.HEmbedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        manifold: PoincareBall,
    ):
        super().__init__(num_embeddings, embedding_dim, manifold)

    # The model outputs the distances between the nodes involved in the input edges as these are
    # used to compute the loss.
    def forward(self, edges: torch.Tensor) -> torch.Tensor:
        embeddings = super().forward(edges)
        edge_distances = self.manifold.dist(x=embeddings[:, :, 0, :], y=embeddings[:, :, 1, :])
        return edge_distances
