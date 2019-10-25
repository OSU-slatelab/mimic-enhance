import torch
from data_io import mag

class Embedding(torch.nn.Module):

    def __init__(
        self,
        embedding_size,
    ):
        super(Embedding, self).__init__()

        self.embedding_size = embedding_size

        weight = torch.zeros((embedding_size, embedding_size))
        self.embedding = torch.nn.Embedding.from_pretrained(weight)

    def pretrain(self, training_set, mimic_model, device):
        """ This is a hack of embeddings and autograd to allow soft
            'prototypical' posterior distributions for each senone. """

        # Enable grad, so we can hack it to update our senone "embedding"
        self.embedding.weight.requires_grad = True

        # Initialize counts to a small value, so we don't divide by zero
        senone_counts = torch.zeros([self.embedding_size, 1]).to(device) + 0.01

        # Go through dataset, and add up posteriors and counts
        for example in training_set:

            # Generate posterior
            clean_mag = mag(example['clean'].to(device), truncate=True)
            senones = example['senone'].to(device)

            posteriors = mimic_model(clean_mag)[-1]
            posteriors = posteriors[:,:,:senones.shape[1]].transpose(1, 2)

            # Embed senone, so we can update the result
            embedded = self.embedding(senones)
            
            # Multiply posteriors so that we can add to the gradient
            embedded *= posteriors

            # Propagate gradient to the embedding
            embedded.sum().backward()

            # Count instances of senones
            example_senone_counts = senones[0].bincount(minlength = self.embedding_size).float().unsqueeze(1)
            senone_counts += example_senone_counts

            # Divide and update
            with torch.no_grad():
                self.embedding.weight *= (senone_counts - example_senone_counts) / senone_counts
                self.embedding.weight += self.embedding.weight.grad / senone_counts
                self.embedding.weight.grad.zero_()

        # Turn off grad again
        self.embedding.weight.requires_grad = False

    def forward(self, x):
        return self.embedding(x)
