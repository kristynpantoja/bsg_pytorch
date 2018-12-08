import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F

class BSG(nn.Module):
    def __init__(self, unigram_dict, vocab_size, input_dim=50, hidden_dim=50, latent_dim=100, margin=1., model_name='BSG with the hinge loss'):
        super().__init__()
        """
        :param vocab_size: the number of unique words
        :param input_dim: the number of components in the encoder's word embeddings
        :param hidden_dim: the number of components in the encoder's hidden layer
        :param latent_dim: the number of components in the latent vector(also output word mu's)
        :param margin: margin constant present in the hinge loss
        """
        self.model_name = model_name
        self.vocab_size = vocab_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.margin = margin

        self.losses = []

        self.unigram_dist = torch.distributions.Categorical(torch.tensor(list(unigram_dict.values())))

        # encoder layers
        self.encoder_embedding = nn.Embedding(self.vocab_size, self.input_dim, padding_idx = 0) # R
        self.encoder_lin1 = nn.Linear(self.input_dim*2, self.hidden_dim) # M
        self.encoder_mu = nn.Linear(self.hidden_dim, self.latent_dim) # U -> mu
        self.encoder_logsigma = nn.Linear(self.hidden_dim, 1) # W -> log sigma

        # word embeddings' parameters for normal distributions of word types
        self.type_means = nn.Embedding(self.vocab_size, self.latent_dim)
        self.type_logvars = nn.Embedding(self.vocab_size, 1)


    def encoder(self, centers_batch, contexts_batch):
        sums = []
        for center, context in zip(centers_batch, contexts_batch):
            embed_center = self.encoder_embedding(center)
            embed_context = self.encoder_embedding(context)
            assert embed_context.shape[1] == self.input_dim, "context embedding is not a 2d tensor"
            center_repeats = embed_center.repeat(2*window, 1)
            concat = torch.cat((embed_context, center_repeats),1)
            sum_relu_en1 = F.relu(self.encoder_lin1(concat)).sum(0) # a vector
            sums.append(sum_relu_en1) # the vectors of sums
        sums = torch.stack(sums)
        mu = self.encoder_mu(sums)
        logsigma = self.encoder_logsigma(sums)
        return mu, logsigma

    def reparameterize(self, centers_batch, posterior_mean, posterior_logvar):
        eps = Variable(centers_batch.data.new().resize_as_(posterior_mean.data).normal_())
        z = posterior_mean + posterior_logvar.exp().sqrt() * eps
        return z

    def KL(self, word_idx, post_mu, post_logsigma):
        post_sigma = post_logsigma.exp()
        type_mean = self.type_means(word_idx)
        type_var = self.type_logvars(word_idx).exp().view([-1,post_logsigma.shape[1]])
        var_division = post_sigma / type_var
        diff = post_mu - type_mean
        diff_term = (diff * diff).sum(1) / type_var
        logvar_division = type_var.log() - post_logsigma
        # compute KL
        KLD = 0.5 * ( (var_division + diff_term + logvar_division).sum(1) - self.latent_dim )
        return KLD

    def forward(self, centers_batch, contexts_batch):
        mu, logsigma = model.encoder(centers_batch, contexts_batch)
        # repeat mu, logsigma 2*window times,
        #  one for each context word in an input
        #  - do this for each input in the batch
        mus = mu.repeat(1,window*2).view(-1,mu.shape[1])
        logsigmas = logsigma.repeat(1,window*2).view(-1,logsigma.shape[1])
        # compute KLs
        KL_contexts = self.KL(contexts_batch.view([-1,1]), mus, logsigmas)
        negative_contexts_batch = self.unigram_dist.sample(contexts_batch.shape) + 1
        KL_negative_contexts = self.KL(negative_contexts_batch.view([-1,1]), mus, logsigmas)
        KL_center_word = self.KL(centers_batch, mu, logsigma)

        # compute hard margin of KLs of negative and positive context words
        hard_margin_arg = KL_contexts - KL_negative_contexts + self.margin
        loss = torch.max(hard_margin_arg, torch.zeros_like(hard_margin_arg)).sum() + KL_center_word.sum()
        return loss / centers_batch.shape[0]

def train(model, args, optimizer, center_words, context_words):
    '''
    model - object of class BSG
    args - dict of args
    optimizer - nn.optim
    centers_batch, contexts_batch
    '''
    for epoch in range(args.num_epoch):
        all_indices = torch.randperm(context_words.size(0)).split(args.batch_size)
        loss_epoch = 0.0
        model.train()                   # switch to training mode
        for batch_indices in all_indices:
            if not args.nogpu: batch_indices = batch_indices.cuda()
            context_words_input = Variable(context_words[batch_indices])
            center_words_input = Variable(center_words[batch_indices])
            loss = model(center_words_input, context_words_input)
            # optimize
            optimizer.zero_grad()       # clear previous gradients
            loss.backward()             # backprop
            optimizer.step()            # update parameters
            # report
            loss_epoch += loss.data[0]    # add loss to loss_epoch
        if epoch % 5 == 0:
            print('Epoch {}, loss={}'.format(epoch, loss_epoch / len(all_indices)))
        model.losses.append(loss_epoch / len(all_indices))

    return model
