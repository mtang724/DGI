import torch
import torch.nn as nn
from layers import GCN, AvgReadout, Discriminator
import torch.nn.functional as F

class DGI(nn.Module):
    def __init__(self, n_in, n_h, activation):
        super(DGI, self).__init__()
        self.gcn = GCN(n_in, n_h, activation)

        # self.gcn1 = GCN(n_h, n_h, activation)
        # #
        # self.gcn2 = GCN(n_h, n_h, activation)

        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(n_h)

    def forward(self, seq1, seq2, adj, sparse, msk, samp_bias1, samp_bias2):
        h_1 = self.gcn(seq1, adj, sparse)

        # h_1 = self.gcn1(h_1, adj, sparse)
        # #
        # h_1 = self.gcn2(h_1, adj, sparse)

        c = self.read(h_1, msk)
        c = self.sigm(c)

        h_2 = self.gcn(seq2, adj, sparse)

        # h_2 = self.gcn1(h_2, adj, sparse)
        # #
        # h_2 = self.gcn2(h_2, adj, sparse)

        ret = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)

        return ret

    # Detach the return variables
    def embed(self, seq, adj, sparse, msk):
        # print(seq)
        h_1 = self.gcn(seq, adj, sparse)
        # print(h_1)
        # h_2 = self.gcn1(h_1, adj, sparse)
        # print(h_2)
        # h_3 = self.gcn2(h_2, adj, sparse)
        c = self.read(h_1, msk)

        return seq.detach(), h_1.detach(), 0, 0, c.detach()

