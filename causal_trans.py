

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.distributions as dist
from .networks import ContentEncoder, ContentDecoder, Conv2dBlock, ResBlocks, DualAttentionBlock
from pretrainedmodels import resnet34

class ResnetEncoder(nn.Module):
    def __init__(self):
        super(ResnetEncoder, self).__init__()
        self.model = resnet34(pretrained='imagenet')
#         self.model = resnet34(pretrained=None)


    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        return x

class fc_net(nn.Module):
    def __init__(self, out_dim, use_sigmoid=False, use_bias=False):
        super(fc_net, self).__init__()
        # image: (3 x 224 x 224)
        # insize: (batch_size x 256 x 56 x 56)
        self.use_sigmoid = use_sigmoid
        
        self.fc = nn.Sequential(*[
                    nn.AdaptiveAvgPool2d((1, 1)),
                    Flatten(),
                    nn.Linear(512, out_dim, bias=use_bias)
                    ]) 
        
    def forward(self, x):
        x = self.fc(x)
        if self.use_sigmoid:
            x = torch.sigmoid(x)
        return x

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Encoder(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, dim, activ, pad_type, use_attention=False):
        super(Encoder, self).__init__()

        self.emb = ResnetEncoder()
        #self.emb = ContentEncoder(n_downsample, n_res, input_dim, dim, 'in', activ, pad_type=pad_type)
        # q(t|x)
        self.logits_t = fc_net(out_dim=1, use_sigmoid=True)
        # q(y|x,t)
        if use_attention:
            self.hqy = nn.Sequential(ResBlocks(1, 512), 
                                DualAttentionBlock(512, 512))
        else:
            self.hqy = ResBlocks(2, 512)
        self.qy_t0 = fc_net(out_dim=1, use_sigmoid=True)
        self.qy_t1 = fc_net(out_dim=1, use_sigmoid=True)
        # q(z|x,t,y)
        if use_attention:
            self.hqz = nn.Sequential(ResBlocks(1, 512), 
                                     DualAttentionBlock(512, 512))
        else:
            self.hqz = ResBlocks(2, 512)

        self.muq_t0 = ResBlocks(1, 512, use_bias=False)
        self.sigmaq_t0 = ResBlocks(1, 512)
        self.muq_t1 = ResBlocks(1, 512, use_bias=False)
        self.sigmaq_t1 = ResBlocks(1, 512)

    def reparameterize_normal(self, mu, sig):
        eps = torch.randn_like(sig)
        return mu + eps * sig

    def reparameterize_bernoulli(self, prob):
        eps = torch.rand(prob.shape)
        eps = eps.type(prob.type())
        return torch.sigmoid(torch.log(eps) - torch.log(1.-eps) + 
                             torch.log(prob) - torch.log(1.-prob))

    def forward(self, x, t=None):
        x = self.emb(x)
        # CEVAE
        # q(t|x)
        logits_t = self.logits_t(x)
        qt = dist.bernoulli.Bernoulli(logits_t)

        # q(y|x,t)
        hqy = self.hqy(x)
        qy_t0 = self.qy_t0(hqy)
        qy_t1 = self.qy_t1(hqy)
        
        if self.train:
            qt_sample = t.view(-1, 1, 1, 1).contiguous()
        else:
            qt_sample = qt.sample().view(-1, 1, 1, 1).contiguous()
        
        qy = qt_sample * qy_t1 + (1. - qt_sample) * qy_t0
        
        # q(z|x,t,y)
        hqz = self.hqz(x * qy.view(qy.shape[0], 1, 1, 1))
        
        muq_t0, sigmaq_t0 = self.muq_t0(hqz), F.softplus(self.sigmaq_t0(hqz))
        muq_t1, sigmaq_t1 = self.muq_t1(hqz), F.softplus(self.sigmaq_t1(hqz))

        qt_sample = qt_sample.view(qt_sample.shape[0], 1, 1, 1)
        z = self.reparameterize_normal(qt_sample * muq_t1 + (1. - qt_sample) * muq_t0,
                                       qt_sample * sigmaq_t1 + (1. - qt_sample) * sigmaq_t0)
        
        out = {}
        out['z'] = z
        out['qt'] = qt
        out['qy'] = qy 
        return out


def create_attn_fc(nin, nout, spectral_norm):
    conv = nn.Conv2d(in_channels=nin, out_channels=nout,
                     kernel_size=1, padding=0, bias=False)
    torch.nn.init.xavier_uniform(conv.weight)
    if spectral_norm:
        conv = nn.utils.spectral_norm(conv, eps=1e-12)
    return conv


class Attention(nn.Module):
    def __init__(self, query_nin, key_nin, val_nin, d_k, d_v, nout, spectral_norm):
        super().__init__()
        self.query_nin, self.key_nin, self.val_nin = query_nin, key_nin, val_nin
        self.d_k, self.d_v, self.nout = d_k, d_v, nout
        self.divisor = (self.d_k)**0.5

        self.query_conv = create_attn_fc(query_nin, d_k, spectral_norm)
        self.key_conv = create_attn_fc(key_nin, d_k, spectral_norm)
        self.val_conv = create_attn_fc(val_nin, d_v, spectral_norm)
        self.output_conv = create_attn_fc(d_v, nout, spectral_norm)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, query_in, key_in, val_in):
        query_h, query_w = tuple(query_in.shape[2:])
        key_h, key_w = tuple(key_in.shape[2:])
        def unroll(x):
            nin, h, w = tuple(x.shape[1:])
            return x.view(-1, nin, h*w)

        queries = unroll(self.query_conv(query_in))  # B x d_k x HW
        keys = unroll(self.key_conv(key_in))  # B x d_k x H'W'
        values = unroll(self.val_conv(val_in))  # B x d_v x H'W'

        keys_t = keys.transpose(1, 2)  # B x H'W' x d_k
        attended_coeff = self.softmax(
            torch.bmm(keys_t, queries)/self.divisor)  # B x H'W' x HW
        attended_vals = torch.bmm(values, attended_coeff)  # B x d_v x HW

        reshaped_attended_vals = attended_vals.view(-1, self.d_v, query_h, query_w)
        output = self.output_conv(reshaped_attended_vals)

        return output


class Encoder_v2(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, dim, activ, pad_type, use_attention=False):
        super(Encoder_v2, self).__init__()

        self.emb = ResnetEncoder()
        #self.emb = ContentEncoder(n_downsample, n_res, input_dim, dim, 'in', activ, pad_type=pad_type)
        # q(t|x)
        self.logits_t = fc_net(out_dim=1, use_sigmoid=True)

        # q(y|x,t)
        self.hqy_t0 = ResBlocks(1, 512)
        self.hqy_t1 = ResBlocks(1, 512)
        self.qy = fc_net(out_dim=1, use_sigmoid=True)
        self.attn = Attention(512, 512, 512, 128, 128, 512, True)

        # q(z|x,t,y)
        self.hqz = ResBlocks(1, 512)

        self.muq_t0 = ResBlocks(1, 512, use_bias=False)
        self.sigmaq_t0 = ResBlocks(1, 512)
        self.muq_t1 = ResBlocks(1, 512, use_bias=False)
        self.sigmaq_t1 = ResBlocks(1, 512)

    def reparameterize_normal(self, mu, sig):
        eps = torch.randn_like(sig)
        return mu + eps * sig

    def reparameterize_bernoulli(self, prob):
        eps = torch.rand(prob.shape)
        eps = eps.type(prob.type())
        return torch.sigmoid(torch.log(eps) - torch.log(1.-eps) + 
                             torch.log(prob) - torch.log(1.-prob))

    
    def forward(self, x, t=None):
        x = self.emb(x)

        # CEVAE
        # q(t|x)
        logits_t = self.logits_t(x)
        qt = dist.bernoulli.Bernoulli(logits_t)

        # q(y|x,t)
        hqy_t0 = self.hqy_t0(x)
        hqy_t1 = self.hqy_t1(x)
        
        if self.train:
            qt_sample = t.view(-1, 1, 1, 1).contiguous()
        else:
            qt_sample = qt.sample().view(-1, 1, 1, 1).contiguous()
        
        query = qt_sample * hqy_t1 + (1. - qt_sample) * hqy_t0
        qy = self.qy(query)
        hqz = self.hqz(self.attn(query, x, x))
        
        muq_t0, sigmaq_t0 = self.muq_t0(hqz), F.softplus(self.sigmaq_t0(hqz))
        muq_t1, sigmaq_t1 = self.muq_t1(hqz), F.softplus(self.sigmaq_t1(hqz))

        qt_sample = qt_sample.view(qt_sample.shape[0], 1, 1, 1)
        z = self.reparameterize_normal(qt_sample * muq_t1 + (1. - qt_sample) * muq_t0,
                                       qt_sample * sigmaq_t1 + (1. - qt_sample) * sigmaq_t0)
        
        out = {}
        out['z'] = z
        out['qt'] = qt
        out['qy'] = qy 
        return out

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Encoder_v3(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, dim, activ, pad_type, use_attention=False):
        super(Encoder_v2, self).__init__()

        self.emb = ResnetEncoder()
        #self.emb = ContentEncoder(n_downsample, n_res, input_dim, dim, 'in', activ, pad_type=pad_type)
        # q(t|x)
        self.logits_t = fc_net(out_dim=1, use_sigmoid=True)

        # q(y|x,t)
        self.hqy_t0 = ResBlocks(1, 512)
        self.hqy_t1 = ResBlocks(1, 512)
        self.qy = fc_net(out_dim=1, use_sigmoid=True)
        self.attn = Attention(512, 512, 512, 128, 128, 512, True)

        # q(z|x,t,y)
        self.hqz = ResBlocks(1, 512)

        self.muq_t0 = ResBlocks(1, 512, use_bias=False)
        self.sigmaq_t0 = ResBlocks(1, 512)
        self.muq_t1 = ResBlocks(1, 512, use_bias=False)
        self.sigmaq_t1 = ResBlocks(1, 512)
        self.conv1 = conv1x1(1, 512)

    def reparameterize_normal(self, mu, sig):
        eps = torch.randn_like(sig)
        return mu + eps * sig

    def reparameterize_bernoulli(self, prob):
        eps = torch.rand(prob.shape)
        eps = eps.type(prob.type())
        return torch.sigmoid(torch.log(eps) - torch.log(1.-eps) + 
                             torch.log(prob) - torch.log(1.-prob))

    
    def forward(self, x, t=None):
        x = self.emb(x) # Resnet34 features

        # CEVAE
        # q(t|x)
        logits_t = self.logits_t(x)
        qt = dist.bernoulli.Bernoulli(logits_t)

        # q(y|x,t)
        hqy_t0 = self.hqy_t0(x)
        hqy_t1 = self.hqy_t1(x)

        # q(z|x) for spatial information from resnet 34
        qz = conv1(x)
        
        if self.train:
            qt_sample = t.view(-1, 1, 1, 1).contiguous()
        else:
            qt_sample = qt.sample().view(-1, 1, 1, 1).contiguous()
        
        query = qt_sample * hqy_t1 + (1. - qt_sample) * hqy_t0
        qy = self.qy(query)
        hqz = self.hqz(self.attn(query, x, x))
        
        # resdiual from 1x1 conv
        hqz += qz

        muq_t0, sigmaq_t0 = self.muq_t0(hqz), F.softplus(self.sigmaq_t0(hqz))
        muq_t1, sigmaq_t1 = self.muq_t1(hqz), F.softplus(self.sigmaq_t1(hqz))

        qt_sample = qt_sample.view(qt_sample.shape[0], 1, 1, 1)
        z = self.reparameterize_normal(qt_sample * muq_t1 + (1. - qt_sample) * muq_t0,
                                       qt_sample * sigmaq_t1 + (1. - qt_sample) * sigmaq_t0)
        
        out = {}
        out['z'] = z
        out['qt'] = qt
        out['qy'] = qy 
        return out

class Decoder(nn.Module):
    def __init__(self, n_downsample, n_res, output_dim, input_dim, res_norm, activ, pad_type):
        super(Decoder, self).__init__()
        # p(x|z)
        #self.recon = ContentDecoder(n_downsample, n_res, output_dim, input_dim, res_norm='in', activ=activ, pad_type=pad_type)
        #self.recon = ContentDecoder(5, 3, 512, 3, res_norm='bn', activ=activ, pad_type=pad_type)
        # p(t|z)
        #self.logits_t = fc_net(out_dim=1, use_sigmoid=True)

        # p(y|t,z)
        #self.mu2_t0 = fc_net(out_dim=1, use_sigmoid=True)
        self.out = fc_net(out_dim=1, use_sigmoid=True)
        
    def reparameterize_normal(self, mu, sig):
        eps = torch.randn_like(sig)
        return mu + eps * sig

    def reparameterize_bernoulli(self, prob):
        eps = torch.rand(prob.shape)
        eps = eps.type(prob.type())
        return torch.sigmoid(torch.log(eps) - torch.log(1.-eps) + 
                             torch.log(prob+1e-20) - torch.log(1.-prob))
    
    def forward(self, z, _t=None):
        out = {}
        z = z.view(z.shape[0], 512, 4, 4)
        # p(x|z)
        # TODO(02/18/20 Danny): Do we reconstruct interveined or uninterveined image?
        # It's kind of weird to reconstruct interveined one because it's more reasonable
        # to learn a "uninterveined confounder". 
        # If you agree, the encoder should learn q(t|do(x)), q(y|do(x), t) and q(z|do(x), t, y)
        #           and the decoder should learn p(no_do(x)|z) and and p(y|z). 
        #x = self.recon(z)
        
        # CEVAE
        # p(t|z)
        #logits_t = self.logits_t(z)
        #t = dist.bernoulli.Bernoulli(logits_t)
        
        # p(y|t,z)
        y_out = self.out(z)
        #py_t1 = self.mu2_t1(z)
        
        # 這個y_out是拿來做training的（for criterion)
        #if self.train:
        #    t_sample = _t
        #else:
        #    t_sample = t
        #y_out = t_sample * py_t1 + (1. - t_sample) * py_t0
        

        # VAE
        #y_out = self.mu2_t0(z)

        #out['x'] = x
        #out['t'] = t
        out['y_out'] = y_out
        
        #y = torch.where(_t == 0, py_t0, py_t1)
        #y = y_out
        #out['y'] = y
        return out


class CEVAE(nn.Module):
    def __init__(self, input_dim, dim, n_downsample, n_res, norm, activ, pad_type, use_attention):
        super(CEVAE, self).__init__()
        self.encoder = Encoder(n_downsample, n_res, input_dim, dim, activ, pad_type, use_attention)
        self.decoder = Decoder(n_downsample, n_res, dim * (2**n_downsample), input_dim, norm, activ, pad_type)
        
    def forward(self, x):
        # This is a reduced VAE implementation where we assume 
        # the outputs are multivariate Gaussian distribution with 
        # mean = hiddens and std_dev = all ones.
        out = {}
        enc_out = self.encoder(x['x'], x['t'])
        z = enc_out['z']
        # CEVAE
        dec_out = self.decoder(z, x['t'])
        # VAE
        #dec_out = self.decoder(z)

        out.update(enc_out)
        out.update(dec_out)
        return out


class CEVAE_Att(nn.Module):
    def __init__(self, input_dim, dim, n_downsample, n_res, norm, activ, pad_type, use_attention):
        super(CEVAE_Att, self).__init__()
        self.encoder = Encoder_v2(n_downsample, n_res, input_dim, dim, activ, pad_type, use_attention)
        self.decoder = Decoder(n_downsample, n_res, dim * (2**n_downsample), input_dim, norm, activ, pad_type)
        
    def forward(self, x):
        # This is a reduced VAE implementation where we assume 
        # the outputs are multivariate Gaussian distribution with 
        # mean = hiddens and std_dev = all ones.
        out = {}
        enc_out = self.encoder(x['x'], x['t'])
        z = enc_out['z']
        # CEVAE
        dec_out = self.decoder(z, x['t'])
        # VAE
        #dec_out = self.decoder(z)

        out.update(enc_out)
        out.update(dec_out)
        return out

class Causal_Transformer(nn.Module):
    def __init__(self, input_dim, dim, n_downsample, n_res, norm, activ, pad_type, use_attention):
        super(Causal_Transformer, self).__init__()
        self.encoder = Encoder_v3(n_downsample, n_res, input_dim, dim, activ, pad_type, use_attention)
        self.decoder = Decoder(n_downsample, n_res, dim * (2**n_downsample), input_dim, norm, activ, pad_type)
        
    def forward(self, x):
        # This is a reduced VAE implementation where we assume 
        # the outputs are multivariate Gaussian distribution with 
        # mean = hiddens and std_dev = all ones.
        out = {}
        enc_out = self.encoder(x['x'], x['t'])
        z = enc_out['z']
        # CEVAE
        dec_out = self.decoder(z, x['t'])
        # VAE
        #dec_out = self.decoder(z)

        out.update(enc_out)
        out.update(dec_out)
        return out


if __name__ == '__main__':
    m = Encoder(2, 4, 3, 64, 'relu', 'reflect', 50, 2, 100, 'relu').cuda()
    a = torch.rand(1, 3, 224, 224).cuda()
    d = Decoder(2, 4, 256, 3, 'in', 'relu', 'reflect', 50, 2, 100, 'relu').cuda()
    import ipdb; ipdb.set_trace()    

