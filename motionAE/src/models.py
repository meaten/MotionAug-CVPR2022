import torch
import torch.nn as nn

torch.manual_seed(1)

class lstmEncoder_wo_Norm(nn.Module):
    def __init__(self, input_len, dim, dim_z, dropout=0.0):
        super(lstmEncoder_wo_Norm, self).__init__()
        
        self.input_len = input_len
        self.dim = dim
        self.dim_z = dim_z
        
        self.linear = nn.Linear(
            in_features=self.dim,
            out_features=self.dim_z
        )
        
        self.rnn = nn.LSTM(
            input_size=self.dim_z,
            hidden_size=self.dim_z * 2,
            num_layers=1,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        self.linear1 = nn.Linear(
            in_features=self.dim_z * 4,
            out_features=self.dim_z * 2
        )
        
        self.linear2 = nn.Linear(
            in_features=self.dim_z * 2,
            out_features=self.dim_z
        )
        
        self.activation = nn.ReLU()
            
    def forward(self, x, x_lengths):
        x = self.activation(self.linear(x))
        
        x = torch.nn.utils.rnn.pack_padded_sequence(x, x_lengths, batch_first=True, enforce_sorted=False)
        out, (_, _) = self.rnn(x)
        out, out_lengths = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        
        out = torch.sum(out, dim=1) / out_lengths[:, None].type(torch.FloatTensor).cuda()
        
        out = self.activation(self.linear1(out))
        z = self.linear2(out)
        
        return z

class lstmEncoder(nn.Module):
    def __init__(self, input_len, dim, dim_z, dropout=0.0):
        super(lstmEncoder, self).__init__()
        
        self.input_len = input_len
        self.dim = dim
        self.dim_z = dim_z
        
        self.linear = nn.Linear(
            in_features=self.dim,
            out_features=self.dim_z
        )
        
        self.norm = nn.LayerNorm([self.input_len, self.dim])
        
        self.rnn = nn.LSTM(
            input_size=self.dim_z,
            hidden_size=self.dim_z * 2,
            num_layers=1,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        self.linear1 = nn.Linear(
            in_features=self.dim_z * 4,
            out_features=self.dim_z * 2
        )
        
        self.linear2 = nn.Linear(
            in_features=self.dim_z * 2,
            out_features=self.dim_z
        )
        
        self.activation = nn.ReLU()
            
    def forward(self, x, x_lengths):
        x = self.norm(x)
        x = self.activation(self.linear(x))
        
        x = torch.nn.utils.rnn.pack_padded_sequence(x, x_lengths, batch_first=True, enforce_sorted=False)
        out, (_, _) = self.rnn(x)
        out, out_lengths = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        
        out = torch.sum(out, dim=1) / out_lengths[:, None].type(torch.FloatTensor).cuda()
        
        out = self.activation(self.linear1(out))
        z = self.linear2(out)
        
        return z
    
    
class lstmEncoder_vae(nn.Module):
    def __init__(self, input_len, dim, dim_z, dropout=0.0):
        super(lstmEncoder_vae, self).__init__()
        
        self.input_len = input_len
        self.dim = dim
        self.dim_z = dim_z
        
        self.linear = nn.Linear(
            in_features=self.dim,
            out_features=self.dim_z
        )
        
        self.norm = nn.LayerNorm([self.input_len, self.dim])
        
        self.rnn = nn.LSTM(
            input_size=self.dim_z,
            hidden_size=self.dim_z * 2,
            num_layers=1,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        self.mu_linear1 = nn.Linear(
            in_features=self.dim_z * 4,
            out_features=self.dim_z * 2
        )
        
        self.mu_linear2 = nn.Linear(
            in_features=self.dim_z * 2,
            out_features=self.dim_z
        )
        
        self.log_var_linear1 = nn.Linear(
            in_features=self.dim_z * 4,
            out_features=self.dim_z * 2
        )
        
        self.log_var_linear2 = nn.Linear(
            in_features=self.dim_z * 2,
            out_features=self.dim_z
        )
        
        self.activation = nn.ReLU()
        
    def forward(self, x, x_lengths):
        x = self.norm(x)
        x = self.activation(self.linear(x))
        
        x = torch.nn.utils.rnn.pack_padded_sequence(x, x_lengths, batch_first=True, enforce_sorted=False)
        out, (_, _) = self.rnn(x)
        out, out_lengths = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
    
        out = torch.sum(out, dim=1) / out_lengths[:, None].type(torch.FloatTensor).cuda()
        
        mu = self.activation(self.mu_linear1(out))
        mu = self.mu_linear2(mu)
        
        log_var = self.activation(self.log_var_linear1(out))
        log_var = self.log_var_linear2(log_var)
        
        return [mu, log_var]
    
    
class lstmEncoder_cvae(nn.Module):
    def __init__(self, input_len, dim, dim_z, class_num, dropout=0.0):
        super(lstmEncoder_cvae, self).__init__()
        
        self.input_len = input_len
        self.dim = dim
        self.dim_z = dim_z
        self.class_num = class_num
        
        self.linear = nn.Linear(
            in_features=self.dim + self.class_num,
            out_features=self.dim + self.class_num
        )
        
        # self.norm = nn.BatchNorm1d(self.dim)
        self.norm = nn.LayerNorm([self.input_len, self.dim + self.class_num])
            
        self.rnn = nn.LSTM(
            input_size=self.dim + self.class_num,
            hidden_size=self.dim_z * 2,
            num_layers=1,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        self.mu_linear1 = nn.Linear(
            in_features=self.dim_z * 4,
            out_features=self.dim_z * 2
        )
        
        self.mu_linear2 = nn.Linear(
            in_features=self.dim_z * 2,
            out_features=self.dim_z
        )
        
        self.log_var_linear1 = nn.Linear(
            in_features=self.dim_z * 4,
            out_features=self.dim_z * 2
        )
        
        self.log_var_linear2 = nn.Linear(
            in_features=self.dim_z * 2,
            out_features=self.dim_z
        )
        
        self.activation = nn.ReLU()
        
    def forward(self, x, x_lengths, class_vector):
        
        """
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        x = x.permute(0, 2, 1)
        """
        x = torch.cat((x, class_vector[:, None, :].expand(-1, self.input_len, -1)), 2)
        x = self.activation(self.linear(x))
        x = self.norm(x)
        
        x = torch.nn.utils.rnn.pack_padded_sequence(x, x_lengths, batch_first=True, enforce_sorted=False)
        out, (_, _) = self.rnn(x)
        out, out_lengths = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        out = torch.sum(out, dim=1) / out_lengths[:, None].type(torch.FloatTensor).cuda()
        
        mu = self.activation(self.mu_linear1(out))
        mu = self.mu_linear2(mu)
        
        log_var = self.activation(self.log_var_linear1(out))
        log_var = self.log_var_linear2(log_var)
        
        return [mu, log_var]
    
    
class Encoder_c(nn.Module):
    def __init__(self, dim_z, class_num):
        super(Encoder_c, self).__init__()
        
        self.dim_z = dim_z
        self.class_num = class_num
        
        self.mu_linear1 = nn.Linear(
            in_features=self.class_num,
            out_features=int(self.dim_z / 2)
        )
        
        self.mu_linear2 = nn.Linear(
            in_features=int(self.dim_z / 2),
            out_features=self.dim_z
        )
    
        self.log_var_linear1 = nn.Linear(
            in_features=self.class_num,
            out_features=int(self.dim_z / 2)
        )
        
        self.log_var_linear2 = nn.Linear(
            in_features=int(self.dim_z / 2),
            out_features=self.dim_z
        )
        
        self.activation = nn.ReLU()
        
    def forward(self, class_vector):
        mu = self.activation(self.mu_linear1(class_vector))
        mu = self.mu_linear2(mu)
        
        log_var = self.activation(self.log_var_linear1(class_vector))
        log_var = self.log_var_linear2(log_var)
        
        return [mu, log_var]
        
    
class lstmDecoder(nn.Module):
    def __init__(self, input_len, dim, dim_z, dropout=0.0):
        super(lstmDecoder, self).__init__()
        
        self.input_len = input_len
        self.dim = dim
        self.dim_z = dim_z
        
        self.linear = nn.Linear(
            in_features=self.dim_z,
            out_features=self.dim_z
        )
        
        self.rnn = nn.LSTM(
            input_size=self.dim_z,
            hidden_size=self.dim_z * 2,
            num_layers=1,
            batch_first=True,
            dropout=dropout,
            bidirectional=False
        )
        
        self.linear1 = nn.Linear(
            in_features=self.dim_z * 2,
            out_features=self.dim_z)
       
        self.linear2 = nn.Linear(
            in_features=self.dim_z,
            out_features=self.dim)
        
        self.activation = nn.ReLU()
            
    def forward(self, x):
        x = self.activation(self.linear(x))
        
        x = x[:, None, :].expand(-1, self.input_len, -1)
        
        out, (_, _) = self.rnn(x)
        out = self.activation(self.linear1(out))
        recon = self.linear2(out)
        
        return recon
    
    
class lstmDecoder_c(nn.Module):
    def __init__(self, input_len, dim, dim_z, class_num, dropout=0.0):
        super(lstmDecoder_c, self).__init__()
        
        self.input_len = input_len
        self.dim = dim
        self.dim_z = dim_z
        self.class_num = class_num
        
        self.linear = nn.Linear(
            in_features=self.dim_z + self.class_num,
            out_features=self.dim_z + self.class_num)
        
        self.rnn = nn.LSTM(
            input_size=self.dim_z + self.class_num,
            hidden_size=self.dim_z * 2,
            num_layers=1,
            batch_first=True,
            dropout=dropout,
            bidirectional=False
        )
        
        self.linear1 = nn.Linear(
            in_features=self.dim_z * 2,
            out_features=self.dim_z)
       
        self.linear2 = nn.Linear(
            in_features=self.dim_z,
            out_features=self.dim)
        
        self.activation = nn.ReLU()
    
    def forward(self, x, class_vector):
        x = torch.cat((x, class_vector), 1)
        x = self.activation(self.linear(x))
        
        x = x[:, None, :].expand(-1, self.input_len, -1)
        
        out, (_, _) = self.rnn(x)
        out = self.activation(self.linear1(out))
        recon = self.linear2(out)
        
        return recon
    
    
class lstmDecoder_feedback(nn.Module):
    def __init__(self, input_len, dim, dim_z, dropout=0.0, residual=False):
        super(lstmDecoder_feedback, self).__init__()
        
        self.input_len = input_len
        self.dim = dim
        self.dim_z = dim_z
        self.residual = residual
        
        self.linear = nn.Linear(
            in_features=self.dim_z,
            out_features=self.dim_z
        )
        
        self.pose_linear1 = nn.Linear(
            in_features=self.dim_z,
            out_features=self.dim_z
        )
        
        self.pose_linear2 = nn.Linear(
            in_features=self.dim_z,
            out_features=self.dim
        )
        
        self.rnn_cell = nn.LSTMCell(
            input_size=self.dim_z + self.dim,
            hidden_size=self.dim_z * 2
        )
        
        self.linear1 = nn.Linear(
            in_features=self.dim_z * 2,
            out_features=self.dim_z)
       
        self.linear2 = nn.Linear(
            in_features=self.dim_z,
            out_features=self.dim)
        
        self.activation = nn.ReLU()
    
    def forward(self, x):
        outputs = []
        
        x = self.activation(self.linear(x))
        
        pose = self.activation(self.pose_linear1(x))
        pose = self.pose_linear2(pose)
        
        batch_size, _ = x.size()
        input = torch.cat((x, pose), 1)
        (h, c) = [torch.zeros(batch_size, self.dim_z * 2).cuda()] * 2
        for i in range(self.input_len):
            (h, c) = self.rnn_cell(input, (h, c))
            h_ = self.activation(self.linear1(h))
            if self.residual:
                pose = self.linear2(h_) + pose
            else:
                pose = self.linear2(h_)
            
            outputs += [pose]
            
            input = torch.cat((x, pose), 1)
        
        outputs = torch.stack(outputs, 1)
        return outputs
        
        
class lstmDecoder_initfeed(nn.Module):
    def __init__(self, input_len, dim, dim_z, dropout=0.0, residual=False):
        super(lstmDecoder_initfeed, self).__init__()
        
        self.input_len = input_len
        self.dim = dim
        self.dim_z = dim_z
        self.residual = residual
        
        self.linear = nn.Linear(
            in_features=self.dim_z,
            out_features=self.dim_z
        )

        self.rnn_cell = nn.LSTMCell(
            input_size=self.dim_z + self.dim,
            hidden_size=self.dim_z * 2
        )
        
        self.linear1 = nn.Linear(
            in_features=self.dim_z * 2,
            out_features=self.dim_z)
       
        self.linear2 = nn.Linear(
            in_features=self.dim_z,
            out_features=self.dim)
        
        self.activation = nn.ReLU()
    
    def forward(self, x, pose):
        outputs = []
        
        x = self.activation(self.linear(x))
        
        batch_size, _ = x.size()
        input = torch.cat((x, pose), 1)
        (h, c) = [torch.zeros(batch_size, self.dim_z * 2).cuda()] * 2
        for i in range(self.input_len):
            (h, c) = self.rnn_cell(input, (h, c))
            h_ = self.activation(self.linear1(h))
            if self.residual:
                pose = self.linear2(h_) + pose
            else:
                pose = self.linear2(h_)
            
            outputs += [pose]
            
            input = torch.cat((x, pose), 1)
        
        outputs = torch.stack(outputs, 1)
        return outputs
    
    
class Discriminator_frame(nn.Module):
    def __init__(self, dim, dim_z):
        super(Discriminator_frame, self).__init__()
        
        self.dim = dim
        self.dim_z = dim_z
        
        self.linear1 = nn.Linear(
            in_features=self.dim,
            out_features=self.dim_z
        )
        
        self.linear2 = nn.Linear(
            in_features=self.dim_z,
            out_features=1
        )
        
        self.activation = nn.ReLU()
        
    def forward(self, x):
        out = self.activation(self.linear1(x))
        out = self.linear2(out)
        return out
    

class Discriminator_seq(nn.Module):
    def __init__(self, input_len, dim, dim_z, dropout=0.0):
        super(Discriminator_seq, self).__init__()
        
        self.input_len = input_len
        self.dim = dim
        self.dim_z = dim_z
        
        self.norm = nn.LayerNorm([self.input_len, self.dim])
        
        self.linear = nn.Linear(
            in_features=self.dim,
            out_features=self.dim_z
        )
        
        self.rnn = nn.LSTM(
            input_size=self.dim_z,
            hidden_size=self.dim_z * 2,
            num_layers=1,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        self.linear1 = nn.Linear(
            in_features=self.dim_z * 4,
            out_features=self.dim_z
        )
        
        self.linear2 = nn.Linear(
            in_features=self.dim_z,
            out_features=1
        )
        
        self.activation = nn.ReLU()
        
    def forward(self, x, x_lengths):
        x = self.norm(x)
        x = self.activation(self.linear(x))
        
        x = torch.nn.utils.rnn.pack_padded_sequence(x, x_lengths, batch_first=True, enforce_sorted=False)
        out, (_, _) = self.rnn(x)
        out, out_lengths = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        
        last_seq_idxs = torch.LongTensor([x - 1 for x in out_lengths])
        out = torch.cat([out[:, 0, self.dim_z * 2:],  # forward output
                         out[range(out.shape[0]), last_seq_idxs, :self.dim_z * 2]],  # backward output
                        axis=1)
        
        # out = torch.sum(out, dim=1) / out_lengths[:, None].type(torch.FloatTensor).cuda()
        
        out = self.activation(self.linear1(out))
        out = self.linear2(out)
        
        return out
    
    
class Estimator_length(nn.Module):
    def __init__(self, dim_z):
        super(Estimator_length, self).__init__()
        
        self.dim_z = dim_z
        
        self.linear1 = nn.Linear(
            in_features=self.dim_z,
            out_features=int(self.dim_z / 4),
        )
        
        self.linear2 = nn.Linear(
            in_features=int(self.dim_z / 4),
            out_features=1
        )
        
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.linear1(x))
        return self.linear2(x)
    
class lstmAE_wo_Norm(nn.Module):
    def __init__(self, input_len, dim, dim_z):
        super(lstmAE_wo_Norm, self).__init__()
        
        self.encoder = lstmEncoder_wo_Norm(input_len, dim, dim_z)
        self.decoder = lstmDecoder(input_len, dim, dim_z)
        
    def forward(self, x, x_lengths):
        z = self.encoder(x, x_lengths)
        return [x, self.decoder(z), x_lengths]
        
        
class lstmAE(nn.Module):
    def __init__(self, input_len, dim, dim_z):
        super(lstmAE, self).__init__()
        
        self.encoder = lstmEncoder(input_len, dim, dim_z)
        self.decoder = lstmDecoder(input_len, dim, dim_z)
        
    def forward(self, x, x_lengths):
        z = self.encoder(x, x_lengths)
        return [x, self.decoder(z), x_lengths]
    
    
class lstmAE_feedback(nn.Module):
    def __init__(self, input_len, dim, dim_z, residual=False):
        super(lstmAE_feedback, self).__init__()
        
        self.encoder = lstmEncoder(input_len, dim, dim_z)
        self.decoder = lstmDecoder_feedback(input_len, dim, dim_z, residual=residual)
        
    def forward(self, x, x_lengths):
        z = self.encoder(x, x_lengths)
        return [x, self.decoder(z), x_lengths]
    
    
class lstmVAE(nn.Module):
    def __init__(self, input_len, dim, dim_z):
        super(lstmVAE, self).__init__()
        
        self.encoder = lstmEncoder_vae(input_len, dim, dim_z)
        self.decoder = lstmDecoder(input_len, dim, dim_z)
        
        self.estimator_length = Estimator_length(dim_z)
    
    def forward(self, x, x_lengths):
        mu, log_var = self.encoder(x, x_lengths)
        z = reparametrize(mu, log_var)
        return [x, self.decoder(z), x_lengths, mu, log_var, z, self.estimator_length(z)]
    
    
class lstmCVAE(nn.Module):
    def __init__(self, input_len, dim, dim_z, class_num):
        super(lstmCVAE, self).__init__()
        
        self.encoder = lstmEncoder_cvae(input_len, dim, dim_z, class_num)
        self.decoder = lstmDecoder_c(input_len, dim, dim_z, class_num)
        
    def forward(self, x, x_lengths, class_vector):
        mu, log_var = self.encoder(x, x_lengths, class_vector)
        z = reparametrize(mu, log_var)
        return [x, self.decoder(z, class_vector), x_lengths, mu, log_var, z]
    
        
class lstmCVAE2(nn.Module):
    def __init__(self, input_len, dim, dim_z, class_num):
        super(lstmCVAE2, self).__init__()
        
        self.encoder = lstmEncoder_cvae(input_len, dim, dim_z, class_num)
        self.encoder_class = Encoder_c(dim_z, class_num)
        self.decoder = lstmDecoder_c(input_len, dim, dim_z, class_num)
        
    def forward(self, x, x_lengths, class_vector):
        mu, log_var = self.encoder(x, x_lengths, class_vector)
        mu_c, log_var_c = self.encoder_class(class_vector)
        z = reparametrize(mu, log_var)
        return [x, self.decoder(z, class_vector), x_lengths, mu, log_var, z, mu_c, log_var_c]
    
    
class lstmVAE_feedback(nn.Module):
    def __init__(self, input_len, dim, dim_z, residual=False):
        super(lstmVAE_feedback, self).__init__()
        
        self.encoder = lstmEncoder_vae(input_len, dim, dim_z)
        self.decoder = lstmDecoder_feedback(input_len, dim, dim_z, residual=residual)
    
    def forward(self, x, x_lengths):
        mu, log_var = self.encoder(x, x_lengths)
        z = reparametrize(mu, log_var)
        return [x, self.decoder(z), x_lengths, mu, log_var, z]
    
    
class lstmVAE_initfeed(nn.Module):
    def __init__(self, input_len, dim, dim_z, residual=False):
        super(lstmVAE_initfeed, self).__init__()
        
        self.encoder = lstmEncoder_vae(input_len, dim, dim_z)
        self.decoder = lstmDecoder_initfeed(input_len, dim, dim_z, residual=residual)
        self.estimator_length = Estimator_length(dim_z)
    
    def forward(self, x, x_lengths):
        mu, log_var = self.encoder(x, x_lengths)
        z = reparametrize(mu, log_var)
        pose = x[:, 0, :]  # batch, time, pose
        return [x, self.decoder(z, pose), x_lengths, mu, log_var, z, self.estimator_length(z)]
        
    
def reparametrize(mu, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(mu)
    return eps * std + mu
