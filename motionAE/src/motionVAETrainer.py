from motionAE.src.motionAETrainer import motionAETrainer

import os
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from torch.distributions.normal import Normal
from torch.distributions.kl import _batch_trace_XXT, kl_divergence

from motionAE.src.models import lstmAE_wo_Norm, lstmVAE, lstmVAE_feedback
from motionAE.src.MMD import MMD_loss


class motionVAETrainer(motionAETrainer):

    def load_param(self, arg_parser, **kwargs):
        super().load_param(arg_parser, **kwargs)

        self.KL_coef = arg_parser.parse_float('KL_coef')
        self.KL_tolerance = arg_parser.parse_float('KL_tolerance')
        self.length_coef = arg_parser.parse_float('length_coef')
        self.sampling_method = arg_parser.parse_string('sampling_method')
        
        # for sampling
        self.mu = None
        self.std = None
        self.cluster_idx = None
        self.n_cluster = 3

        # for MMD calculation
        self.extractor_path = os.path.join(
            arg_parser.parse_string("output_path"), "ext", "model.pth")

    def build_model(self, gpu=True):
        if self.architecture == 'lstmVAE':
            self.model = lstmVAE(self.input_length, self.dim_pose, self.dim_z)
        elif self.architecture == 'lstmVAE_feedback':
            self.model = lstmVAE_feedback(
                self.input_length,
                self.dim_pose,
                self.dim_z,
                residual=self.residual)
        else:
            raise(ValueError)

        if gpu is True:
            self.model = self.model.cuda()

    def save_model(self):
        super().save_model()
        mu, log_var = self.encode_all_data(return_mu_log_var=True)
        std = np.exp(0.5 * log_var)
        self.mu = mu
        self.std = std

        path = os.path.join(self.path, "mu.npy")
        with open(path, 'wb') as f:
            np.save(f, mu)

        path = os.path.join(self.path, "std.npy")
        with open(path, 'wb') as f:
            np.save(f, std) 

    def load_model(self, gpu=True):
        super().load_model(gpu=gpu)

        self.mu = np.load(os.path.join(self.path, "mu.npy"))
        self.std = np.load(os.path.join(self.path, "std.npy"))

    def test(self):
        #super().test()
        #self.sample_from_latent()
        #self.plot_pdf()
        #self.plot_mean()
        #self.comp_sampling()
        #self.calc_MMD()
        #self.calc_MMD(test_set=True)
        pass

    def sample(self, batch_size=20, gpu=True):
        z_sample = self.sample_z(batch_size)

        if gpu:
            z_sample = self._to_torch(z_sample)
        else:
            z_sample = torch.from_numpy(z_sample.astype(np.float32))
        self.model.decoder.eval()
        with torch.no_grad():
            motions = self.model.decoder(z_sample)
            lengths = self.input_length - self.model.estimator_length(z_sample)
        self.model.decoder.train()
        if gpu:
            motions = self._to_numpy(motions)
            lengths = self._to_numpy(lengths)
        else:
            motions = motions.numpy()
            lengths = lengths.numpy()

        lengths = np.round(lengths).squeeze(axis=1).astype('int32')
        motions = [motion[:length] for motion, length in zip(motions, lengths)]

        return motions

    def sample_z(self, batch_size):
        if self.sampling_method == 'interpolate':
            return self.interpolate_sampling(batch_size)
        elif self.sampling_method == 'prior':
            return self.prior_sampling(batch_size)
        elif self.sampling_method == 'clustering-interpolate':
            return self.clustering_interpolate_sampling(batch_size)
        else:
            raise ValueError(f"unknown sampling method {self.sampling_method}")

    def interpolate_sampling(self, batch_size):
        idx1 = np.random.choice(len(self.mu), size=batch_size)
        idx2 = np.random.choice(len(self.mu), size=batch_size)
        mu = np.mean([self.mu[idx1], self.mu[idx2]], axis=0)
        std = np.mean([self.std[idx1], self.std[idx2]], axis=0)
        return np.random.normal(loc=mu, scale=std, size=[batch_size, self.dim_z])

    def prior_sampling(self, batch_size):
        mu = 0
        std = np.sqrt(self.KL_tolerance)
        return np.random.normal(loc=mu, scale=std, size=[batch_size, self.dim_z])

    def clustering_interpolate_sampling(self, batch_size, return_idx=False):

        if self.cluster_idx is None:
            from sklearn.decomposition import PCA
            from sklearn.cluster import KMeans
            pca = PCA(n_components=13, random_state=2)
            kmeans = KMeans(n_clusters=self.n_cluster)
            self.cluster_idx = kmeans.fit_predict(pca.fit_transform(self.mu))

        _, counts = np.unique(self.cluster_idx, return_counts=True)
        p = counts / len(self.cluster_idx)
        #cs = np.random.randint(self.n_cluster, size=batch_size)
        cs = np.random.choice(self.n_cluster, size=batch_size, p=p)
        idxs = [np.random.choice(np.where(self.cluster_idx == c)[0], size=2) for c in cs]
        mu = np.mean([self.mu[idx] for idx in idxs], axis=1)
        std = np.mean([self.std[idx] for idx in idxs], axis=1)

        z = np.random.normal(loc=mu, scale=std, size=[batch_size, self.dim_z])
        if return_idx:
            return z, cs
        else:
            return z
        
    def encode_all_data(self, return_mu_log_var=False):
        batch_size = 512
        num_data = len(self.motions)

        loop = int(np.ceil(num_data / batch_size))

        mu, log_var, z = [], [], []
        for i in range(loop):
            idx = np.array(range(len(self.motions)))[
                i * batch_size:(i + 1) * batch_size]
            inputs = self.sample_motions(idx=idx)
            results = self.model(*inputs)
            mu.append(self._to_numpy(results[3]))
            log_var.append(self._to_numpy(results[4]))
            z.append(self._to_numpy(results[5]))

        if return_mu_log_var:
            mu = np.concatenate(mu, axis=0)
            log_var = np.concatenate(log_var, axis=0)
            return mu, log_var
        else:
            z = np.concatenate(z, axis=0)
            return z

    def plot_mean(self):
        #self.z = self.encode_all_data(return_mu_log_var=True)[0]

        self.z_fit = np.random.normal(loc=0, scale=np.sqrt(self.KL_tolerance), size=[1000, self.dim_z])
        
        z_train = self.encode_all_data()
        
        batch_size = 1000
        z_sample = self.sample_z(batch_size)

        self.z = [z_train, z_sample]

        self.labels = [0, 1]
        self.classes = ['train', 'sample']
        
        path = os.path.join(self.path, self.sampling_method, "plot_mean")
        os.makedirs(path, exist_ok=True)
        self.plot_pca(path)

        #for perplexity in [5, 15, 50]:
        #    self.plot_tsne(path, perplexity)

        for n_neighbors in [2, 5]:
            for min_dist in [0.1, 0.2, 0.5, 0.9]:
                self.plot_umap(path, n_neighbors, min_dist)
                if self.dim_z > 50:
                    self.plot_pca_umap(path, n_neighbors, min_dist)

    def plot_pca(self, path):
        print(f"plotting pca figure...")
        path = os.path.join(path, f"pca.png")

        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=2)
        pca.fit(self.z_fit)
        points = [pca.transform(z) for z in self.z]
        
        self.viz_points(points, self.labels, self.classes, path)

    def plot_tsne(self, path, perplexity):
        print(f"plotting tsne figure with perplexity {perplexity}...")
        path = os.path.join(path, f"tsne_perplexity{perplexity}.png")

        from sklearn.manifold import TSNE
        points = TSNE(n_components=2, random_state=0, n_jobs=8, perplexity=perplexity).fit_transform(
            np.concatenate(self.z, axis=0))
        points = np.split(points, [len(self.z[0])])
        
        self.viz_points(points, self.labels, self.classes, path)

    def plot_umap(self, path, n_neighbors, min_dist):
        print(f"plotting umap figure with n_neighbors {n_neighbors}, min_dist {min_dist}...")
        path = os.path.join(path, f"umap_neighbor{n_neighbors}_minD{min_dist}.png")

        from umap import UMAP
        umap = UMAP(n_components=2, random_state=0, n_neighbors=n_neighbors, min_dist=min_dist, n_jobs=8)
        umap.fit(self.z_fit)
        points = [umap.transform(z) for z in self.z]

        self.viz_points(points, self.labels, self.classes, path)

    def plot_pca_umap(self, path, n_neighbors, min_dist):
        print(f"plotting pca+umap figure with n_neighbors {n_neighbors}, min_dist {min_dist}...")
        path = os.path.join(path, f"pca+umap_neighbor{n_neighbors}_minD{min_dist}.png")

        n_dim = 13
        from sklearn.decomposition import PCA
        from umap import UMAP
        pca = PCA(n_components=n_dim, random_state=2)
        umap = UMAP(n_components=2, random_state=0, n_neighbors=n_neighbors, min_dist=min_dist, n_jobs=8)
        umap.fit(pca.fit_transform(self.z_fit))
        points = [umap.transform(pca.transform(z)) for z in self.z]

        self.viz_points(points, self.labels, self.classes, path)
    
    def viz_points(self, points, labels, classes, path, colors=None, markers=None):
        plt.close()
        plt.figure(figsize=(20, 20))

        if colors is None:
            colors = list(matplotlib.colors.XKCD_COLORS.values())
        if markers is None:
            markers = ['.'] * len(points)
        for p, l, cla in zip(points, labels, classes):
            plt.scatter(p[:, 0], p[:, 1], marker=markers[l], c=colors[l], label=cla, s=150)
        plt.legend(fontsize=30)
        plt.savefig(path)

    def comp_sampling(self):
        print("plotting comparison of sampling method...")
        path = os.path.join(self.path, "comp_sampling")
        os.makedirs(path, exist_ok=True)
        batch_size = 100

        self.z_fit = np.random.normal(loc=0, scale=np.sqrt(self.KL_tolerance), size=[1000,self.dim_z])
        #z_train = self.encode_all_data(return_mu_log_var=True)[0]
        z_train = self.encode_all_data()
        z_prior = self.prior_sampling(batch_size)
        z_sample, cs = self.clustering_interpolate_sampling(batch_size, return_idx=True)
        
        self.z = [z_prior]
        [(self.z.append(z_train[np.where(self.cluster_idx == c)]),
          self.z.append(z_sample[np.where(cs == c)]))
         for c in range(self.n_cluster)]
        
        self.labels = range(1+2*self.n_cluster)
        self.classes = ['prior'] 
        [(self.classes.append(f"cluster#{i+1}"), self.classes.append(f"sample#{i+1}"))
         for i in range(self.n_cluster)]
        import itertools
        n_neighbors = [2, 5]
        min_dists = [0.1, 0.2, 0.5, 0.9]
        for n_neighbor, min_dist in itertools.product(n_neighbors, min_dists):
            self.plot_comp_sampling(path, n_neighbor, min_dist)

    def plot_comp_sampling(self, path, n_neighbors, min_dist):
        print(f"plotting pca+umap figure with n_neighbors {n_neighbors}, min_dist {min_dist}...")
        path = os.path.join(path, f"pca+umap_neighbor{n_neighbors}_minD{min_dist}.png")

        n_dim = 13
        from sklearn.decomposition import PCA
        from umap import UMAP
        pca = PCA(n_components=n_dim, random_state=2)
        umap = UMAP(n_components=2, random_state=0, n_neighbors=n_neighbors, min_dist=min_dist, n_jobs=8)
        umap.fit(pca.fit_transform(self.z_fit))
        
        points = [umap.transform(pca.transform(z)) for z in self.z]

        colors = []
        markers = []
        m_list = ['^', 's', '*']
        for l, cla in zip(self.labels, self.classes) :
            if cla == 'prior':
                m = '.'
                c = [0.5, 0.5, 0.5]
            else:
                m = m_list[int(cla[-1])-1]
                if 'cluster' in cla:
                    c = [0.0, 0.0, 0.0, 1.0]
                elif 'sample' in cla:
                    c = [0.0, 0.0, 0.0, 0.0]
                    strength = 1.0
                    c[int(cla[-1])-1] = 1.0
                    c[3] = strength
                

            colors.append(c)
            markers.append(m)

        self.viz_points(points, self.labels, self.classes, path, colors=colors, markers=markers)
        
    def gaussian(self, x, mu, std):
        coef = 1 / np.sqrt(2 * np.pi) / std
        return np.exp(- (x - mu)**2 / (2 * std**2)) * coef

    def plot_pdf(self):
        dirpath = os.path.join(self.path, 'plot_mean', self.sampling_method)
        os.makedirs(dirpath, exist_ok=True)
        path_pdf = os.path.join(dirpath, "pdf.png")
        print(f"plotting the learned probability distribution to {path_pdf}")
        z_mean, z_log_var = self.encode_all_data(return_mu_log_var=True)

        # visualize histgram for the first dimension
        d = np.argmax(np.var(z_mean, axis=0))
        z_mean = z_mean[:, d]
        z_std = np.exp(0.5 * z_log_var[:, d])
        bins = 100000
        dist_range = (-5, 5)
        xs = np.linspace(dist_range[0], dist_range[1], bins)
        
        proba = [np.mean(self.gaussian(x, z_mean, z_std)) for x in xs]
        # proba = [self.gaussian(x, np.mean(z_mean), np.mean(z_log_var)) for x in xs]

        proba_normal = [
            self.gaussian(
                x, 0, np.sqrt(
                    self.KL_tolerance)) for x in xs]

        size = 3
        idxs = np.random.choice(range(len(z_mean)), size=size)
        probas = np.array([self.gaussian(x, z_mean[idxs], z_std[idxs]) for x in xs])

        plt.figure(figsize=(10, 10))
        plt.plot(xs,
                 proba, label="accumulated VAE pdf")
        plt.plot(xs,
                 proba_normal, label="Normal pdf")
        for i in range(size):
            plt.plot(xs,
                     probas[:, i], label=f"sample VAE pdf#{i}")
        plt.legend()
        plt.savefig(path_pdf)

    def sample_from_latent(self):
        path_sample = os.path.join(self.path, self.sampling_method, 'sample')

        for path in [path_sample]:
            try:
                shutil.rmtree(path)
            except FileNotFoundError:
                pass
            os.makedirs(path)

        batch_size = 20

        sampled_motions = self.sample(batch_size)
        names = [str(i).zfill(4) for i in range(batch_size)]

        self.write_bvhs(sampled_motions, names, 1 / self.fps, path_sample)

    def loss(self, *args):
        #print(super().loss(*args).item(), self.kld_loss(*args).item(), self.length_loss(*args).item())
        return super().loss(*args) + self.KL_coef * self.kld_loss(*args) + \
            self.length_coef * self.length_loss(*args)

    def kld_loss(self, *args):
        mu = args[3]
        log_var = args[4]

        q = Normal(mu, torch.exp(0.5 * log_var))
        pi = Normal(0, np.sqrt(self.KL_tolerance))
        return torch.mean(torch.sum(kl_divergence(q, pi), dim=1))

        # return torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 -
        # log_var.exp(), dim=1), dim=0).cuda()

    def length_loss(self, *args):
        input_length = self._to_torch(args[2])
        pred_length = self.input_length - args[6]

        return ((input_length - pred_length) ** 2).mean()

    def calc_MMD(self, ntrial=10, test_set=False):
        
        if test_set:
            print('Test Set')
        else:
            print('Train Set')

        self.model.eval()
        
        extractor = lstmAE_wo_Norm(self.max_len, self.dim_pose, self.dim_z).cuda()
        extractor.load_state_dict(torch.load(self.extractor_path))
        extractor.eval()
        
        inputs = self.sample_motions(test=test_set)
        motions, lengths = inputs
        batch_size, _, _ = motions.shape
        motions_pad = torch.zeros(
            [batch_size, self.max_len, self.dim_pose]).cuda()
        motions_pad[:, :self.input_length, :] = motions
        with torch.no_grad():
            feature_real = extractor.encoder(motions_pad, lengths)

        mmds = []
        for _ in range(ntrial):
            
            z_sample = self.sample_z(batch_size)
            z_sample = self._to_torch(z_sample)
            with torch.no_grad():
                motions = self.model.decoder(z_sample)
                motions_pad = torch.zeros(
                    [batch_size, self.max_len, self.dim_pose]).cuda()
                motions_pad[:, :self.input_length, :] = motions
                lengths = self._to_numpy(
                    self.input_length - self.model.estimator_length(z_sample)).squeeze().astype('int32')
                feature_fake = extractor.encoder(motions_pad, lengths)
                mmd = MMD_loss()(feature_real, feature_fake)
            """
            inputs = self.sample_motions(batch_size=len(self.test_motions))
            motions, lengths = inputs
            batch_size, _, _ = motions.shape
            motions_pad = torch.zeros(
                [batch_size, self.max_len, self.dim_pose]).cuda()
            motions_pad[:, :self.input_length, :] = motions
            with torch.no_grad():
                feature_fake = extractor.encoder(motions_pad, lengths)
            mmd = MMD_loss()(feature_real, feature_fake)
            """
            mmds = np.append(mmds, self._to_numpy(mmd))
        self.model.train()

        print(f"mmd: {mmds.mean()} "
              f"min: {mmds.min()} "
              f"max: {mmds.max()} ")
        
        suffix = "_test" if test_set else "_train"
        np.save(os.path.join(self.path, "mmd" + suffix + ".npy"), mmds)
