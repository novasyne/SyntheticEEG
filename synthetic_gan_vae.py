import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)

BANDS = ["Alpha", "Beta", "Delta", "Theta", "Gamma"]

DATASET_CONFIG = {
    "EDMA": {
        "real_file":      "Processed/EEG_Dataset_EDMA.csv",
        "label_col":      "Label",
        "frontal_ch":     ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz"],
        "parietal_ch":    ["P3", "P4", "Pz"],
        "participant_fn": lambda df: df["Subject"].str.extract(
            r"EEGStress(?:Baseline|Stressed)_(\d+)_\d+"
        )[0].fillna("unknown"),
    },
    "MS": {
        "real_file":      "Processed/EEG_Dataset_MS.csv",
        "label_col":      "Task",
        "frontal_ch":     ["F3", "F4"],
        "parietal_ch":    ["Pz"],
        "participant_fn": lambda df: df["Subject"].astype(str),
    },
    "MW": {
        "real_file":      "Processed/EEG_Dataset_MW.csv",
        "label_col":      "Label",
        "frontal_ch":     ["AF3", "AF4", "F3", "F4", "F7", "F8"],
        "parietal_ch":    ["P7", "P8"],
        "participant_fn": lambda df: df["Subject"].astype(str),
    },
    "SAM": {
        "real_file":      "Processed/EEG_Dataset_SAM.csv",
        "label_col":      "Label",
        "frontal_ch":     ["F3", "F4", "F7", "F8", "Fz",
                           "FC1", "FC2", "FC5", "FC6", "FT9", "FT10"],
        "parietal_ch":    ["P3", "P4", "Pz", "PO9", "PO10"],
        "participant_fn": lambda df: df["Subject"].astype(str),
    },
}

N_SYNTHETIC_SAMPLES = 70
N_FEATURES          = 10
N_CLASSES           = 2
LATENT_DIM          = 8
NOISE_DIM           = 16

GAN_EPOCHS   = 1000
GAN_LR       = 1e-4
GAN_N_CRITIC = 5
GAN_GP_LAMBDA = 10.0
VAE_EPOCHS   = 800
VAE_LR       = 1e-3
BATCH_SIZE   = 32


def _region_mean(df: pd.DataFrame, band: str, channels: list) -> np.ndarray:
    cols = [f"{band}{ch}" for ch in channels if f"{band}{ch}" in df.columns]
    if not cols:
        raise ValueError(f"No columns for band={band}, channels={channels}")
    return df[cols].mean(axis=1).values


def load_and_aggregate(name: str) -> pd.DataFrame:
    cfg       = DATASET_CONFIG[name]
    label_col = cfg["label_col"]
    df        = pd.read_csv(cfg["real_file"])
    df        = df[df[label_col].isin([0, 1])].reset_index(drop=True)

    participant = cfg["participant_fn"](df).values

    frontal_ch  = cfg["frontal_ch"]
    parietal_ch = cfg["parietal_ch"]

    agg = {}
    for band in BANDS:
        agg[f"{band} Frontal"]  = _region_mean(df, band, frontal_ch)
        agg[f"{band} Parietal"] = _region_mean(df, band, parietal_ch)

    feat_df   = pd.DataFrame(agg)
    feat_cols = list(feat_df.columns)

    feat_df = np.log1p(feat_df.clip(lower=0))

    feat_arr = feat_df.values.astype(float)
    for subj in np.unique(participant):
        mask = participant == subj
        mu   = feat_arr[mask].mean(axis=0)
        sd   = feat_arr[mask].std(axis=0)
        sd[sd == 0] = 1
        feat_arr[mask] = (feat_arr[mask] - mu) / sd

    result = pd.DataFrame(feat_arr, columns=feat_cols)
    result["Label"] = df[label_col].values.astype(int)
    return result


class CGANGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(NOISE_DIM + N_CLASSES, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, N_FEATURES),
        )

    def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, c], dim=1))


class CGANDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_FEATURES + N_CLASSES, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, c], dim=1))


def _one_hot(labels: torch.Tensor, n_classes: int = N_CLASSES) -> torch.Tensor:
    return torch.nn.functional.one_hot(labels, n_classes).float()


def _gradient_penalty(disc: nn.Module,
                       real: torch.Tensor, fake: torch.Tensor,
                       cond: torch.Tensor) -> torch.Tensor:
    batch_size = real.size(0)
    alpha      = torch.rand(batch_size, 1)
    interp     = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interp   = disc(interp, cond)
    grad       = torch.autograd.grad(
        outputs=d_interp,
        inputs=interp,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True, retain_graph=True, only_inputs=True,
    )[0]
    gp = ((grad.norm(2, dim=1) - 1) ** 2).mean()
    return gp


def train_cgan(X_norm: np.ndarray, y: np.ndarray,
               verbose: bool = True) -> CGANGenerator:
    X_t = torch.tensor(X_norm, dtype=torch.float32)
    y_t = torch.tensor(y,      dtype=torch.long)

    n   = len(X_t)
    gen = CGANGenerator()
    dis = CGANDiscriminator()
    opt_g = torch.optim.Adam(gen.parameters(), lr=GAN_LR, betas=(0.0, 0.9))
    opt_d = torch.optim.Adam(dis.parameters(), lr=GAN_LR, betas=(0.0, 0.9))

    bs = min(BATCH_SIZE, n)

    for epoch in range(GAN_EPOCHS):
        for _ in range(GAN_N_CRITIC):
            idx  = torch.randint(0, n, (bs,))
            xr   = X_t[idx]
            yr   = y_t[idx]
            cr   = _one_hot(yr)

            z    = torch.randn(bs, NOISE_DIM)
            yf   = torch.tensor(
                np.random.choice(N_CLASSES, bs,
                                 p=np.bincount(y, minlength=N_CLASSES) / len(y)),
                dtype=torch.long)
            cf   = _one_hot(yf)

            with torch.no_grad():
                xf = gen(z, cf)

            gp    = _gradient_penalty(dis, xr, xf.detach(), cr)
            d_loss = dis(xf.detach(), cf).mean() - dis(xr, cr).mean() + GAN_GP_LAMBDA * gp

            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()

        z  = torch.randn(bs, NOISE_DIM)
        yg = torch.tensor(
            np.random.choice(N_CLASSES, bs,
                             p=np.bincount(y, minlength=N_CLASSES) / len(y)),
            dtype=torch.long)
        cg = _one_hot(yg)
        g_loss = -dis(gen(z, cg), cg).mean()

        opt_g.zero_grad()
        g_loss.backward()
        opt_g.step()

        if verbose and (epoch + 1) % 200 == 0:
            print(f"    Epoch {epoch+1:4d}/{GAN_EPOCHS}  "
                  f"D={d_loss.item():+.3f}  G={g_loss.item():+.3f}")

    return gen


def sample_cgan(gen: CGANGenerator, class_dist: np.ndarray,
                n_samples: int) -> tuple:
    gen.eval()
    labels = np.random.choice(N_CLASSES, n_samples, p=class_dist)
    with torch.no_grad():
        z = torch.randn(n_samples, NOISE_DIM)
        c = _one_hot(torch.tensor(labels, dtype=torch.long))
        X = gen(z, c).numpy()
    return X, labels


class CVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(N_FEATURES + N_CLASSES, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
        )
        self.fc_mu     = nn.Linear(32, LATENT_DIM)
        self.fc_logvar = nn.Linear(32, LATENT_DIM)

        self.dec = nn.Sequential(
            nn.Linear(LATENT_DIM + N_CLASSES, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, N_FEATURES),
        )

    def encode(self, x: torch.Tensor, c: torch.Tensor):
        h = self.enc(torch.cat([x, c], dim=1))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.dec(torch.cat([z, c], dim=1))

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        mu, logvar = self.encode(x, c)
        z          = self.reparameterize(mu, logvar)
        recon      = self.decode(z, c)
        return recon, mu, logvar


def _vae_loss(recon: torch.Tensor, x: torch.Tensor,
              mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    mse = nn.functional.mse_loss(recon, x, reduction="mean")
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return mse + kld


def train_cvae(X_norm: np.ndarray, y: np.ndarray,
               verbose: bool = True) -> CVAE:
    X_t = torch.tensor(X_norm, dtype=torch.float32)
    y_t = torch.tensor(y,      dtype=torch.long)

    n    = len(X_t)
    vae  = CVAE()
    opt  = torch.optim.Adam(vae.parameters(), lr=VAE_LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, VAE_EPOCHS, eta_min=1e-5)

    bs = min(BATCH_SIZE, n)

    for epoch in range(VAE_EPOCHS):
        vae.train()
        idx    = torch.randint(0, n, (bs,))
        xb     = X_t[idx]
        cb     = _one_hot(y_t[idx])
        recon, mu, logvar = vae(xb, cb)
        loss   = _vae_loss(recon, xb, mu, logvar)

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
        opt.step()
        sched.step()

        if verbose and (epoch + 1) % 200 == 0:
            print(f"    Epoch {epoch+1:4d}/{VAE_EPOCHS}  Loss={loss.item():.4f}")

    return vae


def sample_cvae(vae: CVAE, class_dist: np.ndarray, n_samples: int) -> tuple:
    vae.eval()
    labels = np.random.choice(N_CLASSES, n_samples, p=class_dist)
    with torch.no_grad():
        z = torch.randn(n_samples, LATENT_DIM)
        c = _one_hot(torch.tensor(labels, dtype=torch.long))
        X = vae.decode(z, c).numpy()
    return X, labels


def generate_and_save(name: str):
    print(f"\n{'─'*60}")
    print(f"  Dataset: {name}")
    print(f"{'─'*60}")

    agg = load_and_aggregate(name)
    feat_cols = [c for c in agg.columns if c != "Label"]
    X_raw = agg[feat_cols].values.astype(float)
    y     = agg["Label"].values.astype(int)
    print(f"  Real data: {len(X_raw)} samples, "
          f"class dist: {np.bincount(y).tolist()}")

    scaler  = StandardScaler().fit(X_raw)
    X_norm  = scaler.transform(X_raw)
    class_p = np.bincount(y) / len(y)

    print(f"\n  [GAN] Training Conditional WGAN-GP …")
    gen     = train_cgan(X_norm, y, verbose=True)
    X_g, yg = sample_cgan(gen, class_p, N_SYNTHETIC_SAMPLES)
    X_g_inv = scaler.inverse_transform(X_g)

    gan_df = pd.DataFrame(X_g_inv, columns=feat_cols)
    gan_df["Label"] = yg.astype(int)
    out_gan = f"Synthetic_Aggregated_{name}_GAN.csv"
    gan_df.to_csv(out_gan, index=False)
    print(f"  [GAN] Saved → {out_gan}  "
          f"(label dist: {np.bincount(yg).tolist()})")

    print(f"\n  [VAE] Training Conditional VAE …")
    vae     = train_cvae(X_norm, y, verbose=True)
    X_v, yv = sample_cvae(vae, class_p, N_SYNTHETIC_SAMPLES)
    X_v_inv = scaler.inverse_transform(X_v)

    vae_df = pd.DataFrame(X_v_inv, columns=feat_cols)
    vae_df["Label"] = yv.astype(int)
    out_vae = f"Synthetic_Aggregated_{name}_VAE.csv"
    vae_df.to_csv(out_vae, index=False)
    print(f"  [VAE] Saved → {out_vae}  "
          f"(label dist: {np.bincount(yv).tolist()})")


def main():
    print("=" * 60)
    print("Synthetic EEG Generation — GAN and VAE")
    print(f"Datasets: {list(DATASET_CONFIG.keys())}")
    print(f"Samples per dataset per method: {N_SYNTHETIC_SAMPLES}")
    print("=" * 60)

    for name in DATASET_CONFIG:
        generate_and_save(name)

    print("\nAll datasets processed.")


if __name__ == "__main__":
    main()
