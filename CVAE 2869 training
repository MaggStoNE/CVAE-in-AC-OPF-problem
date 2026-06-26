import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from torch.utils.tensorboard import SummaryWriter
import joblib

# DIRECTORIES:
# Directory with generated data:
GEN_DATA_FILE_DIR = r"C:\gen_data\2869bus_data.csv"
# Directory where you want to save best trained model weights + scalers
BEST_MODEL_DIR = r"C:\best_model"

# --- Model architecture ---
TARGET_DIM = 6758
CONDITION_DIM = 2622
LATENT_DIM = 128
HIDDEN_1 = 16384
HIDDEN_2 = 8192
HIDDEN_3 = 4096
HIDDEN_4 = 2048
HIDDEN_5 = 1024
HIDDEN_6 = 512
HIDDEN_7 = 256

# --- Learning hiperparameters ---
BATCH_SIZE = 64
LEARNING_RATE = 0.00001
DROPOUT_RATE = 0.2
NUM_EPOCHS = 1000
BETA = 1.0

# --- Data division into traning, validation and testing groups ---
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10


class Encoder(nn.Module):
    def __init__(self, target_dim, condition_dim, latent_dim):
        super(Encoder, self).__init__()
        self.dropout = nn.Dropout(p=DROPOUT_RATE)

        self.fc1 = nn.Linear(target_dim + condition_dim, HIDDEN_1)
        self.fc2 = nn.Linear(HIDDEN_1, HIDDEN_2)
        self.fc3 = nn.Linear(HIDDEN_2, HIDDEN_3)
        self.fc4 = nn.Linear(HIDDEN_3, HIDDEN_4)
        self.fc5 = nn.Linear(HIDDEN_4, HIDDEN_5)
        self.fc6 = nn.Linear(HIDDEN_5, HIDDEN_6)
        self.fc7 = nn.Linear(HIDDEN_6, HIDDEN_7)

        self.fc_mu = nn.Linear(HIDDEN_7, latent_dim)
        self.fc_logvar = nn.Linear(HIDDEN_7, latent_dim)

    def forward(self, x, c):
        inputs = torch.cat([x, c], dim=1)

        h = self.dropout(F.relu(self.fc1(inputs)))
        h = self.dropout(F.relu(self.fc2(h)))
        h = self.dropout(F.relu(self.fc3(h)))
        h = self.dropout(F.relu(self.fc4(h)))
        h = self.dropout(F.relu(self.fc5(h)))
        h = self.dropout(F.relu(self.fc6(h)))
        h = F.relu(self.fc7(h))

        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim, condition_dim, target_dim):
        super(Decoder, self).__init__()
        self.dropout = nn.Dropout(p=DROPOUT_RATE)

        self.fc1 = nn.Linear(latent_dim + condition_dim, HIDDEN_7)
        self.fc2 = nn.Linear(HIDDEN_7, HIDDEN_6)
        self.fc3 = nn.Linear(HIDDEN_6, HIDDEN_5)
        self.fc4 = nn.Linear(HIDDEN_5, HIDDEN_4)
        self.fc5 = nn.Linear(HIDDEN_4, HIDDEN_3)
        self.fc6 = nn.Linear(HIDDEN_3, HIDDEN_2)
        self.fc7 = nn.Linear(HIDDEN_2, HIDDEN_1)

        self.fc_out = nn.Linear(HIDDEN_1, target_dim)

    def forward(self, z, c):
        inputs = torch.cat([z, c], dim=1)

        h = self.dropout(F.relu(self.fc1(inputs)))
        h = self.dropout(F.relu(self.fc2(h)))
        h = self.dropout(F.relu(self.fc3(h)))
        h = self.dropout(F.relu(self.fc4(h)))
        h = self.dropout(F.relu(self.fc5(h)))
        h = self.dropout(F.relu(self.fc6(h)))

        h = F.relu(self.fc7(h))

        return torch.sigmoid(self.fc_out(h))


class ACOPF_CVAE(nn.Module):
    def __init__(self, target_dim=TARGET_DIM, condition_dim=CONDITION_DIM, latent_dim=LATENT_DIM):
        super(ACOPF_CVAE, self).__init__()
        self.encoder = Encoder(target_dim, condition_dim, latent_dim)
        self.decoder = Decoder(latent_dim, condition_dim, target_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, c):
        mu, logvar = self.encoder(x, c)
        z = self.reparameterize(mu, logvar)
        reconstructed_x = self.decoder(z, c)
        return reconstructed_x, mu, logvar

def cvae_loss_function(recon_x, x, mu, logvar, beta=1.0):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_divergence

if __name__ == "__main__":
    writer = SummaryWriter()

    df_all = pd.read_csv(GEN_DATA_FILE_DIR, sep=';', decimal=',')
    df_all = df_all.drop(columns=['sample_id', 'total_cost'])

    condition_cols = [col for col in df_all.columns if col.startswith('load')]

    target_cols = [col for col in df_all.columns if col.startswith('bus') or col.startswith('gen')]

    print(f"Found {len(condition_cols)} input columns (loads).")
    print(f"Found {len(target_cols)} output columns (voltages, angles, generation).")

    df_cond = df_all[condition_cols]
    df_target = df_all[target_cols]

    c_raw = df_cond.values
    x_raw = df_target.values

    total_samples = len(x_raw)

    idx_train_end = round(TRAIN_RATIO * total_samples)
    idx_val_end = round((TRAIN_RATIO + VAL_RATIO) * total_samples)

    x_train_raw = x_raw[:idx_train_end]
    c_train_raw = c_raw[:idx_train_end]

    x_val_raw = x_raw[idx_train_end:idx_val_end]
    c_val_raw = c_raw[idx_train_end:idx_val_end]

    x_test_raw = x_raw[idx_val_end:]
    c_test_raw = c_raw[idx_val_end:]

    print(f"Number of training samples: {len(x_train_raw)}")
    print(f"Number of validation samples: {len(x_val_raw)}")

    scaler_x = MinMaxScaler(feature_range=(0, 1))
    scaler_c = MinMaxScaler(feature_range=(0, 1))

    x_train_scaled = scaler_x.fit_transform(x_train_raw)
    c_train_scaled = scaler_c.fit_transform(c_train_raw)

    x_val_scaled = scaler_x.transform(x_val_raw)
    c_val_scaled = scaler_c.transform(c_val_raw)

    train_dataset = TensorDataset(torch.tensor(x_train_scaled, dtype=torch.float32),
                                  torch.tensor(c_train_scaled, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(x_val_scaled, dtype=torch.float32),
                                torch.tensor(c_val_scaled, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Used device: {device}")

    model = ACOPF_CVAE(target_dim=TARGET_DIM, condition_dim=CONDITION_DIM, latent_dim=LATENT_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    num_epochs = NUM_EPOCHS
    best_val_loss = float('inf')

    print("Starting training process...")
    for epoch in range(num_epochs):

        model.train()
        train_loss = 0.0
        total_train_samples = 0

        for batch_x, batch_c in train_loader:
            batch_x = batch_x.to(device)
            batch_c = batch_c.to(device)

            optimizer.zero_grad()
            recon_batch, mu, logvar = model(batch_x, batch_c)
            loss = cvae_loss_function(recon_batch, batch_x, mu, logvar, beta=BETA)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            total_train_samples += batch_x.size(0)

        avg_train_loss = train_loss / total_train_samples
        writer.add_scalar("loss/train", avg_train_loss, epoch + 1)

        model.eval()
        val_loss = 0.0
        total_val_samples = 0

        with torch.no_grad():
            for val_x, val_c in val_loader:
                val_x, val_c = val_x.to(device), val_c.to(device)

                recon_val, mu_val, logvar_val = model(val_x, val_c)
                v_loss = cvae_loss_function(recon_val, val_x, mu_val, logvar_val, beta=BETA)

                val_loss += v_loss.item()
                total_val_samples += val_x.size(0)

        avg_val_loss = val_loss / total_val_samples
        writer.add_scalar("loss/validation", avg_val_loss, epoch + 1)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(BEST_MODEL_DIR, 'cvae_acopf_weights.pth'))
            best_epoch = epoch + 1

        print(f"Epoc [{epoch + 1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    print(f"Saved best model of epoc: {best_epoch} | Valid_loss = {best_val_loss}")
    writer.flush()
    writer.close()

    joblib.dump(scaler_x, os.path.join(BEST_MODEL_DIR, 'scaler_x.pkl'))
    joblib.dump(scaler_c, os.path.join(BEST_MODEL_DIR, 'scaler_c.pkl'))

    # Type "tensorboard --logdir=runs" in terminal for the link to the server with Tensorboard saved data
