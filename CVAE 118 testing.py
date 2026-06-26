import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import pandas as pd
import pandapower as pp
import os
from tqdm import tqdm
from time import perf_counter
from pandapower.converter.matpower.from_mpc import from_mpc


#DIRECTORIES:
# Directory with generated data:
GEN_DATA_FILE_DIR = r"C:\gen_data\118bus_data.csv"
# Directory where the model weights + scalers are saved:
BEST_MODEL_DIR = r"C:\best_model"
# Directory with PGLib models:
PGLIB_MODELS_DIR = r"C:\PGLib_models"
# System model:
cvae_model = from_mpc(os.path.join(PGLIB_MODELS_DIR, 'pglib_opf_case118_ieee.m'), f_hz=60)
net = cvae_model

TARGET_DIM = 344
CONDITION_DIM = 198
LATENT_DIM = 54
HIDDEN_1 = 1024
HIDDEN_2 = 512
HIDDEN_3 = 256
HIDDEN_4 = 128

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

NUM_OBC = 99
NUM_BUS = 118

class Encoder(nn.Module):
    def __init__(self, target_dim, condition_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(target_dim + condition_dim, HIDDEN_1)
        self.fc2 = nn.Linear(HIDDEN_1, HIDDEN_2)
        self.fc3 = nn.Linear(HIDDEN_2, HIDDEN_3)
        self.fc4 = nn.Linear(HIDDEN_3, HIDDEN_4)
        self.fc_mu = nn.Linear(HIDDEN_4, latent_dim)
        self.fc_logvar = nn.Linear(HIDDEN_4, latent_dim)

    def forward(self, x, c):
        inputs = torch.cat([x, c], dim=1)
        h = F.relu(self.fc1(inputs))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        h = F.relu(self.fc4(h))
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim, condition_dim, target_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim + condition_dim, HIDDEN_4)
        self.fc2 = nn.Linear(HIDDEN_4, HIDDEN_3)
        self.fc3 = nn.Linear(HIDDEN_3, HIDDEN_2)
        self.fc4 = nn.Linear(HIDDEN_2, HIDDEN_1)
        self.fc5 = nn.Linear(HIDDEN_1, target_dim)

    def forward(self, z, c):
        inputs = torch.cat([z, c], dim=1)
        h = F.relu(self.fc1(inputs))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        h = F.relu(self.fc4(h))
        return torch.sigmoid(self.fc5(h))


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

    def inference(self, c):
        z = torch.randn(c.size(0), self.decoder.fc1.in_features - c.size(1)).to(c.device)
        return self.decoder(z, c)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Used device: {device}")

    model = ACOPF_CVAE(target_dim=TARGET_DIM, condition_dim=CONDITION_DIM, latent_dim=LATENT_DIM).to(device)
    model.load_state_dict(torch.load(os.path.join(BEST_MODEL_DIR, 'cvae_acopf_weights.pth'), map_location=device))

    model.eval()

    scaler_c = joblib.load(os.path.join(BEST_MODEL_DIR, 'scaler_c.pkl'))
    scaler_x = joblib.load(os.path.join(BEST_MODEL_DIR, 'scaler_x.pkl'))

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
    idx_val_end = round((1.0-TEST_RATIO) * total_samples)

    x_test_raw = x_raw[idx_val_end:]
    c_test_raw = c_raw[idx_val_end:]

    num_test_samples = len(c_test_raw)
    print(f"Number of test samples: {num_test_samples}\n")

    net = cvae_model
    convergent_generations = 0
    start = perf_counter()

    for case in tqdm(range(num_test_samples), desc="Testing IEEE118-bus with AC-OPF FLAT-START"):

        net.load.p_mw = c_test_raw[case][0:(NUM_OBC * 2):2]
        net.load.q_mvar = c_test_raw[case][1:(NUM_OBC * 2):2]

        try:
            pp.runopp(net, init='flat', numba=True)

            if net.OPF_converged:
                convergent_generations = convergent_generations + 1

        except pp.optimal_powerflow.OPFNotConverged:
            continue

    end = perf_counter()
    total_time = end - start
    avg_speed = total_time / num_test_samples

    print(f"FLAT-start method achieved convergence in {convergent_generations}/{num_test_samples} cases")
    print(f"Average iteration speed: {avg_speed:.6f} s/it")


    convergent_generations = 0
    nn_time_sum = 0
    start = perf_counter()

    for case in tqdm(range(num_test_samples), desc="Testing IEEE118-bus with AC-OPF WARM-START"):

        nn_start = perf_counter()
        c_test_scaled = scaler_c.transform(c_test_raw[case].reshape(1,-1))
        c_test_tensor = torch.tensor(c_test_scaled, dtype=torch.float32).to(device)

        with torch.no_grad():
            x_pred_scaled_tensor = model.inference(c_test_tensor)

        x_pred_scaled = x_pred_scaled_tensor.cpu().numpy()
        x_pred_raw = scaler_x.inverse_transform(x_pred_scaled)

        vm_pu = x_pred_raw[0][0:(NUM_BUS*2):2]
        va_degree = x_pred_raw[0][1:(NUM_BUS*2):2]

        p_ext_grid = x_pred_raw[0][236:237]
        q_ext_grid = x_pred_raw[0][237:238]

        p_gen = x_pred_raw[0][238:344:2]
        q_gen = x_pred_raw[0][239:344:2]

        net.res_bus.vm_pu = vm_pu
        net.res_bus.va_degree = va_degree

        net.res_ext_grid.p_mw = p_ext_grid
        net.res_ext_grid.q_mvar = q_ext_grid

        net.res_gen.p_mw = p_gen
        net.res_gen.q_mvar = q_gen

        nn_end = perf_counter()
        nn_time = nn_end - nn_start
        print(f"Measuring the transit time through the CVAE network: {nn_time:.6f} s")
        nn_time_sum = nn_time_sum + nn_time
        (print(f"Total time: {nn_time_sum:.6f} s"))

        net.load.p_mw = c_test_raw[case][0:(NUM_OBC*2):2]
        net.load.q_mvar = c_test_raw[case][1:(NUM_OBC*2):2]

        try:
            pp.runopp(net, init='results', numba=True)

            if net.OPF_converged:
                convergent_generations = convergent_generations + 1

        except pp.optimal_powerflow.OPFNotConverged:
            continue

    end = perf_counter()
    total_time = end - start
    avg_speed = total_time/num_test_samples
    avg_CVAE_speed = nn_time_sum / num_test_samples

    print(f"The WARM-start method achieved convergence in {convergent_generations}/{len(c_test_raw)} cases")
    print(f"Average iteration speed: {avg_speed:.6f} s/it")
    print(f"Average CVAE network transition speed: {avg_CVAE_speed:.6f} s")


    net = cvae_model
    convergent_generations = 0
    start = perf_counter()

    for case in tqdm(range(num_test_samples), desc="Testing IEEE118-bus with DC-OPF"):

        net.load.p_mw = c_test_raw[case][0:(NUM_OBC * 2):2]
        net.load.q_mvar = c_test_raw[case][1:(NUM_OBC * 2):2]

        try:
            pp.rundcopp(net, numba=True)
            if net.OPF_converged:
                convergent_generations = convergent_generations + 1

        except pp.optimal_powerflow.OPFNotConverged:
            continue

    end = perf_counter()
    total_time = end - start
    avg_speed = total_time / num_test_samples

    print(f"The DC-OPF method achieved convergence in {convergent_generations}/{num_test_samples} cases")
    print(f"Average iteration speed: {avg_speed:.6f} s/it")
