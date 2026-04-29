import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import RGCNConv





# =========================================================
# Page setup
# =========================================================
st.set_page_config(
    page_title="Drug Side Effects Prediction App",
    page_icon="💊",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f4f8ff 0%, #eef7f9 45%, #fdfcff 100%);
    color: #1f2937;
}

.main .block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
    padding-left: 3rem;
    padding-right: 3rem;
    background: rgba(255, 255, 255, 0.94);
    border-radius: 24px;
    box-shadow: 0 8px 30px rgba(31, 41, 55, 0.10);
    margin-top: 2rem;
}

.hero-banner {
    background: linear-gradient(135deg, #0f2a3a 0%, #123f59 100%);
    border-radius: 22px;
    padding: 2rem;
    margin-bottom: 1.8rem;
    color: white;
    box-shadow: 0 10px 28px rgba(15, 42, 58, 0.25);
}

.hero-banner h1 {
    color: white;
    margin-bottom: 0.4rem;
}

.hero-banner p {
    color: #d7ecf7;
    font-size: 17px;
    line-height: 1.5;
}

h2, h3 {
    color: #1d3557;
    font-weight: 700;
}

.stButton > button {
    background: linear-gradient(135deg, #2f80ed 0%, #56ccf2 100%);
    color: white;
    border: none;
    border-radius: 14px;
    padding: 0.7rem 1.4rem;
    font-weight: 700;
    font-size: 16px;
    box-shadow: 0 5px 16px rgba(47, 128, 237, 0.28);
}

[data-testid="stDataFrame"] {
    border-radius: 16px;
    overflow: hidden;
    border: 1px solid #e5edf5;
}

footer, #MainMenu {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

st.image(BANNER_PATH, use_container_width=True)

st.markdown("""
<h1>💊 Drug Side Effects Prediction</h1>
<p class="app-subtitle">
Select a drug and clinical indication to predict likely side effects using the trained RGCN model.
</p>
""", unsafe_allow_html=True)

# =========================================================
# File paths
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "rgcn_model_v2_22apr.pth")
ARTIFACT_PATH = os.path.join(BASE_DIR, "rgcn_streamlit_artifacts.pkl")
DRUG_IND_PAIR_PATH = os.path.join(BASE_DIR, "rgcn_drug_indication_pairs.csv")
BANNER_PATH = os.path.join(BASE_DIR, "banner.png")

# =========================================================
# Load artifacts
# =========================================================
@st.cache_resource
def load_artifacts():
    with open(ARTIFACT_PATH, "rb") as f:
        artifacts = pickle.load(f)
    return artifacts

@st.cache_data
def load_drug_ind_pairs():
    pairs = pd.read_csv(DRUG_IND_PAIR_PATH)

    pairs["drug_name"] = pairs["drug_name"].astype(str).str.lower().str.strip()
    pairs["indication_name"] = pairs["indication_name"].astype(str).str.lower().str.strip()

    pairs = pairs.drop_duplicates()

    return pairs


drug_ind_pairs = load_drug_ind_pairs()


artifacts = load_artifacts()

drug2id = artifacts["drug2id"]
ind2id = artifacts["ind2id"]
se2id = artifacts["se2id"]

id2drug = artifacts["id2drug"]
id2ind = artifacts["id2ind"]
id2se = artifacts["id2se"]

num_nodes = artifacts["num_nodes"]
num_relations = artifacts["num_relations"]

drug_offset = artifacts["drug_offset"]
ind_offset = artifacts["ind_offset"]
se_offset = artifacts["se_offset"]

edge_index = artifacts["edge_index"]
edge_type = artifacts["edge_type"]

best_threshold = artifacts.get("best_threshold", 0.5)

config = artifacts["model_config"]
EMBED_DIM = config["EMBED_DIM"]
HIDDEN_DIM = config["HIDDEN_DIM"]
NUM_LAYERS = config["NUM_LAYERS"]
DROPOUT = config["DROPOUT"]
MLP_HIDDEN = config["MLP_HIDDEN"]


# =========================================================
# Model classes
# Must match the training notebook exactly
# =========================================================
class RGCNEncoder(nn.Module):
    def __init__(
        self,
        num_nodes,
        num_relations,
        emb_dim=64,
        hidden_dim=128,
        num_layers=2,
        dropout=0.3
    ):
        super().__init__()

        self.node_emb = nn.Embedding(num_nodes, emb_dim)

        self.convs = nn.ModuleList()

        if num_layers == 1:
            self.convs.append(RGCNConv(emb_dim, hidden_dim, num_relations))
        else:
            self.convs.append(RGCNConv(emb_dim, hidden_dim, num_relations))

            for _ in range(num_layers - 2):
                self.convs.append(RGCNConv(hidden_dim, hidden_dim, num_relations))

            self.convs.append(RGCNConv(hidden_dim, hidden_dim, num_relations))

        self.dropout = dropout

    def forward(self, edge_index, edge_type):
        x = self.node_emb.weight

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_type)

            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x


class MLPTripleScorer(nn.Module):
    def __init__(self, hidden_dim, mlp_hidden=256, dropout=0.3):
        super().__init__()

        input_dim = hidden_dim * 5

        self.net = nn.Sequential(
            nn.Linear(input_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(mlp_hidden // 2, 1)
        )

    def forward(self, z_drug, z_ind, z_se):
        x = torch.cat(
            [
                z_drug,
                z_ind,
                z_se,
                z_drug * z_se,
                z_ind * z_se,
            ],
            dim=-1
        )

        return self.net(x).squeeze(-1)


class ADRRGCNModel(nn.Module):
    def __init__(
        self,
        num_nodes,
        num_relations,
        emb_dim=64,
        hidden_dim=128,
        num_layers=2,
        dropout=0.3,
        mlp_hidden=256
    ):
        super().__init__()

        self.encoder = RGCNEncoder(
            num_nodes=num_nodes,
            num_relations=num_relations,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        self.scorer = MLPTripleScorer(
            hidden_dim=hidden_dim,
            mlp_hidden=mlp_hidden,
            dropout=dropout
        )

    def encode(self, edge_index, edge_type):
        return self.encoder(edge_index, edge_type)

    def score_batch(self, z, triples_batch):
        triples_batch = torch.as_tensor(
            triples_batch,
            dtype=torch.long,
            device=z.device
        )

        d_local = triples_batch[:, 0]
        i_local = triples_batch[:, 1]
        s_local = triples_batch[:, 2]

        d_global = d_local + drug_offset
        i_global = i_local + ind_offset
        s_global = s_local + se_offset

        z_drug = z[d_global]
        z_ind = z[i_global]
        z_se = z[s_global]

        return self.scorer(z_drug, z_ind, z_se)


# =========================================================
# Load trained model
# =========================================================
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ADRRGCNModel(
        num_nodes=num_nodes,
        num_relations=num_relations,
        emb_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        mlp_hidden=MLP_HIDDEN
    )

    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    return model, device


model, device = load_model()

edge_index = edge_index.to(device)
edge_type = edge_type.to(device)


# =========================================================
# Prediction helper
# =========================================================
@torch.no_grad()
def predict_side_effects(drug_name, indication_name, top_k=5):
    drug_name = drug_name.lower().strip()
    indication_name = indication_name.lower().strip()

    if drug_name not in drug2id:
        return None, f"Drug '{drug_name}' was not found in the training vocabulary."

    if indication_name not in ind2id:
        return None, f"Indication '{indication_name}' was not found in the training vocabulary."

    drug_id = drug2id[drug_name]
    ind_id = ind2id[indication_name]

    all_se_ids = list(se2id.values())

    triples = np.array(
        [(drug_id, ind_id, se_id) for se_id in all_se_ids],
        dtype=np.int64
    )

    z = model.encode(edge_index, edge_type)

    logits = model.score_batch(z, triples)
    probs = torch.sigmoid(logits).detach().cpu().numpy()

    results = pd.DataFrame({
        "side_effect": [id2se[se_id] for se_id in all_se_ids],
        "probability": probs
    })

    results = results.sort_values(
        by="probability",
        ascending=False
    ).reset_index(drop=True)

    results["predicted_label"] = results["probability"].apply(
        lambda x: "Likely" if x >= best_threshold else "Possible"
    )

    return results.head(top_k), None


# =========================================================
# Dropdown UI
# =========================================================
drug_options = sorted(drug_ind_pairs["drug_name"].unique())

selected_drug = st.selectbox(
    "Select drug",
    drug_options,
    index=0
)

filtered_indications = sorted(
    drug_ind_pairs[
        drug_ind_pairs["drug_name"] == selected_drug
    ]["indication_name"].unique()
)

if len(filtered_indications) == 0:
    st.warning("No indications found for the selected drug.")
    st.stop()

selected_indication = st.selectbox(
    "Select indication",
    filtered_indications,
    index=0
)

top_k = st.slider(
    "Number of side effects to show",
    min_value=1,
    max_value=20,
    value=5
)

st.markdown("---")

if st.button("Predict Side Effects"):
    results, error = predict_side_effects(
        selected_drug,
        selected_indication,
        top_k=top_k
    )

    if error:
        st.error(error)
    else:
        st.subheader("Predicted Side Effects")

        st.write(
            f"**Drug:** {selected_drug}  \n"
            f"**Indication:** {selected_indication}  \n"
            f"**Validation threshold:** {best_threshold:.2f}"
        )

        display_df = results[["side_effect", "predicted_label"]].copy()

        display_df = display_df.rename(columns={
            "side_effect": "Predicted Side Effect",
            "predicted_label": "Prediction"
        })

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
    )


   

# =========================================================
# Optional notes
# =========================================================
with st.expander("How this prediction works"):
    st.write(
        """
        The app takes the selected drug and indication, then evaluates that pair against every side effect in the model vocabulary. The RGCN creates node embeddings from the trained graph, and the MLP scorer assigns a score to each drug–indication–side-effect combination.
        A validation threshold of 85% is used for prediction. Side effects with scores above this threshold are presented as likely side effects.
        One limitation is that the model can only predict side effects that were included in the 500,000-row training subset. Side effects not represented in that subset are outside the model’s vocabulary and cannot be predicted by the current app.
        """
    )
