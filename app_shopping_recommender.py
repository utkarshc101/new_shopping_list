
import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import pickle
import numpy as np

# === Define model class ===
class Recommender(nn.Module):
    def __init__(self, n_users, n_items, n_age, n_income, embed_dim=64):
        super().__init__()
        self.user_embed = nn.Embedding(n_users, embed_dim)
        self.item_embed = nn.Embedding(n_items, embed_dim)
        self.age_embed = nn.Embedding(n_age, embed_dim)
        self.income_embed = nn.Embedding(n_income, embed_dim)
        self.fc = nn.Linear(embed_dim * 4, 1)

    def forward(self, user, item, age, income):
        x = torch.cat([
            self.user_embed(user),
            self.item_embed(item),
            self.age_embed(age),
            self.income_embed(income)
        ], dim=1)
        return torch.sigmoid(self.fc(x)).squeeze()

# === Load data and encoders ===
df_prod = pd.read_csv("product.csv")
with open("torch_encoders.pkl", "rb") as f:
    enc = pickle.load(f)

item_enc = enc["item_enc"]
age_enc = enc["age_enc"]
inc_enc = enc["inc_enc"]

# Convert income labels from $ to ₹ (multiplying by 50)
converted_income_labels = []
for label in inc_enc.classes_:
    if "Under $" in label:
        amount = int(label.replace("Under $", "").replace("K", "")) * 50
        converted_income_labels.append(f"Under ₹{amount:,}00")
    elif "$" in label and "-" in label:
        parts = label.replace("$", "").replace("K", "").split("-")
        low = int(parts[0]) * 50
        high = int(parts[1]) * 50
        converted_income_labels.append(f"₹{low:,}00–₹{high:,}00")
    else:
        converted_income_labels.append(label)  # fallback

# Update selectbox display labels
inc_label_map = dict(zip(inc_enc.classes_, converted_income_labels))

n_users = len(enc["user_enc"].classes_)   # ✅ Match training size
n_items = len(enc["item_enc"].classes_)
n_age = len(enc["age_enc"].classes_)
n_income = len(enc["inc_enc"].classes_)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Recommender(n_users, n_items, n_age, n_income).to(device)
model.load_state_dict(torch.load("recommender_model.pt", map_location=torch.device('cpu')))
model.eval()

# === Streamlit UI ===
st.title("🛒 AI Shopping Cart Recommender (PyTorch)")

age_option = st.selectbox("Select your Age Group", age_enc.classes_)
#inc_option = st.selectbox("Select your Income Group", inc_enc.classes_)
selected_income_display = st.selectbox("Select your Income Group (in ₹):", converted_income_labels)
# Map back to encoder label
income_label = [k for k, v in inc_label_map.items() if v == selected_income_display][0]

selected_items = st.multiselect("Select a few items you like or often buy:",
                                df_prod["SUB_COMMODITY_DESC"].dropna().unique())

if st.button("Suggest More Products"):
    if not selected_items:
        st.warning("Please select at least one item.")
    else:
        # Convert user profile
        age_idx = torch.tensor([age_enc.transform([age_option])[0]], dtype=torch.long).to(device)
        #inc_idx = torch.tensor([inc_enc.transform([inc_option])[0]], dtype=torch.long).to(device)
        inc_idx = torch.tensor([inc_enc.transform([income_label])[0]], dtype=torch.long).to(device)
        # Recommend items not in selected list
        selected_ids = df_prod[df_prod["SUB_COMMODITY_DESC"].isin(selected_items)]["PRODUCT_ID"]
        all_items = df_prod[~df_prod["PRODUCT_ID"].isin(selected_ids)]
        known_items = set(item_enc.classes_)
        all_items = all_items[all_items["PRODUCT_ID"].isin(known_items)]
        all_items["item_enc"] = item_enc.transform(all_items["PRODUCT_ID"])



        user_batch = torch.zeros(len(all_items), dtype=torch.long).to(device)  # dummy user id 0
        item_batch = torch.tensor(all_items["item_enc"].values, dtype=torch.long).to(device)
        age_batch = age_idx.repeat(len(all_items)).to(device)
        inc_batch = inc_idx.repeat(len(all_items)).to(device)

        with torch.no_grad():
            scores = model(user_batch, item_batch, age_batch, inc_batch).cpu().numpy()

        all_items["score"] = scores
        top_items = all_items.sort_values("score", ascending=False).head(10)["SUB_COMMODITY_DESC"].tolist()

        st.subheader("🧠 Recommended for You:")
        for item in top_items:
            st.write("•", item)

