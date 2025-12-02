import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate, Dropout, Multiply
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from tqdm import tqdm
import time

DATA_PATH = "train-1.txt"
LATENT = 32
DROPOUT = 0.1
LEARNING_RATE = 0.0001
EPOCHS = 30
BATCH_SIZE = 1024

NEGATIVE_SAMPLE = 4

EVALUATION_NEGATIVE_COUNT = 99
TOP_K   = 20

OPTIMIZER = "adam"   # or "sgd"


class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = self.load_dataset()
        self.df_encoded, self.user_map, self.item_map = self.encoding_user_item()
        self.user_count = self.df_encoded["user_id"].max() + 1
        self.item_count = self.df_encoded["item_id"].max() + 1

        # Full data interactions
        self.interactions_all = self.ui_interactions(self.df_encoded)
        # LOO split
        self.train, self.test = self.train_test_split()
        self.interactions_train = self.ui_interactions(self.train)

        print("Number of users: ", self.user_count)
        print("Number of items: ", self.item_count)
        print("Number of interactions: ", len(self.df_encoded))
        print("Train size (LOO): ", len(self.train))
        print("Test size  (LOO): ", len(self.test))
        print("Data has been loaded successfully.")

    def load_dataset(self):
        # This function loads dataset file and return a dataframe (user,item)
        records = []
        with open(self.file_path, 'r') as file:
            for l in file:
                l_parts = l.strip().split()
                if len(l_parts) < 2:
                    continue
                user = l_parts[0]
                items = l_parts[1:]
                for item in items:
                    records.append((user, item))
        data_frame = pd.DataFrame(records, columns=['user', 'item'])
        return data_frame

    def encoding_user_item(self):
        # mapping encoded data to original data (user_id to user)
        user_cat = self.df["user"].astype(int).astype("category")
        item_cat = self.df["item"].astype(int).astype("category")
        user_map = dict(enumerate(user_cat.cat.categories))
        item_map = dict(enumerate(item_cat.cat.categories))
        return pd.DataFrame({
            "user_id": user_cat.cat.codes,
            "item_id": item_cat.cat.codes
        }), user_map, item_map

    def ui_interactions(self, df=None):
        if df is None:
            df = self.df_encoded
        return df.groupby("user_id")["item_id"].agg(set).to_dict()

    def train_test_split(self, seed=42):
        # leave one out of each user to make a test set
        test  = self.df_encoded.groupby("user_id").sample(1, random_state=seed)
        train = self.df_encoded.drop(test.index)
        return train.reset_index(drop=True), test.reset_index(drop=True)

    def decode_user(self, user_id):
        return self.user_map[user_id]

    def decode_item(self, item_id):
        return self.item_map[item_id]


def create_training_set(df_train, item_count, interactions, negative_sample=4):
    users = []
    items = []
    targets = []
    # For each positive sample, we take 4 random negative samples.
    for u, i in tqdm(df_train[["user_id", "item_id"]].values, desc="create training set"):
        # One positive sample
        users.append(u)
        items.append(i)
        targets.append(1.0)

        ui = interactions[u]
        for _ in range(negative_sample):
            # Take random negative samples for this user
            ni = np.random.randint(item_count)
            # Loop until we find an item that the user has not interacted with
            while ni in ui:
                ni = np.random.randint(item_count)
            users.append(u)
            items.append(ni)
            targets.append(0.0)

    users   = np.array(users, dtype=np.int32)
    items   = np.array(items, dtype=np.int32)
    targets = np.array(targets, dtype=np.float32)
    ri = np.random.permutation(len(users))
    return users[ri], items[ri], targets[ri]


def neucf(num_users, item_count, dim=32, dropout=0.25):
    user_in = Input(shape=(1,), dtype='int32', name='user_input')
    item_in = Input(shape=(1,), dtype='int32', name='item_input')

    # GMF Layer
    gmf_u = Flatten()(Embedding(num_users, dim, name='gmf_user')(user_in))
    gmf_i = Flatten()(Embedding(item_count, dim, name='gmf_item')(item_in))
    gmf_vector = Multiply()([gmf_u, gmf_i])

    # MLP Layer [64,32,16,8]
    mlp_u = Flatten()(Embedding(num_users, 32, name='mlp_user')(user_in))
    mlp_i = Flatten()(Embedding(item_count, 32, name='mlp_item')(item_in))
    mlp_vector = Concatenate()([mlp_u, mlp_i])
    mlp_vector = Dense(64, activation='relu')(mlp_vector)
    mlp_vector = Dropout(dropout)(mlp_vector)
    mlp_vector = Dense(32, activation='relu')(mlp_vector)
    mlp_vector = Dropout(dropout)(mlp_vector)
    mlp_vector = Dense(16, activation='relu')(mlp_vector)
    mlp_vector = Dropout(dropout)(mlp_vector)
    mlp_vector = Dense(8, activation='relu')(mlp_vector)

    # NeuMF layer
    vec = Concatenate()([gmf_vector, mlp_vector])
    output = Dense(1, activation='sigmoid', name='output')(vec)
    return Model([user_in, item_in], output)


def get_train_eval_splits(mode, data_loader):
    mode = mode.lower()
    if mode == "full":
        print("--> FULL data for training")
        df_train = data_loader.df_encoded
        df_test  = None
        interactions_for_train = data_loader.ui_interactions(df_train)
        interactions_for_eval  = None
    else:
        print("--> Leave one out (LOO) split")
        df_train = data_loader.train
        df_test  = data_loader.test
        interactions_for_train = data_loader.ui_interactions(df_train)
        interactions_for_eval = data_loader.interactions_all

    return df_train, df_test, interactions_for_train, interactions_for_eval


def evaluate(model, df_test, interactions, item_count, negative_count=99, top_k=20):
    print("\nEvaluating model")
    test_u = df_test['user_id'].values
    test_i = df_test['item_id'].values

    eu = []
    ei = []

    for u, i in tqdm(zip(test_u, test_i),total=len(test_u), desc="Sampling negatives"):
        eu.append(u)
        ei.append(i)
        interacted = interactions[u]
        added = 0
        while added < negative_count:
            j = np.random.randint(item_count)
            if j not in interacted and j != i:
                eu.append(u)
                ei.append(j)
                added += 1

    eu = np.array(eu, dtype=np.int32)
    ei = np.array(ei, dtype=np.int32)

    predictions = model.predict([eu, ei], batch_size=2048, verbose=1)

    predictions = predictions.reshape(-1, negative_count + 1)

    ndcgs = 0.0
    hits = 0

    for scores in predictions:
        s = scores[0]
        count_larger = np.sum(scores > s)
        rank = count_larger + 1

        if rank <= top_k:
            hits += 1
            ndcgs += 1.0 / np.log2(rank + 1)

    n_users = len(predictions)
    hr = hits / n_users
    ndcg = ndcgs / n_users
    return ndcg, hr


def save_neucf_recommendations(model, loader: DataLoader, outfile="outputs/recommendations_neucf.txt"):
    K = 20
    all_items = np.arange(loader.item_count, dtype=np.int32)
    lines = []
    outdir = os.path.dirname(outfile)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    for u in tqdm(range(loader.user_count), desc="Saving Top 20 Recommendations"):
        user_arr = np.full(loader.item_count, u, dtype=np.int32)
        scores = model.predict(
            [user_arr, all_items],
            batch_size=loader.item_count,
            verbose=0
        ).reshape(-1)

        interacted = loader.interactions_all.get(u)
        # Lower the score of the items that this user has interacted with
        if interacted:
            scores[list(interacted)] = -np.inf 

        ids = np.argpartition(-scores, K)[:K]
        ids = ids[np.argsort(-scores[ids])]

        user_og  = str(loader.user_map[u])
        items_og = [str(loader.item_map[i]) for i in ids]
        lines.append(f"{user_og} " + " ".join(items_og))

    with open(outfile, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved recommendations to: {outfile}")


def save_model(model, file_name="models/neucf_model.h5"):
    folder = os.path.dirname(file_name)
    if folder:
        os.makedirs(folder, exist_ok=True)
    model.save(file_name)
    print("Model", file_name, "has been saved.")


if __name__ == "__main__":
    data_loader = DataLoader(DATA_PATH)

    print("Training mode:")
    print("  1. Split  (LOO train/test)")
    print("  2. Full")

    choice = input().strip()

    if choice == "2":
        mode = "full"
    else:
        mode = "split"

    print(f"Selected mode: {mode.upper()}")


    df_train, df_test, interactions_for_train, interactions_for_eval = get_train_eval_splits(
        mode=mode,
        data_loader=data_loader
    )

    train_user, train_item, train_target = create_training_set(
        df_train=df_train,
        item_count=data_loader.item_count,
        interactions=interactions_for_train,
        negative_sample=NEGATIVE_SAMPLE
    )
    print("Training...")
    if OPTIMIZER.lower() == "sgd":
        optimizer = SGD(learning_rate=0.01, momentum=0.9)
    else:
        optimizer = Adam(learning_rate=LEARNING_RATE)

    model = neucf(
        num_users=data_loader.user_count,
        item_count=data_loader.item_count,
        dim=LATENT,
        dropout=DROPOUT
    )

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
    )

    es = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )

    history = model.fit(
        x=[train_user, train_item],
        y=train_target,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.1,
        callbacks=[es],
        verbose=1
    )

    # If there is a test set, we evaluate the model
    if df_test is not None:
        ndcg, hr = evaluate(
            model=model,
            df_test=df_test,
            interactions=interactions_for_eval,
            item_count=data_loader.item_count,
            negative_count=EVALUATION_NEGATIVE_COUNT,
            top_k=TOP_K)
        print(f"HR@{TOP_K}:   {hr:.4f}")
        print(f"NDCG@{TOP_K}: {ndcg:.4f}")


    # Plot history
    plt.figure()
    pd.Series(history.history['loss']).plot(logy=True, label="training")
    pd.Series(history.history['val_loss']).plot(logy=True, label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    ctime= int(round(time.time() * 1000))
    time_in_seconds = int(tf.timestamp())
    os.makedirs("results", exist_ok=True)
    plot_path = f"results/neucf_history_{ctime}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")

    plt.show()

    # Save recommendations into "outputs/" folder
    save_neucf_recommendations(
        model=model,
        loader=data_loader,
        outfile="outputs/neucf_recommendations.txt"
    )

    # Save model into "models/"" folder
    save_model(model, file_name="models/neucf_model.h5")



# Hyperparams tuning

# ------------------------------ Config A ------------------------------
# LATENT = 64
# DROPOUT = 0.1
# LEARNING_RATE = 0.0001
# EPOCHS = 30
# BATCH_SIZE = 1024
# --
# Training...
# Epoch 1/30
# 5299/5299 ━━━━━━━━━━━━━━━━━━━━ 31s 5ms/step - loss: 0.5006 - val_loss: 0.4291
# Epoch 2/30
# 5299/5299 ━━━━━━━━━━━━━━━━━━━━ 23s 4ms/step - loss: 0.4234 - val_loss: 0.4177
# Epoch 3/30
# 5299/5299 ━━━━━━━━━━━━━━━━━━━━ 22s 4ms/step - loss: 0.4066 - val_loss: 0.3935
# Epoch 4/30
# 5299/5299 ━━━━━━━━━━━━━━━━━━━━ 22s 4ms/step - loss: 0.3583 - val_loss: 0.3108
# Epoch 5/30
# 5299/5299 ━━━━━━━━━━━━━━━━━━━━ 25s 5ms/step - loss: 0.2865 - val_loss: 0.2704
# Epoch 6/30
# 5299/5299 ━━━━━━━━━━━━━━━━━━━━ 24s 4ms/step - loss: 0.2498 - val_loss: 0.2545
# Epoch 7/30
# 5299/5299 ━━━━━━━━━━━━━━━━━━━━ 24s 5ms/step - loss: 0.2293 - val_loss: 0.2443
# Epoch 8/30
# 5299/5299 ━━━━━━━━━━━━━━━━━━━━ 23s 4ms/step - loss: 0.2101 - val_loss: 0.2345
# Epoch 9/30
# 5299/5299 ━━━━━━━━━━━━━━━━━━━━ 40s 4ms/step - loss: 0.1908 - val_loss: 0.2274
# Epoch 10/30
# 5299/5299 ━━━━━━━━━━━━━━━━━━━━ 22s 4ms/step - loss: 0.1714 - val_loss: 0.2228
# Epoch 11/30
# 5299/5299 ━━━━━━━━━━━━━━━━━━━━ 22s 4ms/step - loss: 0.1532 - val_loss: 0.2195
# Epoch 12/30
# 5299/5299 ━━━━━━━━━━━━━━━━━━━━ 22s 4ms/step - loss: 0.1348 - val_loss: 0.2185
# Epoch 13/30
# 5299/5299 ━━━━━━━━━━━━━━━━━━━━ 22s 4ms/step - loss: 0.1178 - val_loss: 0.2194
# Epoch 14/30
# 5299/5299 ━━━━━━━━━━━━━━━━━━━━ 22s 4ms/step - loss: 0.1017 - val_loss: 0.2210
# Epoch 15/30
# 5299/5299 ━━━━━━━━━━━━━━━━━━━━ 22s 4ms/step - loss: 0.0872 - val_loss: 0.2252

# Evaluating model
# Sampling negatives: 100%|██████████| 31668/31668 [00:09<00:00, 3187.53it/s]
# 1547/1547 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step
# HR@20:   0.9420
# NDCG@20: 0.5785

# ------------------------------ Config B ------------------------------
# LATENT = 32
# DROPOUT = 0.1
# LEARNING_RATE = 0.0001
# EPOCHS = 30
# BATCH_SIZE = 1024
# --
# Training...
# Epoch 1/30
# 5299/5299 ━━━━━━━━━━━━━━━━━━━━ 34s 5ms/step - loss: 0.5015 - val_loss: 0.4232
# Epoch 2/30
# 5299/5299 ━━━━━━━━━━━━━━━━━━━━ 23s 4ms/step - loss: 0.4184 - val_loss: 0.4136
# Epoch 3/30
# 5299/5299 ━━━━━━━━━━━━━━━━━━━━ 21s 4ms/step - loss: 0.4027 - val_loss: 0.3618
# Epoch 4/30
# 5299/5299 ━━━━━━━━━━━━━━━━━━━━ 21s 4ms/step - loss: 0.3346 - val_loss: 0.2944
# Epoch 5/30
# 5299/5299 ━━━━━━━━━━━━━━━━━━━━ 22s 4ms/step - loss: 0.2769 - val_loss: 0.2678
# Epoch 6/30
# 5299/5299 ━━━━━━━━━━━━━━━━━━━━ 21s 4ms/step - loss: 0.2527 - val_loss: 0.2544
# Epoch 7/30
# 5299/5299 ━━━━━━━━━━━━━━━━━━━━ 21s 4ms/step - loss: 0.2357 - val_loss: 0.2443
# Epoch 8/30
# 5299/5299 ━━━━━━━━━━━━━━━━━━━━ 21s 4ms/step - loss: 0.2206 - val_loss: 0.2365
# Epoch 9/30
# 5299/5299 ━━━━━━━━━━━━━━━━━━━━ 21s 4ms/step - loss: 0.2059 - val_loss: 0.2308
# Epoch 10/30
# 5299/5299 ━━━━━━━━━━━━━━━━━━━━ 20s 4ms/step - loss: 0.1914 - val_loss: 0.2269
# Epoch 11/30
# 5299/5299 ━━━━━━━━━━━━━━━━━━━━ 21s 4ms/step - loss: 0.1773 - val_loss: 0.2237
# Epoch 12/30
# 5299/5299 ━━━━━━━━━━━━━━━━━━━━ 21s 4ms/step - loss: 0.1634 - val_loss: 0.2219
# Epoch 13/30
# 5299/5299 ━━━━━━━━━━━━━━━━━━━━ 20s 4ms/step - loss: 0.1493 - val_loss: 0.2219
# Epoch 14/30
# 5299/5299 ━━━━━━━━━━━━━━━━━━━━ 21s 4ms/step - loss: 0.1363 - val_loss: 0.2229
# Epoch 15/30
# 5299/5299 ━━━━━━━━━━━━━━━━━━━━ 21s 4ms/step - loss: 0.1238 - val_loss: 0.2244
# Epoch 16/30
# 5299/5299 ━━━━━━━━━━━━━━━━━━━━ 21s 4ms/step - loss: 0.1121 - val_loss: 0.2279
# Evaluating model
# Sampling negatives: 100%|██████████| 31668/31668 [00:09<00:00, 3304.30it/s]
# 1547/1547 ━━━━━━━━━━━━━━━━━━━━ 4s 2ms/step
# HR@20:   0.9417
# NDCG@20: 0.5684

# ------------------------------ Config C ------------------------------
# LATENT = 16
# DROPOUT = 0.1
# LEARNING_RATE = 0.0001
# EPOCHS = 30
# BATCH_SIZE = 1024