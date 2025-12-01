import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate, Dropout, Multiply
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = self.load_dataset()
        self.df_encoded, self.user_map, self.item_map = self.encoding_user_item()
        self.user_count = self.df_encoded["user_id"].max() + 1
        self.item_count = self.df_encoded["item_id"].max() + 1
        self.train, self.test = self.train_test_split()
        self.interactions = self.ui_interactions()
        print("Number of users: ", self.user_count)
        print("Number of items: ", self.item_count)
        print("Number of interactions: ", len(self.df_encoded))
        print("Train set size: ", len(self.train))
        print("Test set size: ", len(self.test))
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
        user_cat = self.df["user"].astype(int).astype("category")
        item_cat = self.df["item"].astype(int).astype("category")
        # mapping encoded data to original data (user_id to user)
        user_map = dict(enumerate(user_cat.cat.categories))
        item_map = dict(enumerate(item_cat.cat.categories))
        return pd.DataFrame({
            "user_id": user_cat.cat.codes,
            "item_id": item_cat.cat.codes
        }), user_map, item_map

    def ui_interactions(self, df=None):
        return (self.df_encoded if df is None else df).groupby("user_id")["item_id"].agg(set).to_dict()

    def train_test_split(self, seed=42):
        # leave one out of each item to make a test set
        test  = self.df_encoded.groupby("user_id").sample(1, random_state=seed)
        train = self.df_encoded.drop(test.index)
        return train.reset_index(drop=True), test.reset_index(drop=True)

    def decode_user(self, user_id):
        return self.user_map[user_id]

    def decode_item(self, item_id):
        return self.item_map[item_id]

def create_training_set(df_train, item_count, interactions, negative_count = 4):
    users = []
    items = []
    targets = []
    # For each positive sample, we take 4 random negative samples.
    for u, i in tqdm(df_train[["user_id", "item_id"]].values, desc="Make training set with LOO (Leave one out)"):
        # One positive sample
        users.append(u)
        items.append(i)
        targets.append(1)
        ui = interactions[u]
        for _ in range(negative_count):
            # Take 4 random negative samples for this user
            ni = np.random.randint(item_count)
            # Loop until we find an item that the user has not interacted with
            while ni in ui:
                ni = np.random.randint(item_count)
            users.append(u)
            items.append(ni)
            targets.append(0)
    users  = np.array(users, dtype=np.int32)
    items  = np.array(items, dtype=np.int32)
    targets = np.array(targets, dtype=np.int8)
    ri = np.random.permutation(len(users))
    return users[ri], items[ri], targets[ri]

def neucf(num_users, item_count, dim=32, dropout=0.25):
    user_in = Input(shape=(1,), dtype='int32', name='user_input')
    item_in = Input(shape=(1,), dtype='int32', name='item_input')
    # GMF Layer
    gmf_u = Flatten()(Embedding(num_users, dim, name='gmf_user')(user_in))
    gmf_i = Flatten()(Embedding(item_count, dim, name='gmf_item')(item_in))
    gmf_vector = Multiply()([gmf_u, gmf_i])
    # MLP Layer
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



data_loader = DataLoader("train-1.txt")

LATENT = 32
DROPOUT = 0.25
LEARNING_RATE = 0.001
EPOCHS = 30
BATCH_SIZE = 1024

train_user, train_item, train_target = create_training_set(
    df_train= data_loader.train,
    item_count=data_loader.item_count,
    interactions=data_loader.interactions,
    negative_count=4)

es = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

model = neucf(
    num_users=data_loader.user_count,
    item_count=data_loader.item_count,
    dim=LATENT,
    dropout=DROPOUT
)

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss="binary_crossentropy",
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

def save_neucf_recommendations(model, loader: DataLoader, outfile="neucf_recommendations.txt"):
    K = 20
    all_items = np.arange(loader.item_count, dtype=np.int32)
    lines = []
    for u in tqdm(range(loader.user_count), desc="Users"):
        user_arr = np.full(loader.item_count, u, dtype=np.int32)
        scores = model.predict(
            [user_arr, all_items],
            batch_size=loader.item_count,
            verbose=0
        ).reshape(-1)
        hist = loader.interactions.get(u)
        if hist:
            scores[list(hist)] = -np.inf

        idx = np.argpartition(-scores, K)[:K]
        idx = idx[np.argsort(-scores[idx])]

        user_raw  = str(loader.user_map[u])
        items_raw = [str(loader.item_map[i]) for i in idx]
        lines.append(f"{user_raw} " + " ".join(items_raw))
    with open(outfile, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved recommendations to: {outfile}")

save_neucf_recommendations( model, loader=data_loader, outfile="neucf_recommendations.txt")

def save_model(model, file_name="model.h5"):
    model.save(file_name)
    print("Model ",file_name, " has been saved.")

def load_model(file_name):
    # load the model from .h5 file
    the_model = tf.keras.models.load_model(
        file_name,
        compile=False
    )
    print("The model ",{file_name}, "has been loaded successfully.")
    return the_model