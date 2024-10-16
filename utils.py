import re
import os
import json
import torch
import emoji
import transformers

from torch.utils.data import Dataset

emoji_list = emoji.EMOJI_DATA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_json(file_path):
    with open(file_path, "r") as f:
        file = json.load(f)
    return file

def Preprocess(df):
    df["text"] = df["text"].str.lower()
    df["text"] = df["text"].str.replace(r"(:[^:]+:)", "")
    return df

def NotEmpty(df):
    df["text"] = df["text"].str.replace(" ", "")
    df = df[(df["text"] != "") & (df["text"].notnull())]
    return df

def rmemoji(text):
    return emoji.replace_emoji(text, "").strip()

def emoji2description(text):
    return emoji.replace_emoji(text, replace = lambda chars, emoji_list: " " + " ".join(emoji_list["zh"].split("_")).strip(":") + " ")

def remove_punctuation_regex(input_string):
    return re.sub(r"[^\w\s]", " ", input_string)

def Contain_KeyWord(text, keyword):
    return any(e in str(text) for e in keyword)

def Remove_Emoji(df):
    df_rmemoji = df.copy()
    df_rmemoji["text"] = df_rmemoji["text"].astype(str).apply(rmemoji)
    df_rmemoji = NotEmpty(df_rmemoji)
    return df_rmemoji

def Emoji2Desc(df):
    df_emoji2desc = df.copy()
    df_emoji2desc["text"] = df_emoji2desc["text"].astype(str).apply(emoji2description)
    df_emoji2desc = NotEmpty(df_emoji2desc)
    return df_emoji2desc

def Remove_Punc(df):
    df_rmpunc = df.copy()
    df_rmpunc["text"] = df_rmpunc["text"].astype(str).apply(remove_punctuation_regex)
    df_rmpunc = NotEmpty(df_rmpunc)
    return df_rmpunc

class CustomDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataset
        self.text = dataset['text']
        self.targets = dataset['target']
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens = True,
            max_length = self.max_len,
            padding = "max_length",
            return_token_type_ids = True,
            truncation = True
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        return {
            "ids": torch.tensor(ids, dtype = torch.long),
            "mask": torch.tensor(mask, dtype = torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype = torch.long),
            "targets": torch.tensor(self.targets[index], dtype = torch.float)
        }
    
class BERTClass(torch.nn.Module):
    def __init__(self, model):
        super(BERTClass, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained(model)
        self.l2 = torch.nn.Dropout(0.3)
        ## BERT-base
        self.l3 = torch.nn.Linear(768, 4)

        ## BERT-large
        # self.l3 = torch.nn.Linear(1024, 4)

    def forward(self, ids, mask, token_type_ids):
        output_1 = self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids)[0]
        output_2 = self.l2(output_1)
        output = self.l3(output_2[:, 0, :])
        return output
    
def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs.logits, targets)

def load_ckp(checkpoint_fpath, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)

    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint["state_dict"])

    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint["optimizer"])

    # initialize valid_loss_min from checkpoint to valid_loss_min
    valid_loss_min = checkpoint["valid_loss_min"]

    return model, optimizer, checkpoint["epoch"], valid_loss_min.item()

def save_ckp(state, is_best, checkpoint_path, best_model_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    f_path = checkpoint_path
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, f_path)
    # if it is a best model, min validation loss
    if is_best:
        best_fpath = os.path.join(best_model_path, 'checkpoint.pt')
        # copy that checkpoint file to best path given, best_model_path
        torch.save(state, best_fpath)

def train_model(start_epochs,  n_epochs, valid_loss_min_input,
                training_loader, validation_loader, model,
                optimizer, checkpoint_path, best_model_path, val_targets, val_outputs, tokenizer, device = device):

    valid_loss_min = valid_loss_min_input

    for epoch in range(start_epochs, n_epochs + 1):
        train_loss = 0
        valid_loss = 0

        model.train()
        print(f"### Epoch {epoch} training... ###")
        for batch_idx, data in enumerate(training_loader):
            ids = data["ids"].to(device, dtype = torch.long)
            mask = data["mask"].to(device, dtype = torch.long)
            token_type_ids = data["token_type_ids"].to(device, dtype = torch.long)
            targets = data["targets"].to(device, dtype = torch.float)

            outputs = model(ids, mask, token_type_ids)

            optimizer.zero_grad()
            loss = loss_fn(outputs, targets)

            loss.backward()
            optimizer.step()
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.item() - train_loss))

        print(f"### Epoch {epoch} evaluating... ###")
        model.eval()
        with torch.no_grad():
            for batch_idx, data in enumerate(validation_loader, 0):
                ids = data["ids"].to(device, dtype = torch.long)
                mask = data["mask"].to(device, dtype = torch.long)
                token_type_ids = data["token_type_ids"].to(device, dtype = torch.long)
                targets = data["targets"].to(device, dtype = torch.float)
                outputs = model(ids, mask, token_type_ids)

                loss = loss_fn(outputs, targets)
                valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.item() - valid_loss))
                val_targets.extend(targets.cpu().detach().numpy().tolist())
                val_outputs.extend(torch.sigmoid(outputs.logits).cpu().detach().numpy().tolist())

            train_loss = train_loss/len(training_loader)
            valid_loss = valid_loss/len(validation_loader)
            print(f"Epoch: {epoch} \tAvgerage Training Loss: {round(train_loss, 6)} \tAverage Validation Loss: {round(valid_loss, 6)}")

            # create checkpoint variable and add important data
            checkpoint = {
                "epoch": epoch,
                "valid_loss_min": valid_loss,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }

            ## save the model if validation loss has decreased
            if valid_loss <= valid_loss_min:
                print(f"Validation loss decreased ({round(valid_loss_min, 6)} --> {round(valid_loss, 6)}).  Saving model ...")
                # save checkpoint as best model
                save_ckp(checkpoint, True, checkpoint_path, best_model_path)
                model.save_pretrained(best_model_path, safe_serialization = True)
                tokenizer.save_pretrained(best_model_path)
                valid_loss_min = valid_loss

        print(f"### Epoch {epoch}  End ###\n")

    return model