import os
import torch
from torch.utils.data import Dataset, DataLoader

NB_SHORT_SHOP = 4

def load_feature_from_file(file_path):
    try:
        return torch.load(file_path)
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None


class CustomerShopDataset(Dataset):
    def __init__(self, file_path):
        self.embeddings = self.load_embeddings(file_path)

    def load_embeddings(self, file_path):
        embeddings = []
        #print(f"Attempting to open file: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                try:
                    parts = line.strip().split('—>')
                    #print(parts)
                    customer = parts[0]
                    shops = parts[1:] if len(parts) > 1 else []
                    customer_features = customer.split('&')
                    customer_embed = []
                    #print("========customer_features=====")
                    #print(customer_features)

                    for feature in customer_features:
                        #print("++++++++++++")
                        try:
                            attribute, index, path = feature.split('|')
                            #print(feature.split('|'))
                            embed = load_feature_from_file(path)
                            #print(embed)
                            if embed is not None:
                                customer_embed.append(embed)
                            else:
                                print(f"Skipped missing or faulty file: {path}")
                        except ValueError:
                            print(f"Error parsing customer feature: {feature}")
                            continue
                        except FileNotFoundError:
                            print(f"File not found: {path}")
                            continue
                    if not customer_embed:
                        continue
                    customer_embed = torch.cat(customer_embed, dim=-1)

                    shop_embeds = []
                    label = None
                    for i, shop in enumerate(shops):
                        if i == len(shops) - 1:
                            label = torch.tensor(int(shop.split("&")[6].split("|")[1]), dtype=torch.long)
                            continue
                        shop_features = shop.split('&')
                        shop_embed = []
                        #print("++++++++++++++++")
                        #print(shop_features)
                        for feature in shop_features:
                            if '|' in feature:
                                try:
                                    attribute, index, path = feature.split('|')
                                    #print(feature)
                                    #print(path)
                                    embed = load_feature_from_file(path)
                                    #print(embed)
                                    if embed is not None:
                                        shop_embed.append(embed)
                                    else:
                                        print(f"Skipped missing or faulty file: {path}")
                                except ValueError:
                                    print(f"Error parsing shop feature: {feature}")
                                    continue
                                except FileNotFoundError:
                                    print(f"File not found: {path}")
                                    continue
                            else:
                                embed = torch.tensor([float(feature)])
                                shop_embed.append(embed)
                        if not shop_embed:
                            continue
                        shop_embed = torch.cat(shop_embed, dim=-1)
                        shop_embeds.append(shop_embed)
                    if not shop_embeds:
                        continue
                    embeddings.append((customer_embed, shop_embeds, label))
                except Exception as e:
                    print(f"Error processing line: {line.strip()}")
                    print(e)
        return embeddings

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        #return self.embeddings[idx]
        c_t, s_l, l = self.embeddings[idx]

        if len(s_l) <= NB_SHORT_SHOP:
            #short_s_l = s_l[NB_SHORT_SHOP:]
            short_s_l = s_l
        else:
            short_s_l = s_l[:NB_SHORT_SHOP]
        return c_t, s_l, short_s_l, l


def custom_collate_fn(batch):
    customer_embeds, shop_embeds_list, short_shop_embeds_list,  labels = zip(*batch)

    # Find max length of shop_embeds
    max_shops_len = max(len(shop_embeds) for shop_embeds in shop_embeds_list)
    max_embed_size = max(max(shop_embed.shape[-1] for shop_embed in shop_embeds) for shop_embeds in shop_embeds_list)

    padded_shop_embeds = []
    for shop_embeds in shop_embeds_list:
        # Pad shop_embeds to have the same length
        if len(shop_embeds) < max_shops_len:
            padding = [torch.zeros(max_embed_size, dtype=torch.float32)] * (max_shops_len - len(shop_embeds))
            padded_shop_embeds.append(shop_embeds + padding)
        else:
            padded_shop_embeds.append(shop_embeds)

        # Ensure all shop embeds have the same size
        for i in range(len(padded_shop_embeds[-1])):
            if padded_shop_embeds[-1][i].shape[-1] != max_embed_size:
                padded_shop_embeds[-1][i] = torch.cat(
                    [padded_shop_embeds[-1][i], torch.zeros(max_embed_size - padded_shop_embeds[-1][i].shape[-1])]
                )


    # Find max length of short_shop_embeds
    max_short_shops_len = max(len(short_shop_embeds) for short_shop_embeds in short_shop_embeds_list)
    max_embed_size = max(max(short_shop_embed.shape[-1] for short_shop_embed in short_shop_embeds) for short_shop_embeds in short_shop_embeds_list)

    padded_short_shop_embeds = []
    for short_shop_embeds in short_shop_embeds_list:
        # Pad short_shop_embeds to have the same length
        if len(short_shop_embeds) < max_short_shops_len:
            padding = [torch.zeros(max_embed_size, dtype=torch.float32)] * (max_short_shops_len - len(short_shop_embeds))
            padded_short_shop_embeds.append(short_shop_embeds + padding)
        else:
            padded_short_shop_embeds.append(short_shop_embeds)

        # Ensure all short_shop embeds have the same size
        for i in range(len(padded_short_shop_embeds[-1])):
            if padded_short_shop_embeds[-1][i].shape[-1] != max_embed_size:
                padded_short_shop_embeds[-1][i] = torch.cat(
                    [padded_short_shop_embeds[-1][i], torch.zeros(max_embed_size - padded_short_shop_embeds[-1][i].shape[-1])]
                )

    customer_embeds = torch.stack(customer_embeds)
    shop_embeds = torch.stack([torch.stack(embeds) for embeds in padded_shop_embeds])
    short_shop_embeds = torch.stack([torch.stack(embeds) for embeds in padded_short_shop_embeds])
    labels = torch.stack(labels)

    return customer_embeds, shop_embeds, short_shop_embeds, labels


if __name__ == '__main__':
    #file_path = "/Users/zhuxiaoxu/Documents/code/heran/customer_shop_embed_path_new.txt"
    file_path = "/Users/zhuxiaoxu/Documents/code/heran/customer_shop_embed_path_new_h300.txt"
    dataset = CustomerShopDataset(file_path)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)
    for batch in data_loader:
        c, s, short_s, l = batch
        print("=== batch ===")
        print(c.shape)
        print(s.shape)
        print(short_s.shape)
        print(l)
    print(f"Loaded {len(dataset)} customers' embeddings")
