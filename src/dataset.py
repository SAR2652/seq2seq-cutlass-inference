from torch.utils.data import Dataset


class PolynomialDataset(Dataset):
    def __init__(self, factors, tokenizer, expansions=None):
        self.factors = factors
        self.expansions = expansions
        self.tokenizer = tokenizer

    def __len__(self):
        return min(len(self.factors), len(self.expansions))

    def __getitem__(self, idx):
        """Obtain a single tuple comprising of a tokenized factor and its
        corresponding tokenized expansion"""
        factor = self.factors[idx]
        if self.expansions is not None:
            expansion = self.expansions[idx]
        else:
            expansion = None
        factor_input_ids, expansion_label_ids = \
            self.tokenizer.encode(factor, expansion)
        item = dict()
        item['factor'] = factor
        item['expansion'] = expansion
        item['input_ids'] = factor_input_ids
        item['target_ids'] = expansion_label_ids
        # if self.framework == 'pytorch':
        #     item['input_ids'] = item['input_ids'].view(-1, 1)
        #     item['target_ids'] = item['target_ids'].view(-1, 1)
        return item
