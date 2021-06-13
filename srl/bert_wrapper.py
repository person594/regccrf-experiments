import torch
import torch.nn as nn
from transformers import BertModel, BertForTokenClassification, BertTokenizerFast, RobertaModel, RobertaTokenizerFast



class BertWrapper(nn.Module):
    def __init__(self, roberta=False):
        super().__init__()
        if roberta:
            self.bert = RobertaModel.from_pretrained('roberta-base')
            self.tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
        else:
            self.bert = BertModel.from_pretrained('bert-base-uncased')
            self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.n_units = self.bert.config.hidden_size
        self.max_length = self.bert.config.max_position_embeddings
        

    def forward(self, inputs):
        sequence_length = inputs.shape[1]
        if sequence_length <= self.max_length:
            output = self.bert(inputs)[0]
        else:
            output_windows = []
            for window_start in range(0, sequence_length, self.max_length//2):
                input_window = inputs[:,window_start:window_start+self.max_length]
                window_length = input_window.shape[1]
                # input_window: float32[batch_size, window_length]
                output_window = self.bert(input_window)[0]
                # output_window: float32[batch_size, window_length]
                output_windows.append(output_window)
            window_start_index = self.max_length//4
            window_end_index = 3*self.max_length//4
            parts = []
            parts.append(output_windows[0][:,:window_end_index])
            for output_window in output_windows[1:-1]:
                parts.append(output_window[:,window_start_index:window_end_index])
            parts.append(output_windows[-1][:,window_start_index:])
            output = torch.cat(parts, dim=1)
        return output
