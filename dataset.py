import torch
import torch.nn as nn
from torch.utils.data import Dataset


class BilingualDataset(Dataset):

    def __init__(self,ds,tokenizer_src,tokenizer_tgt,src_lang,tgt_lang,seq_len):
        super().__init__()
        self.seq_len=seq_len
        self.ds=ds
        self.tokenizer_src=tokenizer_src
        self.tokenizer_tgt=tokenizer_tgt
        self.src_lang=src_lang
        self.tgt_lang=tgt_lang


        #tokenize Start of sentence
        self.sos_token = torch.tensor([tokenizer_src.token_to_id(['[SOS]'])],dtype=torch.int64)

        #tokenize End of sentence
        self.eos_token = torch.tensor([tokenizer_src.token_to_id(['[EOS]'])],dtype=torch.int64)

        # tokenize the paddings
        self.pad_token = torch.tensor([tokenizer_src.token_to_id(['[PAD]'])],dtype=torch.int64)


    def __len__(self):
        return len(self.ds)
    

    def __getitem__(self, index):
        src_target_pair= self.ds[index]
        src_text=src_target_pair['translation'][self.src_lang]
        tgt_text=src_target_pair['translation'][self.tgt_lang]


        # spliting the sentence into words and then map those words into the corresponding number in vocabulary
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        #sentences that are not fullfiling the full length need to be padded that is why we have mask in attention as to discard
        #their influence in analisys as model works with fixed length.

        # we calculate how many 'filler' padded words we have in each side of input which is how many are needed to reach length

        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2 # start and end tokens SOS and EOS
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1 # here we have only SOS

        if enc_num_padding_tokens <0 or dec_num_padding_tokens<0:
            raise ValueError('sentence is too long')

        # ADD SOS and EOS to the source text
        encoder_input= torch.cat([
            self.sos_token,
            torch.tensor(enc_input_tokens,dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] *enc_num_padding_tokens,dtype=torch.int64)
        ])

        # add SOS to the decoder input
        decoder_input = torch.cat([
            self.sos_token,
            torch.tensor(dec_input_tokens,dtype=torch.int64),
            torch.tensor([self.pad_token]*dec_num_padding_tokens,dtype=torch.int64)
        ])

        # add the Eos to the label (what we expect as the output from the decoder)
        label= torch.cat([
            torch.tensor(dec_input_tokens,dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token]*dec_num_padding_tokens,dtype=torch.int64)
            ])
        
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.sizer(0)== self.seq_len

        return{
            'encoder_input': encoder_input,
            'decoder_input': decoder_input,
            'encoder_mask' : (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), #(1,1,seq_len)
            'decoder_mask' : (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), #(1,seq_len) & (1,seq_len,seq_len)
            'label': label, #(seq_len)
            'src_text': src_text,
            'tgt_text': tgt_text
        }   
    
def causal_mask(size):
    mask = torch.triu(torch.ones(1,size,size), diagonal=1).dtype=(torch.int)
    return mask ==0