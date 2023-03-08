from torch.utils.data import Dataset
from transformers import RobertaTokenizer
import csv
from torch.utils.data import Dataset
import torch


def split(session):
    final_data = []
    split_session = []
    for line in session:
        split_session.append(line)
        final_data.append(split_session[:])
    return final_data


class CustomDataLoader(Dataset):
    def __init__(self, data_path):
        f = open(data_path, 'r')
        rdr = csv.reader(f)
        
        emo_set = set()
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

        self.session_dataset = []
        session = []
        speaker_set = []

        pre_sess = 's'
        for i, line in enumerate(rdr):
            if i == 0:
                header = line
                utt_idx = header.index('Utterance')
                speaker_idx = header.index('Speaker')
                emo_idx = header.index('Emotion')
                sess_idx = header.index('Dialogue_ID')
            else:
                utt = line[utt_idx]
                speaker = line[speaker_idx]
                if speaker in speaker_set:
                    uniq_speaker = speaker_set.index(speaker)
                else:
                    speaker_set.append(speaker)
                    uniq_speaker = speaker_set.index(speaker)
                emotion = line[emo_idx]
                sess = line[sess_idx]

                if pre_sess == 's' or sess == pre_sess:
                    session.append([uniq_speaker, utt, emotion])
                else:
                    self.session_dataset += split(session)
                    session = [[uniq_speaker, utt, emotion]]
                    speaker_set = []
                    emo_set.add(emotion)
                pre_sess = sess
        self.session_dataset += split(session)
        self.emoList = sorted(emo_set)
        self.emoList = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
        f.close()

    def __len__(self):
        return len(self.session_dataset)

    def __getitem__(self, idx):
        return self.session_dataset[idx]
    
    def padding(self, batch_input_token):
        batch_token_ids, batch_attention_masks = batch_input_token['input_ids'], batch_input_token['attention_mask']
        trunc_batch_token_ids, trunc_batch_attention_masks = [], []
        for batch_token_id, batch_attention_mask in zip(batch_token_ids, batch_attention_masks):
            if len(batch_token_id) > self.tokenizer.model_max_length:
                trunc_batch_token_id = [batch_token_id[0]] + batch_token_id[1:][-self.tokenizer.model_max_length+1:]
                trunc_batch_attention_mask = [batch_attention_mask[0]] + batch_attention_mask[1:][-self.tokenizer.model_max_length+1:]
                trunc_batch_token_ids.append(trunc_batch_token_id)
                trunc_batch_attention_masks.append(trunc_batch_attention_mask)
            else:
                trunc_batch_token_ids.append(batch_token_id)
                trunc_batch_attention_masks.append(batch_attention_mask)
        
        max_length = max([len(x) for x in trunc_batch_token_ids])
        padding_tokens, padding_attention_masks = [], []
        for batch_token_id, batch_attention_mask in zip(trunc_batch_token_ids, trunc_batch_attention_masks):
            padding_tokens.append(batch_token_id + [self.tokenizer.pad_token_id for _ in range(max_length-len(batch_token_id))])
            padding_attention_masks.append(batch_attention_mask + [0 for _ in range(max_length-len(batch_token_id))]                                                        )
        return torch.tensor(padding_tokens), torch.tensor(padding_attention_masks)

    def collate_fn(self, sessions):
        batch_input, batch_labels = [], []
        batch_PM_input = []
        for session in sessions:
            input_str = self.tokenizer.cls_token
            
            # PM
            current_speaker, current_utt, current_emotion = session[-1]
            PM_input = []
            for i, line in enumerate(session):
                speaker, utt, emotion = line
                input_str += " " + utt + self.tokenizer.sep_token
                if i < len(session)-1 and current_speaker == speaker:
                    PM_input.append(self.tokenizer.encode(utt, add_special_tokens=True, return_tensors='pt'))
                    
            # CoM
            batch_input.append(input_str)
            batch_labels.append(self.emoList.index(emotion))
            batch_PM_input.append(PM_input)
        batch_input_token = self.tokenizer(batch_input, add_special_tokens=False)
        batch_padding_token, batch_padding_attention_mask = self.padding(batch_input_token)
        
        return batch_padding_token, batch_padding_attention_mask, batch_PM_input, torch.tensor(batch_labels)
