import csv
from torch.utils.data import Dataset


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

    def collate_fn(self, sessions):
        batch_input_token = []
        for session in sessions:
            input_token = ""
            for line in session:
                speaker, utt, emotion = line
                input_token += utt
            batch_input_token.append(input_token)

        return batch_input_token
