import torch
import cv2
from torchvision import transforms
import torch.nn as nn
CHARS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-']


class CRNN(nn.Module):
    def __init__(self, numChannels=1, class_num=11):
        super(CRNN, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=numChannels, out_channels=64,
                      kernel_size=(3, 3), stride=1, padding=1),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=(3, 3), stride=1, padding=1),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(in_channels=128, out_channels=256,
                      kernel_size=(3, 3), stride=1, padding=1),
            nn.Conv2d(in_channels=256, out_channels=256,
                      kernel_size=(3, 3), stride=1, padding=1),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Conv2d(in_channels=256, out_channels=512,
                      kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.Conv2d(in_channels=512, out_channels=512,
                      kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(2, 1)),
        )
        self.lstm = nn.LSTM(512, 256, 2, batch_first=True, bidirectional=True)
        self.fc_layer = nn.Linear(in_features=512, out_features=class_num)

    def map2seq(self, input_tensor):
        seq_tensor = input_tensor.squeeze(dim=2)
        seq_tensor = seq_tensor.permute(0, 2, 1)
        return seq_tensor

    def seq2label(self, seq_tensor):
        lstm_output, (_, _) = self.lstm(seq_tensor)
        batch_size, seq_len, hidden_unit = lstm_output.shape
        flattened_output = lstm_output.reshape(-1, hidden_unit)
        logits = self.fc_layer(flattened_output)
        logits = logits.reshape(batch_size, seq_len, -1)
        logits = logits.permute(1, 0, 2)
        log_probs = logits.log_softmax(2)
        return log_probs

    def forward(self, x):
        x = self.feature_extractor(x)
        seq_tensor = self.map2seq(x)
        log_probs = self.seq2label(seq_tensor)
        return log_probs


def preprocessing(image):
    image = cv2.resize(image, dsize=(120, 32))
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    median_img = cv2.medianBlur(hsv_img, ksize=3)
    laplacian_img = cv2.Laplacian(median_img, -1, ksize=3)
    s = laplacian_img[:, :, 1]
    h = laplacian_img[:, :, 0]
    final_mask = cv2.bitwise_or(h, s)
    return final_mask


def decode(raw_predicted_labels):
    predicted_label = [label for i, label in enumerate(raw_predicted_labels)
                       if label != len(CHARS)-1 and (i == 0 or label != raw_predicted_labels[i-1])]
    return predicted_label


transform = transforms.Compose([
    transforms.Lambda(preprocessing),
    transforms.ToTensor(),
])


def test(state_dict_path, image_path, label):
    model = CRNN()
    # print(model)
    best_model = torch.load(state_dict_path)
    model.load_state_dict(best_model)
    model.eval()
    image = cv2.imread(image_path)
    image = transform(image).unsqueeze(0)
    print("Label         :", list(map(int, label)))
    log_probs = model(image)
    raw_predicted_labels = torch.argmax(log_probs, dim=2).squeeze(1)
    raw_predicted_labels = [x.item()for x in raw_predicted_labels]
    predicted_label = decode(raw_predicted_labels)
    print("Predict Label :", predicted_label)


image_path = 'Dataset/10.png'
state_dict_path = 'best-model-parameters.pt'
label = '1234'
test(state_dict_path, image_path, label)
