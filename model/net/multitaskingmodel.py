# encoding:utf-8
import torch.nn as nn
import timm
import torch

class BinaryClassificationHead(nn.Module):
    def __init__(self, input_num=1024, drop_rate=0, num_classes=2):
        super(BinaryClassificationHead, self).__init__()
        self.head = (nn.Sequential(
            nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity(),
            nn.Linear(input_num, num_classes)  # 512->1
        ))
    def forward(self, x):
        out = self.head(x)
        return out

class MultiClassificationHead(nn.Module):
    def __init__(self, input_num=1024, drop_rate=0, num_classes=32):
        super(MultiClassificationHead, self).__init__()
        self.head = (nn.Sequential(
            nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity(),
            nn.Linear(input_num, num_classes)  # 512->1
        ))
    def forward(self, x):
        out = self.head(x)
        return out

class MultitaskingModel(nn.Module):
    def __init__(self, model_name="efficientnet_b4", drop_rate=0, class_nums = [2, 32], isFreeze = False):
        super(MultitaskingModel, self).__init__()
        self.model = timm.create_model(model_name, pretrained=False)
        self.model.reset_classifier(0)# remove classification layer
        x = torch.randn([1, 3, 512, 512])
        out = self.model(x)
        input_channel_head = out.shape[-1]
        del x, out

        if isFreeze:
            for params in self.model.features.parameters():
                params.requires_grad = False

        self.task_nums = len(class_nums)
        if self.task_nums == 2:
            self.head_binary = BinaryClassificationHead(input_num=input_channel_head, drop_rate=drop_rate, num_classes = class_nums[0])
            self.head_multiclass = MultiClassificationHead(input_num=input_channel_head, drop_rate=drop_rate, num_classes = class_nums[1])
        else:
            self.head_binary = BinaryClassificationHead(input_num=input_channel_head, drop_rate=drop_rate, num_classes=class_nums[0])

    def forward(self, x):
        out = self.model(x)
        if self.task_nums == 2:
            out_binary = self.head_binary(out)
            out_multiclass = self.head_multiclass(out)
            return out_binary, out_multiclass
        else:
            out_binary = self.head_binary(out)
            return out_binary, None

