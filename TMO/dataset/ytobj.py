from .transforms import *
import os
from PIL import Image
import torchvision as tv


class TestYTOBJ(torch.utils.data.Dataset):
    def __init__(self, root):
        classes = sorted(os.listdir(os.path.join(root, 'JPEGImages')))
        self.seqs = []
        for cls in classes:
            seqs = sorted(os.listdir(os.path.join(root, 'JPEGImages', cls)))
            for seq in seqs:
                self.seqs.append(os.path.join(root, 'JPEGImages', cls, seq))
        self.to_tensor = tv.transforms.ToTensor()

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        img_list = sorted(os.listdir(os.path.join(self.seqs[idx])))
        flow_list = sorted(os.listdir(os.path.join(self.seqs[idx].replace('JPEGImages', 'JPEGFlows'))))
        mask_list = sorted(os.listdir(os.path.join(self.seqs[idx].replace('JPEGImages', 'Annotations'))))

        # generate testing snippets
        imgs = []
        flows = []
        masks = []
        for i in range(len(img_list)):
            img = Image.open(os.path.join(self.seqs[idx], img_list[i])).convert('RGB')
            img = img.resize((384, 384), Image.BICUBIC)
            imgs.append(self.to_tensor(img))
        for i in range(len(flow_list)):
            flow = Image.open(os.path.join(self.seqs[idx].replace('JPEGImages', 'JPEGFlows'), flow_list[i])).convert('RGB')
            flow = flow.resize((384, 384), Image.BICUBIC)
            flows.append(self.to_tensor(flow))
        for i in range(len(mask_list)):
            mask = Image.open(os.path.join(self.seqs[idx].replace('JPEGImages', 'Annotations'), mask_list[i])).convert('L')
            mask = mask.resize((384, 384), Image.BICUBIC)
            masks.append(self.to_tensor(mask))

        # gather all frames
        imgs = torch.stack(imgs, dim=0)
        flows = torch.stack(flows, dim=0)
        masks = torch.stack(masks, dim=0)
        masks = (masks > 0.5).long()
        return {'imgs': imgs, 'flows': flows, 'masks': masks, 'files': img_list, 'path': self.seqs[idx]}
