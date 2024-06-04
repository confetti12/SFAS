if imgs[1].size() == torch.Size([2, 80, 80, 3]):
            imgs[1] = imgs[1].permute(0, 3, 1, 2)
        else:
            pass