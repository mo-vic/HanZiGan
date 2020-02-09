from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from PIL import Image, ImageFont, ImageDraw


class HanZi(Dataset):
    def __init__(self, font_file, size=32, from_unicode=None, to_unicode=None):
        super(HanZi, self).__init__()

        assert isinstance(from_unicode, (int, None.__class__))
        assert isinstance(to_unicode, (int, None.__class__))

        if isinstance(from_unicode, int):
            assert 0x0000 <= from_unicode < 0xffff
        if isinstance(to_unicode, int):
            assert 0x0000 < to_unicode <= 0xffff

        assert isinstance(size, int) and size >= 16

        if from_unicode is None and to_unicode is None:
            self.characters = list(range(0x3400, 0x4db5 + 1)) + list(range(0x4e00, 0x9fa5 + 1))
        elif from_unicode is None or to_unicode is None:
            raise ValueError
        else:
            self.characters = list(range(from_unicode, to_unicode + 1))

        self.font = ImageFont.truetype(font_file, size=size)

        w = []
        h = []
        for character in self.characters:
            a, b = self.font.getsize(chr(character))
            w.append(a)
            h.append(b)
        self.size = (max(w), max(h))

    def __getitem__(self, item):
        character = chr(self.characters[item])
        image = Image.new('L', self.size)
        draw = ImageDraw.Draw(image)
        draw.text((0, 0), character, 255, font=self.font)
        del draw

        return ToTensor()(image)

    def __len__(self):
        return len(self.characters)

    def get_size(self):
        return self.size[1], self.size[0]


def build_dataloader(batch_size, num_workers, use_gpu, font_file, size=32, from_unicode=None, to_unicode=None):
    if isinstance(num_workers, int):
        pass
    else:
        print("'num_workers' should have type of int, found {0}".format(type(num_workers)))
        raise ValueError
    if isinstance(use_gpu, bool):
        pass
    else:
        print("'use_gpu' should have type of bool, found {0}".format(type(use_gpu)))
        raise ValueError

    dataset = HanZi(font_file, size, from_unicode, to_unicode)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=use_gpu,
                            drop_last=True)

    return dataloader, dataset.get_size()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=16, help="Font size.")
    parser.add_argument("--from_unicode", type=int, help="Starting point of the unicode.")
    parser.add_argument("--to_unicode", type=int, help="Ending point of the unicode.")
    parser.add_argument("--font", type=str, required=True, help="Path to the font file.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers.")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU or not?")

    args = parser.parse_args()

    dataloader, _ = build_dataloader(args.batch_size, args.num_workers, args.use_gpu, args.font, args.size,
                                     args.from_unicode, args.to_unicode)

    for images in dataloader:
        print("Image size: {0}, image data type: {1}.".format(images.size(), images.dtype))
