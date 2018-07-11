import torch
import torchvision
from torchvision import transforms
from .sampler import RandSubClassSampler


def MNIST(batch_sz, num_workers=2):
    normalize = transforms.Normalize(mean=(0.1307,), std=(0.3081,))

    train_dataset = torchvision.datasets.MNIST(
        root='data',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize
        ]))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_sz, shuffle=True, num_workers=num_workers)
    train_loader.num_classes = 10

    eval_dataset = torchvision.datasets.MNIST('data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        normalize
    ]))
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_sz, shuffle=False, num_workers=num_workers)
    eval_loader.num_classes = 10

    return train_loader, eval_loader

def CIFAR10(batch_sz, num_workers=2):
    normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])

    train_dataset = torchvision.datasets.CIFAR10(
        root='data',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_sz, shuffle=True, num_workers=num_workers)
    train_loader.num_classes = 10

    test_dataset = torchvision.datasets.CIFAR10(
        root='data',
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    eval_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_sz, shuffle=False, num_workers=num_workers)
    eval_loader.num_classes = 10

    return train_loader, eval_loader


def CIFAR100(batch_sz, num_workers=2):
    normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])

    train_dataset = torchvision.datasets.CIFAR100(
        root='data',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_sz, shuffle=True, num_workers=num_workers)
    train_loader.num_classes = 100

    test_dataset = torchvision.datasets.CIFAR100(
        root='data',
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    eval_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_sz, shuffle=False, num_workers=num_workers)
    eval_loader.num_classes = 100

    return train_loader, eval_loader

def Omniglot(batch_sz, num_workers=2):
    # This dataset is only for training the Similarity Prediction Network on Omniglot background set
    binary_flip = transforms.Lambda(lambda x: 1 - x)
    normalize = transforms.Normalize((0.086,), (0.235,))
    train_dataset = torchvision.datasets.Omniglot(
        root='data', download=True, background=True,
        transform=transforms.Compose(
           [transforms.RandomResizedCrop(32, (0.85, 1.)),
            transforms.ToTensor(),
            binary_flip,
            normalize]
        ))
    train_length = len(train_dataset)
    train_imgid2cid = [train_dataset[i][1] for i in range(train_length)]  # train_dataset[i] returns (img, cid)
    # Randomly select 20 characters from 964. By default setting (batch_sz=100), each character has 5 images in a mini-batch.
    train_sampler = RandSubClassSampler(
        inds=range(train_length),
        labels=train_imgid2cid,
        cls_per_batch=20,
        batch_size=batch_sz,
        num_batch=train_length//batch_sz)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_sz, shuffle=False,
                                               num_workers=num_workers, sampler=train_sampler)
    train_loader.num_classes = 964

    test_dataset = torchvision.datasets.Omniglot(
        root='data', download=True, background=False,
        transform=transforms.Compose(
          [transforms.Resize(32),
           transforms.ToTensor(),
           binary_flip,
           normalize]
        ))
    eval_length = len(test_dataset)
    eval_imgid2cid = [test_dataset[i][1] for i in range(eval_length)]
    eval_sampler = RandSubClassSampler(
        inds=range(eval_length),
        labels=eval_imgid2cid,
        cls_per_batch=20,
        batch_size=batch_sz,
        num_batch=eval_length // batch_sz)
    eval_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_sz, shuffle=False,
                                              num_workers=num_workers, sampler=eval_sampler)
    eval_loader.num_classes = 659

    return train_loader, eval_loader


def omniglot_alphabet_func(alphabet, background):
    def create_alphabet_dataset(batch_sz, num_workers=2):
        # This dataset is only for unsupervised clustering
        # train_dataset (with data augmentation) is used during the optimization of clustering criteria
        # test_dataset (without data augmentation) is used after the clustering is converged

        binary_flip = transforms.Lambda(lambda x: 1 - x)
        normalize = transforms.Normalize((0.086,), (0.235,))

        train_dataset = torchvision.datasets.Omniglot(
            root='data', download=True, background=background,
            transform=transforms.Compose(
               [transforms.RandomResizedCrop(32, (0.85, 1.)),
                transforms.ToTensor(),
                binary_flip,
                normalize]
            ))

        # Following part dependents on the internal implementation of official Omniglot dataset loader
        # Only use the images which has alphabet-name in their path name (_characters[cid])
        valid_flat_character_images = [(imgname,cid) for imgname,cid in train_dataset._flat_character_images if alphabet in train_dataset._characters[cid]]
        ndata = len(valid_flat_character_images)  # The number of data after filtering
        train_imgid2cid = [valid_flat_character_images[i][1] for i in range(ndata)]  # The tuple (valid_flat_character_images[i]) are (img, cid)
        cid_set = set(train_imgid2cid)  # The labels are not 0..c-1 here.
        cid2ncid = {cid:ncid for ncid,cid in enumerate(cid_set)}  # Create the mapping table for New cid (ncid)
        valid_characters = {cid2ncid[cid]:train_dataset._characters[cid] for cid in cid_set}
        for i in range(ndata):  # Convert the labels to make sure it has the value {0..c-1}
            valid_flat_character_images[i] = (valid_flat_character_images[i][0],cid2ncid[valid_flat_character_images[i][1]])

        # Apply surgery to the dataset
        train_dataset._flat_character_images = valid_flat_character_images
        train_dataset._characters = valid_characters

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_sz, shuffle=True,
                                                   num_workers=num_workers)
        train_loader.num_classes = len(cid_set)

        test_dataset = torchvision.datasets.Omniglot(
            root='data', download=True, background=background,
            transform=transforms.Compose(
              [transforms.Resize(32),
               transforms.ToTensor(),
               binary_flip,
               normalize]
            ))

        # Apply surgery to the dataset
        test_dataset._flat_character_images = valid_flat_character_images  # Set the new list to the dataset
        test_dataset._characters = valid_characters

        eval_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_sz, shuffle=False,
                                                  num_workers=num_workers)
        eval_loader.num_classes = train_loader.num_classes

        print('=> Alphabet %s has %d characters and %d images.'%(alphabet, train_loader.num_classes, len(train_dataset)))
        return train_loader, eval_loader
    return create_alphabet_dataset

omniglot_evaluation_alphabets_mapping = {
    'Malayalam':'Malayalam',
     'Kannada':'Kannada',
     'Syriac':'Syriac_(Serto)',
     'Atemayar_Qelisayer':'Atemayar_Qelisayer',
     'Gurmukhi':'Gurmukhi',
     'Old_Church_Slavonic':'Old_Church_Slavonic_(Cyrillic)',
     'Manipuri':'Manipuri',
     'Atlantean':'Atlantean',
     'Sylheti':'Sylheti',
     'Mongolian':'Mongolian',
     'Aurek':'Aurek-Besh',
     'Angelic':'Angelic',
     'ULOG':'ULOG',
     'Oriya':'Oriya',
     'Avesta':'Avesta',
     'Tibetan':'Tibetan',
     'Tengwar':'Tengwar',
     'Keble':'Keble',
     'Ge_ez':'Ge_ez',
     'Glagolitic':'Glagolitic'
}

# Create the functions to access the individual alphabet dataset in Omniglot
for funcName, alphabetStr in omniglot_evaluation_alphabets_mapping.items():
    locals()['Omniglot_eval_' + funcName] = omniglot_alphabet_func(alphabet=alphabetStr, background=False)
