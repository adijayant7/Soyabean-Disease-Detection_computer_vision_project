from torchvision import transforms

train_transforms= transforms.Compose(
    [transforms.RandomResizedCrop(size=224,scale=(0.7,1.0))
     ,
     transforms.RandomHorizontalFlip(p=0.5),
     transforms.RandomRotation(degrees=15),
     transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2),
     transforms.ToTensor(),
     transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
])

test_transforms=transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize( 
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
])
val_test_transforms = transforms.Compose([

    transforms.Resize(256),

    transforms.CenterCrop(224),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])