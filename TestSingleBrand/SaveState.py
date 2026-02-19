torch.save({
    "model_state": net.state_dict(),
    "class_names": class_names
}, "/content/drive/MyDrive/honda_resnet18.pth")
