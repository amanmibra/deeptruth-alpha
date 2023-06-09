import time

# torch
import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader

# modal
from modal import Mount, Secret, Image, Stub, gpu

# model
from dataset import VoiceDataset
from cnn import CNNetwork

# script defaults
BATCH_SIZE = 10
EPOCHS = 100
LEARNING_RATE = 0.001

TRAIN_FILE="data/train"
TEST_FILE="data/test"
SAMPLE_RATE=48000

training_image_pip = (
    Image.debian_slim(python_version="3.9")
    .pip_install(
        "torch==2.0.0",
        "torchaudio==2.0.0",
        "pandas",
        "tqdm",
        "wandb",
    )
)

stub = Stub(
    "deeptruth-training",
    image=training_image_pip,
)

@stub.function(
    gpu=gpu.A100(memory=20),
    mounts=[
        Mount.from_local_file(local_path='dataset.py'),
        Mount.from_local_file(local_path='cnn.py'),
    ],
    timeout=EPOCHS * 200,
    secret=Secret.from_name("wandb"),
)
def train(
        model,
        train_dataloader,
        loss_fn,
        optimizer,
        origin_device="cuda",
        epochs=10,
        test_dataloader=None,
        wandb_enabled=False,
    ):
    import os

    import time
    import torch
    import wandb

    print("Begin model training...")
    begin = time.time()

    modal_device = origin_device

    # set model to cuda
    if torch.cuda.is_available() and modal_device != "cuda":
        modal_device = "cuda"
        model = model.to(modal_device)

    # metrics
    training_acc = []
    training_loss = []
    testing_acc = []
    testing_loss = []

    if wandb_enabled:
        wandb.init(project="deeptruth-training")

    for i in range(epochs):
        print(f"Epoch {i + 1}/{epochs}")
        then = time.time()

        # train model
        model, train_epoch_loss, train_epoch_acc = train_epoch.call(model, train_dataloader, loss_fn, optimizer, modal_device)

        # training metrics
        training_loss.append(train_epoch_loss/len(train_dataloader))
        training_acc.append(train_epoch_acc/len(train_dataloader))
        if wandb_enabled:
            wandb.log({'training_loss': training_loss[i], 'training_acc': training_acc[i]})        

        now = time.time()
        print("Training Loss: {:.2f}, Training Accuracy: {:.4f}, Time: {:.2f}s".format(training_loss[i], training_acc[i], now - then))

        if test_dataloader:
            # test model
            test_epoch_loss, test_epoch_acc = validate_epoch.call(model, test_dataloader, loss_fn, optimizer, modal_device)
            
            # testing metrics
            testing_loss.append(test_epoch_loss/len(test_dataloader))
            testing_acc.append(test_epoch_acc/len(test_dataloader))

            print("Testing Loss: {:.2f}, Testing Accuracy  {:.4f}".format(testing_loss[i], testing_acc[i]))

            if wandb_enabled:
                wandb.log({'testing_loss': testing_loss[i], 'testing_acc': testing_acc[i]})

        print ("-------------------------------------------------------- \n")
    
    end = time.time()
    if wandb_enabled:
        wandb.finish()
    print("-------- Finished Training --------")
    print("-------- Total Time -- {:.2f}s --------".format(end - begin))

    return model.to(origin_device)

@stub.function(
    gpu=gpu.A100(memory=20),
    mounts=[
        Mount.from_local_file(local_path='dataset.py'),
        Mount.from_local_file(local_path='cnn.py'),
    ],
    timeout=600,
)
def train_epoch(model, train_dataloader, loss_fn, optimizer, device):
    import torch
    from tqdm import tqdm

    train_loss = 0.0
    train_acc = 0.0
    total = 0.0

    model.train()

    for wav, target in tqdm(train_dataloader):
        wav, target = wav.to(device), target.to(device)

        # calculate loss
        output = model(wav)
        loss = loss_fn(output, target)

        # backprop and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # metrics
        train_loss += loss.item()
        prediction = torch.argmax(output, 1)
        train_acc += (prediction == target).sum().item()/len(prediction)
        total += 1
       
    return model, train_loss, train_acc

@stub.function(
    gpu=gpu.A100(memory=20),
    mounts=[
        Mount.from_local_file(local_path='dataset.py'),
        Mount.from_local_file(local_path='cnn.py'),
    ],
)
def validate_epoch(model, test_dataloader, loss_fn, optimizer, device):
    from tqdm import tqdm

    test_loss = 0.0
    test_acc = 0.0
    total = 0.0

    model.eval()

    with torch.no_grad():
        for wav, target in tqdm(test_dataloader, "Testing batch..."):
            wav, target = wav.to(device), target.to(device)

            output = model(wav)
            loss = loss_fn(output, target)

            optimizer.zero_grad()

            test_loss += loss.item()
            prediciton = torch.argmax(output, 1)
            test_acc += (prediciton == target).sum().item()/len(prediciton)
            total += 1
    
    return test_loss, test_acc

def save_model(model):
    now = time.strftime("%Y%m%d_%H%M%S")
    model_filename = f"models/deeptruth_{now}.pth"
    torch.save(model.state_dict(), model_filename)
    print(f"Trained deeptruth model saved at {model_filename}")

def get_device():
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    return device

@stub.local_entrypoint()
def main():
    print("Initiating model training...")
    device = get_device()

    # dataset/dataloader
    train_dataset = VoiceDataset(TRAIN_FILE, device=device)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = VoiceDataset(TEST_FILE, device=device)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # construct model
    model = CNNetwork()

    # init loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # train model
    model = train.call(
        model,
        train_dataloader,
        loss_fn,
        optimizer,
        device,
        EPOCHS,
        test_dataloader=test_dataloader,
        # wandb_enabled=True
    )

    # save model
    save_model(model)
    