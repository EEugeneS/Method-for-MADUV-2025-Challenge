import os
import random
import torch
import torchaudio
import numpy as np
import torch.nn as nn
from tqdm import tqdm

from beats_wrapper import beats_model
from collections import defaultdict
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from sklearn.metrics import recall_score
from data_pipeline import AudioFeatureDataset

def train_models(output_path,nfft):
    # Set the seeds
    seed = 619
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Set the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = beats_model()
    model = torch.nn.DataParallel(model, device_ids=[0,1,2])
    model = model.to(device)
    
    # Set the loss function and optimiser
    criterion = nn.CrossEntropyLoss()
    num_warmup_steps = 1000
    initial_lr =2e-5
    target_lr = 5e-4
    
    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=0.01)
    def warmup_lambda(current_step):
        if current_step < num_warmup_steps:
            return initial_lr + (target_lr - initial_lr) * (current_step / num_warmup_steps)
        return target_lr
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: warmup_lambda(step) / initial_lr)

    # Set the training hyperparameters
    batch_size = 250
    training_epoch = 200
    validation_epoch = 2
    version = 1

    # Set the tensorboard writer and log file
    writer = SummaryWriter(f'Beats_sample_rate_300KHz_Spec_nfft_{nfft}_bs_{batch_size}_epoch_{training_epoch}_v{version}_full/')
    log_path = f'Beats_sample_rate_300KHz_without_Spec_nfft_{nfft}_bs_{batch_size}_epoch_{training_epoch}_v{version}_log_full.txt'
    model_fold = f'Beats_sample_rate_300KHz_Spec_nfft_{nfft}_bs_{batch_size}_epoch_{training_epoch}_v{version}_model_full/'

    # Create datasets and dataloaders for train and validation sets
    train_path = os.path.join(output_path,'train')
    valid_path =os.path.join(output_path,'valid')
    train_dataset = AudioFeatureDataset(audio_folder=train_path)
    valid_dataset = AudioFeatureDataset(audio_folder=valid_path)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    
    # Training phase
    best_seg_uar = 0
    for epoch in range(training_epoch):
        
        # Get the training loss
        train_loss = train(net=model, data_loader=train_loader,device=device,optimiser=optimizer,criterion=criterion,epoch=epoch, scheduler=warmup_scheduler)
        
        # Print loss and write the log
        print(f"-->\tEpoch {epoch+1:04}:\tTraining Loss: {train_loss:.3f}")
        writer.add_scalar('loss/training', train_loss, epoch+1)
        with open(log_path, 'a') as lf:
            lf.write(f"-->\tEpoch {epoch+1:04}:\tTraining Loss: {train_loss:.3f}\n")

        # Validation phase
        if (epoch + 1) % validation_epoch == 0 or (epoch + 1) % validation_epoch == 1:
            
            # Get the validation loss, predicted probabilities, true labels, and filenames
            valid_loss, predictions, labels, filenames = validate(net=model, data_loader=valid_loader,device=device,criterion=criterion,epoch=epoch)
            pred_uar = recall_score(labels.detach().cpu(), predictions.detach().cpu(), average='macro')
            
            # Print loss and segment-level metric, and write the log
            print(f"-->\tEpoch {epoch+1:04}:\tValidation Loss: {valid_loss:.3f}")
            print(f"-->\tEpoch {epoch+1:04}:\tSegment UAR (Pred): {pred_uar:.3f}")
            writer.add_scalar('loss/validation', valid_loss, epoch+1)
            writer.add_scalar('seg_metric/pred_uar', pred_uar, epoch+1)
            
            with open(log_path, 'a') as lf:
                lf.write(f"-->\tEpoch {epoch+1:04}:\tValidation Loss: {valid_loss:.3f}\n")
                lf.write(f"-->\tEpoch {epoch+1:04}:\tSegment UAR (Pred): {pred_uar:.3f}\n")
            
            max_uar = pred_uar
            
            # Save the model if the segment-level UAR is improved
            if max_uar > best_seg_uar:
                best_seg_uar = max_uar
                model_path = os.path.join(model_fold, f"epoch_{epoch+1}_best_segment_{best_seg_uar:.3f}.pth")
                
                if not os.path.exists(model_fold):
                    os.makedirs(model_fold)

                # Save model weights    
                torch.save(model.state_dict(), model_path)
                print(f"Model saved at {model_path}")
                with open(log_path, 'a') as lf:
                    lf.write(f"Model saved at {model_path}\n")

            # Calculate the subject-level UAR with majority vote
            grouped_predictions = defaultdict(list)
            grouped_labels = defaultdict(list)
            
            epoch_labels = predictions.detach().cpu()
            
            for pred, label, filename in zip(epoch_labels, labels, filenames):
                grouped_predictions[filename].append(pred)
                grouped_labels[filename].append(label)

            final_predictions = []
            final_labels = []
            for filename in grouped_predictions:
                # Majority vote for predictions and labels
                majority_pred = max(set(grouped_predictions[filename]), key=grouped_predictions[filename].count)
                majority_label = max(set(grouped_labels[filename]), key=grouped_labels[filename].count)
                
                final_predictions.append(majority_pred)
                final_labels.append(majority_label.detach().cpu())

            uar = recall_score(final_labels, final_predictions, average='macro')

            # Print the subject-level metric and write the log
            print(f"-->\tEpoch {epoch+1:04}:\tSubject UAR: {uar:.3f}")
            writer.add_scalar('sam_metric/uar', uar, epoch+1)
            with open(log_path, 'a') as lf:
                lf.write(f"-->\tEpoch {epoch+1:04}:\tSubject UAR: {uar:.3f}\n")
    
    
def train(net: beats_model, data_loader: DataLoader,device,optimiser,criterion,epoch,scheduler=None) -> float:
    """
    Training process of the model.

    Args:
        net (beats_model): The model to be trained.
        data_loader (DataLoader): DataLoader for the training set.

    Returns:
        float: The average loss of one training epoch.
    """
    
    net.train()
    total_loss = 0.

    
    for k, (data, y_true, _) in enumerate(data_loader):
        
        x = torch.Tensor(data).to(device)
        y_true = y_true.to(torch.int64).to(device)
        
        optimiser.zero_grad()
        output = net(x)

        loss = criterion(output, y_true)
        loss.backward()
        
        optimiser.step()
        if scheduler is not None:
            scheduler.step()
        
        print(f"Epoch {epoch+1:04} - Batch {k+1}/{len(data_loader)}\tLoss: {loss.item():.3f}")

        total_loss += loss.item()

    epoch_loss = total_loss / len(data_loader)
    return epoch_loss


@torch.no_grad()
def validate(net: beats_model, data_loader: DataLoader,device,criterion,epoch) -> tuple:
    """
    Validation process of the model.

    Args:
        net (beats_model): The model to be validated.
        data_loader (DataLoader): DataLoader for the validation set.

    Returns:
        tuple: A tuple containing the validation loss, the predicted probabilities, the true labels, and the filenames.
    """

    net.eval()
    total_loss = 0.
    all_probabilities = []
    all_labels = []
    all_filenames = []
    all_predicts = []


    for k, (data, y_true, y_name) in enumerate(data_loader):
        
        y_true = y_true.to(torch.int64).to(device)
        
        x = torch.Tensor(data).to(device)

        output = net(x)
        loss = criterion(output, y_true)
        
        print(f"vEpoch {epoch+1:04} - Batch {k+1}/{len(data_loader)}\tLoss: {loss.item():.3f}")

        total_loss += loss.item()
        
        y_prob = torch.softmax(output, dim=1)
        y_pred = torch.argmax(y_prob, dim=1)

        all_probabilities.append(y_prob.squeeze())
        all_labels.append(y_true)
        all_predicts.append(y_pred)
        all_filenames = all_filenames + list(y_name)

    epoch_loss = total_loss / len(data_loader)

    all_predicts = torch.cat(all_predicts, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return epoch_loss, all_predicts, all_labels, all_filenames

def extract_spectrogram(audio_file: str, save_file: str, normalized: bool = False, nfft: int = 0) -> None:
    """
    Extract Features from an audio file and save it as a .npy file.

    Args:
        audio_file (str): Path to the input audio file.
        save_file (str): Path to save the extracted features.
        n_fft (int): Number of FFT components.
        normalized (bool): Normalize Option.
    """    
    waveform, sample_rate = torchaudio.load(audio_file)
    
    # Calculate STFT
    n_fft = nfft
    win_length = n_fft
    hop_length = win_length//2
    transform = torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=win_length//2, win_length=n_fft, normalized=normalized, power=2)
    spectrogram = transform(waveform)
    spectrogram = spectrogram.transpose(-1,-2)
    data = spectrogram
    new_spec = []
    _,T,F = data.shape
    
    # Split the full spectrogram into 128 sub-band images
    split_size = F // 127
    remainder = F % 127
    for m in range(0, F ,split_size):
        if m+split_size>F:
            chunk = data[:,:, -split_size:][0]
            chunk = torch.mean(chunk,dim=-1)
            print(chunk.shape)
            new_spec.append(chunk)
        else:
            chunk = data[:,:, m:m+split_size][0]
            chunk = torch.mean(chunk,dim=-1)
            new_spec.append(chunk)
    new_spec = torch.vstack(new_spec)
    new_spec = new_spec.transpose(1,0)

    np.save(save_file, new_spec)
    print(f"Saved full spectrogram to {save_file}, shape: {new_spec.shape}")
    return spectrogram

def process_audio_files(input_folder: str, output_folder: str, feature_set: str, nfft: int) -> None:
    """
    Process audio files in the input folder and save the extracted features in the output folder.

    Args:
        input_folder (str): Path to the directory containing input audio files.
        output_folder (str): Path to the directory saving the extracted features.
        feature_set (str): The selected feature set.
        nfft (int): Number of FFT components.
    """

    # Create the output directory if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each audio file in the input directory
    spectrograms = []
    for audio_file in tqdm(os.listdir(input_folder)):
        if audio_file.endswith(".wav"):
            audio_path = os.path.join(input_folder, audio_file)
            output_path = os.path.join(output_folder, f"{os.path.splitext(audio_file)[0]}.npy")
            fbank = extract_spectrogram(audio_path, output_path, nfft=nfft)
            spectrograms.append(fbank)
    return spectrograms


def save_features(nfft,input_path, output_path, feature_set):
    
    
    # Create the output directory if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Process all audio files in the input directory
    for folder in ['train','valid','test']:
        if feature_set not in ['full']:
            print("feature_set can only be 'full'.")
            break
        else:
            process_audio_files(os.path.join(input_path, folder),
                                os.path.join(output_path, folder),
                                feature_set,nfft)                
    return output_path


if __name__ == '__main__':
      
    nffts = [300000]
    
    for nfft in nffts:
        # Extract feature
        input_path = './trimmed_data'
        output_path = f'./USV_Feature_full_Spec_nfft_{nfft}_T_128'
        feature_set = 'full'
        output_path = save_features(nfft, input_path, output_path, feature_set)
        print('Feature Extraction Done!')
        # Train model
        train_models(output_path,nfft)
        print('Model Training Done!')
