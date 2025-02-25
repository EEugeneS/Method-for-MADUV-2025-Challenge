import glob
import torch

from torch.utils.data import DataLoader
from data_pipeline import AudioFeatureDataset
from beats_wrapper import beats_model

@torch.no_grad()
def test(net: beats_model, data_loader: DataLoader) -> torch.Tensor:
    """
    Generate predicted probabilities of the test set.

    Args:
        net (beats_model): The model to be tested.
        data_loader (DataLoader): DataLoader for the test set.

    Returns:
        tuple: A tuple containing the predicted probabilities and filenames.
    """
    net.eval()
    all_predicts = []

    with torch.no_grad():
        for i, data in enumerate(data_loader):
            
            x = torch.Tensor(data).to(device)
            output = net.infer(x)
            y_prob = torch.softmax(output, dim=1)
            y_pred = torch.argmax(y_prob, dim=1)
            
            all_predicts.append(y_pred.squeeze())

    all_predicts = torch.cat(all_predicts, dim=0)

    return all_predicts


if __name__ == '__main__':
    # Set the device
    device = torch.device('cuda:6' if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create dataset and dataloader for test set
    feature_folder="../USV_Feature_full_Spec_nfft_300000_T_128/test"
    test_dataset = AudioFeatureDataset(audio_folder=feature_folder,is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=250, shuffle=False)

    # Load the model
    model = beats_model()
    
    segment_model = glob.glob('./Beats_sample_rate_300KHz_Spec_nfft_300000_bs_250_epoch_200_v6_model_full/epoch_19_best_segment_0.729.pth')
    
    checkpoint = torch.load(segment_model[0])
    new_state_dict = {}
    for key, value in checkpoint.items():
        new_key = key[7:] if key.startswith('module.') else key  # Remove 'module.' prefix
        new_state_dict[new_key] = value
    
    model.load_state_dict(new_state_dict)
    model = model.to(device)

    # Test phase
    predicts = test(net=model, data_loader=test_loader)
    predicts = predicts.detach().cpu().numpy().astype(int)
    results = ",".join(map(str, predicts))
    with open(f"./test_result.txt", 'w') as pf:
        pf.write(results)
    
    print(f"The segment-level prediction has been saved to ./test_result.txt")
