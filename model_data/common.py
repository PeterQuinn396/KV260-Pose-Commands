
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # enable gpu processing


def test_vitis_compatible(model, input_shape = (1,3,224,224)):
    """
    Run the torch jit check to make sure the model_data is quantizable
    by Vitis
    :param model:
    :return:
    """
    model.to(device)
    model.eval()
    test_input = torch.rand(input_shape).to(device)
    try:
        tr_func = torch.jit.trace(model, test_input)
        print("Model passes jit trace test")
        return True
    except Exception as e:
        print(e)
        print("Model fails test")
        return False
   

def train():
    pass


def test(model: torch.nn.Module, test_loader: torch.utils.data.DataLoader, loss_fn, max_samples=100):

    model.eval()

    running_corrects = 0
    running_loss = 0

    count = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            lb_inds = torch.argmax(labels, 1)
            loss = loss_fn(outputs, lb_inds)

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == lb_inds)

            # put a max on the number of samples tested
            count += len(inputs)
            if count > max_samples:
                break

    epoch_loss = running_loss / count
    epoch_acc = running_corrects.double() / count

    return epoch_acc, epoch_loss, count