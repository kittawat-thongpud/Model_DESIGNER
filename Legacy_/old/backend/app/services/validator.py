"""
Validation service — runs a model against a test dataset and returns metrics.
"""
from __future__ import annotations
from ..schemas.model_schema import ValidateResponse
from .. import logging_service as logger


def validate_model(model_id: str, dataset: str = "mnist", batch_size: int = 64) -> ValidateResponse:
    """Load a trained model and evaluate on the test set."""
    import torch
    import torch.nn as nn
    from torchvision import datasets, transforms

    from ..storage import load_model, WEIGHTS_DIR
    from .codegen import generate_code

    # Build model from graph
    _, graph = load_model(model_id)
    class_name, code = generate_code(graph)

    namespace: dict = {}
    exec(code, namespace)
    ModelClass = namespace[class_name]
    model = ModelClass()

    # Load weights
    weight_path = WEIGHTS_DIR / f"{model_id}.pt"
    if not weight_path.exists():
        raise FileNotFoundError(f"No trained weights for model '{model_id}'. Train first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(str(weight_path), map_location=device))
    model = model.to(device)
    model.eval()

    # Dataset
    dataset_name = dataset.lower()
    if dataset_name == "mnist":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        test_ds = datasets.MNIST("./data", train=False, download=True, transform=transform)
        class_names = [str(i) for i in range(10)]
    elif dataset_name == "cifar10":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        test_ds = datasets.CIFAR10("./data", train=False, download=True, transform=transform)
        class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0
    per_class_correct: dict[str, int] = {c: 0 for c in class_names}
    per_class_total: dict[str, int] = {c: 0 for c in class_names}

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            for i in range(targets.size(0)):
                label = class_names[targets[i].item()]
                per_class_total[label] += 1
                if predicted[i] == targets[i]:
                    per_class_correct[label] += 1

    accuracy = 100.0 * correct / total
    avg_loss = total_loss / len(test_loader)
    per_class_acc = {
        c: round(100.0 * per_class_correct[c] / max(per_class_total[c], 1), 2)
        for c in class_names
    }

    logger.log("training", "INFO", f"Validation complete: acc={accuracy:.2f}% loss={avg_loss:.4f}", {"model_id": model_id})

    return ValidateResponse(
        model_id=model_id,
        accuracy=round(accuracy, 2),
        loss=round(avg_loss, 4),
        num_samples=total,
        per_class_accuracy=per_class_acc,
    )
