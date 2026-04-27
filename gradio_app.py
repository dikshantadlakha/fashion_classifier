import os
import time
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import gradio as gr
from PIL import Image
from torchvision import models, transforms

BASE_DIR = os.path.dirname(__file__)

CSV_PATH = os.path.join(BASE_DIR, "dataset", "fmnist_small.csv")

ANN_MODEL_PATH = os.path.join(BASE_DIR, "pt_file", "ann_model.pt")
OPTIMIZED_ANN_MODEL_PATH = os.path.join(BASE_DIR, "pt_file", "ann_optimised_model (1).pt")
CNN_MODEL_PATH = os.path.join(BASE_DIR, "pt_file", "cnn_model (6).pt")
VGG16_MODEL_PATH = os.path.join(BASE_DIR, "pt_file", "vgg16_model.pt")

HYPERPARAM_ANN_MODEL_CANDIDATES = [
    os.path.join(BASE_DIR, "pt_file", "ann_hyperparametrised_model.pt"),
    os.path.join(BASE_DIR, "pt_file", "ann_hyperparameterised_model.pt"),
    os.path.join(BASE_DIR, "pt_file", "ann_hyperparameterimised_model.pt"),
]

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


class BasicANN(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.model(x)


class HyperparameterizedANN(nn.Module):
    def __init__(self, num_features, hidden_layer_sizes, dropout_rate=0.4):
        super().__init__()

        layers = []
        input_dim = num_features

        for hidden_size in hidden_layer_sizes:
            layers.extend(
                [
                    nn.Linear(input_dim, hidden_size),
                    nn.BatchNorm1d(hidden_size),
                    nn.ReLU(),
                    nn.Dropout(p=dropout_rate),
                ]
            )
            input_dim = hidden_size

        layers.append(nn.Linear(input_dim, 10))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class OptimizedANN(nn.Module):
    def __init__(self, num_features):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.model(x)


class CNNModel(nn.Module):
    def __init__(self, input_features):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(input_features, 32, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def build_vgg16_model():
    model = models.vgg16(weights=None)

    for param in model.features.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Linear(25088, 1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 10),
    )

    return model


def resolve_existing_path(path_candidates):
    for path in path_candidates:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(f"None of these model files were found: {path_candidates}")


VGG16_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def row_to_image_and_tensor(row):
    label = int(row.iloc[0])
    pixels = row.iloc[1:].to_numpy(dtype="uint8")

    image = Image.fromarray(pixels.reshape(28, 28), mode="L")

    flat_tensor = torch.tensor(pixels, dtype=torch.float32).div(255.0).unsqueeze(0)
    image_tensor = flat_tensor.view(1, 1, 28, 28)

    rgb_pixels = np.stack([pixels.reshape(28, 28)] * 3, axis=-1)
    rgb_image = Image.fromarray(rgb_pixels)

    vgg16_tensor = VGG16_TRANSFORM(rgb_image).unsqueeze(0)

    return label, image, flat_tensor, image_tensor, vgg16_tensor


def load_class_samples():
    class_samples = []

    for class_index, class_name in enumerate(CLASS_NAMES):
        row = dataset_df[dataset_df.iloc[:, 0] == class_index].sample(n=1).iloc[0]

        label, image, flat_tensor, image_tensor, vgg16_tensor = row_to_image_and_tensor(row)

        class_samples.append(
            {
                "label": label,
                "class_name": class_name,
                "image": image,
                "flat_tensor": flat_tensor,
                "image_tensor": image_tensor,
                "vgg16_tensor": vgg16_tensor,
                "gallery_caption": f"{class_index}: {class_name}",
            }
        )

    return class_samples


def load_model(model_class, model_path, model_input_features=784):
    state_dict = torch.load(model_path, map_location="cpu")

    model = model_class(model_input_features)
    model.load_state_dict(state_dict)
    model.eval()

    return model


def build_hyperparameterized_model(num_features, state_dict):
    linear_layer_indices = sorted(
        {
            int(key.split(".")[1])
            for key in state_dict
            if key.startswith("model.")
            and key.endswith(".weight")
            and state_dict[key].ndim == 2
        }
    )

    hidden_layer_sizes = [
        int(state_dict[f"model.{index}.weight"].shape[0])
        for index in linear_layer_indices[:-1]
    ]

    return HyperparameterizedANN(
        num_features,
        hidden_layer_sizes=hidden_layer_sizes,
    )


def load_hyperparameterized_model(model_path):
    state_dict = torch.load(model_path, map_location="cpu")

    model = build_hyperparameterized_model(784, state_dict)
    model.load_state_dict(state_dict)
    model.eval()

    return model


def load_vgg16_model(model_path):
    state_dict = torch.load(model_path, map_location="cpu")

    model = build_vgg16_model()
    model.load_state_dict(state_dict)
    model.eval()

    return model


def run_inference(model, tensor):
    start_time = time.perf_counter()

    with torch.no_grad():
        output = model(tensor)
        probabilities = torch.softmax(output, dim=1)

    inference_time_ms = (time.perf_counter() - start_time) * 1000

    top_probs, top_indices = torch.topk(probabilities, k=3, dim=1)

    predicted_index = int(top_indices[0, 0].item())
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(top_probs[0, 0].item()) * 100

    top_3_predictions = [
        f"{CLASS_NAMES[int(index.item())]}: {float(prob.item()) * 100:.2f}%"
        for prob, index in zip(top_probs[0], top_indices[0])
    ]

    return predicted_class, confidence, top_3_predictions, inference_time_ms


def format_model_result(model, tensor):
    predicted_class, confidence, top_3_predictions, inference_time_ms = run_inference(
        model,
        tensor,
    )

    top_3_text = "\n".join(
        f"{rank}. {prediction}"
        for rank, prediction in enumerate(top_3_predictions, start=1)
    )

    return (
        predicted_class,
        f"{confidence:.2f}%",
        top_3_text,
        f"{inference_time_ms:.2f} ms",
    )


def upscale_display_image(image, size=224):
    return image.resize((size, size), resample=Image.Resampling.NEAREST)


dataset_df = pd.read_csv(CSV_PATH)


MODEL_SPECS = [
    {
        "key": "ann",
        "title": "ANN",
        "model_class": BasicANN,
        "model_path": ANN_MODEL_PATH,
        "input_key": "flat_tensor",
    },
    {
        "key": "ann_optimized",
        "title": "Optimized ANN",
        "model_class": OptimizedANN,
        "model_path": OPTIMIZED_ANN_MODEL_PATH,
        "input_key": "flat_tensor",
    },
    {
        "key": "ann_hyperparameterized",
        "title": "Hyperparameterized ANN",
        "loader": load_hyperparameterized_model,
        "model_path": resolve_existing_path(HYPERPARAM_ANN_MODEL_CANDIDATES),
        "input_key": "flat_tensor",
    },
    {
        "key": "cnn",
        "title": "CNN",
        "model_class": CNNModel,
        "model_path": CNN_MODEL_PATH,
        "model_input_features": 1,
        "input_key": "image_tensor",
    },
    {
        "key": "vgg16_transfer_learning",
        "title": "VGG16 Transfer Learning",
        "loader": load_vgg16_model,
        "model_path": VGG16_MODEL_PATH,
        "input_key": "vgg16_tensor",
    },
]


MODELS = {
    spec["key"]: {
        **spec,
        "model": (
            spec["loader"](spec["model_path"])
            if "loader" in spec
            else load_model(
                spec["model_class"],
                spec["model_path"],
                spec.get("model_input_features", 784),
            )
        ),
    }
    for spec in MODEL_SPECS
}


def format_prediction_result(sample_index, current_samples):
    sample = current_samples[sample_index]

    true_class = f'{sample["class_name"]} ({sample["label"]})'

    results = [
        upscale_display_image(sample["image"]),
        true_class,
    ]

    for spec in MODEL_SPECS:
        results.extend(
            format_model_result(
                MODELS[spec["key"]]["model"],
                sample[spec["input_key"]],
            )
        )

    return tuple(results)


def build_gallery_items(current_samples):
    return [
        (sample["image"], sample["gallery_caption"])
        for sample in current_samples
    ]


def initialize_session():
    current_samples = load_class_samples()
    return current_samples, gr.Gallery(value=build_gallery_items(current_samples), selected_index=0), *format_prediction_result(
        0, current_samples
    )


def on_gallery_select(current_samples, evt: gr.SelectData):
    return format_prediction_result(evt.index, current_samples)


with gr.Blocks() as demo:
    gr.Markdown("# ANN Fashion MNIST Classification")
    gr.Markdown(
        "Select one sample image from the 10 Fashion-MNIST classes and the app will run inference for that choice. Each page refresh loads a new random example for every class."
    )

    session_samples = gr.State()
    gallery = gr.Gallery(
        value=[],
        label="Choose one class sample",
        columns=5,
        rows=2,
        allow_preview=False,
        interactive=True,
        height="auto",
    )
    refresh_button = gr.Button("Get new images")

    with gr.Row():
        selected_image = gr.Image(
            label="Selected image",
            interactive=False,
            height=260,
        )
        with gr.Column():
            true_label_text = gr.Textbox(label="True class", interactive=False)
        model_result_components = []
        for spec in MODEL_SPECS:
            with gr.Column():
                gr.Markdown(f"### {spec['title']} Results")
                predicted_class_text = gr.Textbox(
                    label="Predicted class",
                    interactive=False,
                )
                confidence_text = gr.Textbox(label="Confidence", interactive=False)
                top_3_text = gr.Textbox(
                    label="Top-3 predictions",
                    lines=3,
                    interactive=False,
                )
                inference_time_text = gr.Textbox(
                    label="Inference time",
                    interactive=False,
                )
                model_result_components.extend(
                    [
                        predicted_class_text,
                        confidence_text,
                        top_3_text,
                        inference_time_text,
                    ]
                )

    demo.load(
        fn=initialize_session,
        outputs=[
            session_samples,
            gallery,
            selected_image,
            true_label_text,
            *model_result_components,
        ],
    )

    gallery.select(
        fn=on_gallery_select,
        inputs=[session_samples],
        outputs=[
            selected_image,
            true_label_text,
            *model_result_components,
        ],
    )

    refresh_button.click(
        fn=initialize_session,
        outputs=[
            session_samples,
            gallery,
            selected_image,
            true_label_text,
            *model_result_components,
        ],
    )

    gr.Markdown(
        "This app loads every model listed in `MODEL_SPECS`, runs all inferences on the selected Fashion-MNIST sample, and displays the results side by side. To add another model later, add one more entry to `MODEL_SPECS`."
    )


if __name__ == "__main__":
    demo.launch()
